"""Tests for the parts of the trainer fork that are pure.

Scope, stated honestly: `fit()`, caption sampling and scoring need a live Ray/FSDP/vLLM
stack and are validated by the T0 smoke, not here. What IS testable is the sign convention,
the uid-grouping gate, and the upstream drift pin -- and those are exactly the three places
where a bug would produce plausible numbers instead of an exception.
"""

from __future__ import annotations

import hashlib
import unittest

from ca21_trainer import (  # noqa: E402
    assert_uid_grouping,
    assert_upstream_fit_unchanged,
    caption_advantage,
)


class _Fake:
    """Stands in for RayPPOTrainer: only `fit`'s source text is read."""

    def fit(self):
        return 1


class TestUpstreamPin(unittest.TestCase):
    def test_returns_hash_when_no_expectation_is_given(self):
        got = assert_upstream_fit_unchanged(_Fake, None)
        self.assertEqual(len(got), 64)

    def test_passes_when_the_hash_matches(self):
        got = assert_upstream_fit_unchanged(_Fake, None)
        self.assertEqual(assert_upstream_fit_unchanged(_Fake, got), got)

    def test_FIRES_when_upstream_drifts(self):
        """The gate must fail on a changed fit(), not merely exist."""
        wrong = hashlib.sha256(b"some other version of fit").hexdigest()
        with self.assertRaises(AssertionError) as cm:
            assert_upstream_fit_unchanged(_Fake, wrong)
        self.assertIn("has changed", str(cm.exception))


class TestUidGrouping(unittest.TestCase):
    def test_accepts_complete_groups(self):
        uids = ["a", "a", "b", "b"]
        self.assertEqual(assert_uid_grouping(uids, 2, "t"), 2)

    def test_order_does_not_matter(self):
        """_balance_batch reorders rows; grouping must survive it (ray_trainer.py:597)."""
        self.assertEqual(assert_uid_grouping(["a", "b", "a", "b"], 2, "t"), 2)

    def test_FIRES_on_an_incomplete_group(self):
        with self.assertRaises(AssertionError) as cm:
            assert_uid_grouping(["a", "a", "b"], 2, "t")
        self.assertIn("do not have exactly 2", str(cm.exception))

    def test_FIRES_on_a_missing_uid(self):
        with self.assertRaises(AssertionError) as cm:
            assert_uid_grouping(["a", None], 1, "t")
        self.assertIn("no uid", str(cm.exception))


class TestCaptionAdvantageSign(unittest.TestCase):
    """THE sign test. Backwards, this trains captions to be maximally UNLIKE the sighted
    policy while every gate passes and D-hat rises smoothly."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not available off-cluster")

    def test_low_distortion_gets_POSITIVE_advantage(self):
        import torch

        d = torch.tensor([0.1, 0.9])          # first caption is the better one
        adv = caption_advantage(d, ["g", "g"], lam=1.0)
        self.assertGreater(adv[0].item(), 0.0)
        self.assertLess(adv[1].item(), 0.0)

    def test_lambda_scales_it(self):
        import torch

        d = torch.tensor([0.1, 0.9])
        a1 = caption_advantage(d, ["g", "g"], lam=1.0)
        a2 = caption_advantage(d, ["g", "g"], lam=2.0)
        self.assertAlmostEqual(a2[0].item(), 2.0 * a1[0].item(), places=5)

    def test_lambda_zero_disables_the_term_exactly(self):
        """Arm A must be recoverable by lam=0 alone, with no other code path."""
        import torch

        d = torch.tensor([0.1, 0.9, 0.5])
        adv = caption_advantage(d, ["g", "g", "g"], lam=0.0)
        self.assertTrue(torch.all(adv == 0.0))

    def test_groups_do_not_leak_into_each_other(self):
        import torch

        d = torch.tensor([0.1, 0.9, 100.0, 900.0])
        adv = caption_advantage(d, ["g1", "g1", "g2", "g2"], lam=1.0)
        # g2's huge raw scale must not swamp g1 -- that is what normalising per group is for
        self.assertAlmostEqual(adv[0].item(), adv[2].item(), places=5)


try:
    import torch as _torch
except ImportError:                                   # laptop has no torch; container does
    _torch = None


@unittest.skipUnless(_torch is not None, "needs torch")
class TestBlindRowLayout(unittest.TestCase):
    """The [g_c*N] -> [N, g_c] reshape in _ca21_step, and the worker's slice back out.

    Tested because getting it wrong is silent. The blind rows are BUILT caption-major
    (`for j: for k`), so index j*N + k. If the reshape does not invert exactly that order,
    caption j of row k is scored against a different row's sighted distribution `p` --
    the KL is still finite, the ladder still ranks, and nothing looks wrong.
    """

    @staticmethod
    def _build(g_c, N, S):
        """Mimic _ca21_step's build order, tagging each row with (j, k)."""
        flat = [_torch.full((S,), float(j * 100 + k))
                for j in range(g_c) for k in range(N)]
        t = _torch.stack(flat, dim=0)
        return t.reshape(g_c, N, *t.shape[1:]).transpose(0, 1).contiguous()

    def test_reshape_inverts_the_build_order(self):
        g_c, N, S = 3, 4, 5
        by_row = self._build(g_c, N, S)
        self.assertEqual(tuple(by_row.shape), (N, g_c, S))
        for j in range(g_c):
            for k in range(N):
                self.assertEqual(by_row[k, j, 0].item(), j * 100 + k,
                                 f"caption {j} of row {k} landed in the wrong slot")

    def test_worker_slice_recovers_caption_j_for_a_chunk(self):
        """blind[lo:hi, j] is what compute_caption_distortion forwards."""
        g_c, N, S = 4, 6, 3
        by_row = self._build(g_c, N, S)
        for lo, hi in ((0, 2), (2, 5), (0, 6)):
            for j in range(g_c):
                got = [v.item() for v in by_row[lo:hi, j][:, 0]]
                self.assertEqual(got, [j * 100 + k for k in range(lo, hi)])

    def test_row_and_its_captions_stay_together_under_dp_chunking(self):
        """The reason for this layout: a DP chunk must not split a caption group.

        Chunking dim 0 takes whole rows, so every caption of a row travels with it. The
        old flat [N*g_c] layout was chunked independently of sighted[N], which could put
        caption j of prompt p on a different rank from p's own sighted row.
        """
        g_c, N, S, world = 4, 8, 3, 4
        by_row = self._build(g_c, N, S)
        for r, chunk in enumerate(by_row.chunk(world, dim=0)):
            self.assertEqual(chunk.shape[1], g_c, "a chunk lost captions")
            rows_here = {int(chunk[i, 0, 0].item()) % 100 for i in range(chunk.shape[0])}
            for i in range(chunk.shape[0]):
                for j in range(g_c):
                    self.assertIn(int(chunk[i, j, 0].item()) % 100, rows_here)


if __name__ == "__main__":
    unittest.main(verbosity=2)
