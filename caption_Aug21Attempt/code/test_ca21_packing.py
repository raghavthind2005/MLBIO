"""Tests for the packed -> [B, T, V] extraction.

Index bookkeeping fails in the worst possible way: it produces a tensor of exactly the
right shape containing the wrong rows. Shapes match, no exception fires, the KL is a
plausible positive number, and the run trains on nonsense. Nothing downstream can detect
it.

So the packed path is checked against a naive pad-then-slice reference on inputs where
the answer is known independently, including the ragged case that the reference and the
packed path could easily disagree about.

`unpad_input` is reimplemented here from its contract (keep positions where
attention_mask is 1, record flat indices) rather than imported, so these tests do not
depend on flash-attn being installed and do not inherit its behaviour as an assumption.
"""

from __future__ import annotations

import unittest

import torch

from ca21_packing import (gather_response_logits, gather_response_logits_reference,
                          response_slot_indices)

torch.manual_seed(0)


def fake_unpad(x, attention_mask):
    """The contract of flash-attn's unpad_input: keep unmasked rows, record flat idx."""
    B, S = attention_mask.shape
    flat = attention_mask.reshape(-1).bool()
    indices = torch.arange(B * S)[flat]
    packed = x.reshape(B * S, -1)[flat]
    return packed, indices


class TestWindow(unittest.TestCase):
    def test_window_matches_verls_slice_exactly(self):
        """verl slices [:, -T-1:-1]; the packed window must select the same s values."""
        B, S, T = 2, 10, 3
        am = torch.ones(B, S)
        _, indices = fake_unpad(torch.zeros(B, S, 1), am)
        sel, b_idx, t_idx = response_slot_indices(indices, B, S, T)
        s_sel = (indices % S)[sel]
        self.assertEqual(sorted(set(s_sel.tolist())), [6, 7, 8])   # S-T-1 .. S-2
        self.assertEqual(sorted(set(t_idx.tolist())), [0, 1, 2])

    def test_slot_zero_is_the_earliest_response_position(self):
        B, S, T = 1, 8, 4
        am = torch.ones(B, S)
        _, indices = fake_unpad(torch.zeros(B, S, 1), am)
        sel, _, t_idx = response_slot_indices(indices, B, S, T)
        s_sel = (indices % S)[sel]
        pairs = dict(zip(t_idx.tolist(), s_sel.tolist()))
        self.assertEqual(pairs[0], S - T - 1)
        self.assertEqual(pairs[T - 1], S - 2)


class TestAgreesWithNaiveReference(unittest.TestCase):
    def test_fully_dense_batch(self):
        B, S, V, T = 3, 12, 7, 4
        logits = torch.randn(B, S, V)
        am = torch.ones(B, S)
        packed, indices = fake_unpad(logits, am)

        got, mask = gather_response_logits(packed, indices, B, S, T)
        want, want_mask = gather_response_logits_reference(logits, am, T)

        self.assertEqual(tuple(got.shape), (B, T, V))
        self.assertTrue(torch.allclose(got, want),
                        "packed extraction returned different rows than pad-then-slice")
        self.assertTrue(torch.equal(mask, want_mask))

    def test_left_padded_prompts(self):
        """Real batches are left-padded to a common prompt length."""
        B, S, V, T = 2, 14, 5, 5
        logits = torch.randn(B, S, V)
        am = torch.ones(B, S)
        am[0, :4] = 0                      # sequence 0 has a shorter prompt
        packed, indices = fake_unpad(logits, am)

        got, mask = gather_response_logits(packed, indices, B, S, T)
        want, want_mask = gather_response_logits_reference(logits, am, T)
        self.assertTrue(torch.allclose(got, want))
        self.assertTrue(torch.equal(mask, want_mask))

    def test_RAGGED_responses_that_ended_early(self):
        """THE case the two paths could disagree about.

        A response that hit EOS early is padded on the right, so those positions are
        absent from the packed tensor entirely. They must come back as mask 0 -- not as
        a zero logit vector silently treated as valid, which is a UNIFORM distribution
        and would contribute a real KL term at every vocab entry.
        """
        B, S, V, T = 2, 12, 6, 5
        logits = torch.randn(B, S, V)
        am = torch.ones(B, S)
        am[1, -2:] = 0                     # sequence 1 stopped 2 tokens early
        packed, indices = fake_unpad(logits, am)

        got, mask = gather_response_logits(packed, indices, B, S, T)
        want, want_mask = gather_response_logits_reference(logits, am, T)

        self.assertTrue(torch.equal(mask, want_mask))
        self.assertEqual(mask[1].sum().item(), T - 1,
                         "one response slot should be invalid")
        # valid slots must match the reference; invalid ones must be exactly zero
        self.assertTrue(torch.allclose(got * mask.unsqueeze(-1),
                                       want * want_mask.unsqueeze(-1)))
        self.assertTrue(torch.equal(got[1, mask[1] == 0],
                                    torch.zeros((int((mask[1] == 0).sum()), V))))

    def test_many_random_masks(self):
        for _ in range(30):
            B, S, V, T = 3, 16, 5, 6
            logits = torch.randn(B, S, V)
            am = (torch.rand(B, S) > 0.25).float()
            am[:, -1] = 1                  # keep at least one valid slot per sequence
            packed, indices = fake_unpad(logits, am)
            got, mask = gather_response_logits(packed, indices, B, S, T)
            want, want_mask = gather_response_logits_reference(logits, am, T)
            self.assertTrue(torch.equal(mask, want_mask))
            self.assertTrue(torch.allclose(got * mask.unsqueeze(-1),
                                           want * want_mask.unsqueeze(-1)))


class TestGuards(unittest.TestCase):
    def test_rejects_unpacked_input(self):
        with self.assertRaises(AssertionError) as cm:
            gather_response_logits(torch.randn(2, 3, 4), torch.arange(6), 2, 3, 2)
        self.assertIn("packed logits", str(cm.exception))

    def test_rejects_mismatched_indices(self):
        with self.assertRaises(AssertionError) as cm:
            gather_response_logits(torch.randn(6, 4), torch.arange(5), 2, 3, 2)
        self.assertIn("same unpad_input call", str(cm.exception))


class TestEndToEndWithEstimator(unittest.TestCase):
    def test_two_contexts_with_different_prompt_lengths_align(self):
        """The actual use: sighted and blind have different packed layouts.

        Their prompts differ in length (image tokens vs caption text), so total_nnz
        differs and the packed tensors cannot be compared row-for-row -- but after
        extraction both are [B, T, V] on the same trajectory and the KL is well defined.
        """
        from ca21_estimator import distortion_from_logits

        B, V, T = 2, 9, 4
        S_sighted, S_blind = 20, 13          # different context lengths

        lg_s = torch.randn(B, S_sighted, V)
        am_s = torch.ones(B, S_sighted)
        am_s[0, :6] = 0
        p_s, i_s = fake_unpad(lg_s, am_s)

        lg_b = torch.randn(B, S_blind, V)
        am_b = torch.ones(B, S_blind)
        p_b, i_b = fake_unpad(lg_b, am_b)

        self.assertNotEqual(p_s.shape[0], p_b.shape[0], "layouts should differ")

        out_s, m_s = gather_response_logits(p_s, i_s, B, S_sighted, T)
        out_b, m_b = gather_response_logits(p_b, i_b, B, S_blind, T)
        self.assertEqual(out_s.shape, out_b.shape)

        # Only positions valid in BOTH contexts can be compared.
        mask = m_s * m_b
        res = distortion_from_logits(out_s, out_b, mask)
        self.assertTrue(torch.isfinite(res["kl"]).all())
        self.assertGreaterEqual(res["kl"].min().item(), -1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
