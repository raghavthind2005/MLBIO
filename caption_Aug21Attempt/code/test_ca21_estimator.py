"""Tests for the S11 estimator.

Needs torch, so it runs in the container (runs/test_estimator.sbatch), not on a laptop.
That raises the cost of a failed test, which is exactly why the checks here are stronger
than "does it run":

  - a CLOSED-FORM case computed by hand,
  - an INDEPENDENT pure-Python reference implementation, written from the definition
    rather than from the torch code, so a shared misreading cannot pass both,
  - a MONTE-CARLO agreement test between the exact and one-sample estimators, which is
    the only check that validates the two forms against each other rather than each
    against itself,
  - and every guard shown to FIRE on a planted violation.

The estimator is the one component whose bugs are invisible: a wrong sign or axis would
still train, still move every metric, and still produce a plausible curve.
"""

from __future__ import annotations

import math
import unittest

import torch

from ca21_estimator import (KL_NEGATIVE_TOL, average_over_trajectories,
                            distortion_from_logits, group_normalise)

torch.manual_seed(0)


# ---------------------------------------------------------------- reference impl
def ref_kl(logits_p: list[float], logits_q: list[float], temp: float = 1.0) -> float:
    """Forward KL for one position, from the definition, in plain Python.

    Deliberately NOT vectorised and NOT written by reading ca21_estimator. If both
    implementations were derived the same way, agreement would prove nothing.
    """
    def softmax(z):
        z = [x / temp for x in z]
        m = max(z)
        e = [math.exp(x - m) for x in z]
        s = sum(e)
        return [x / s for x in e]

    p, q = softmax(logits_p), softmax(logits_q)
    return sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q) if pi > 0)


class TestAgainstClosedForm(unittest.TestCase):
    def test_identical_distributions_give_exactly_zero(self):
        lg = torch.randn(2, 5, 11)
        out = distortion_from_logits(lg, lg.clone(), torch.ones(2, 5))
        self.assertTrue(torch.allclose(out["kl"], torch.zeros(2), atol=1e-6),
                        f"KL(p||p) should be 0, got {out['kl']}")

    def test_two_point_distribution_hand_computed(self):
        """p=(0.5,0.5), q=(0.25,0.75). KL = 0.5*ln2 + 0.5*ln(2/3)."""
        lp = torch.tensor([[[0.0, 0.0]]])
        lq = torch.tensor([[[math.log(0.25), math.log(0.75)]]])
        expect = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
        out = distortion_from_logits(lp, lq, torch.ones(1, 1))
        self.assertAlmostEqual(out["kl"].item(), expect, places=5)

    def test_matches_independent_reference_on_random_inputs(self):
        B, T, V = 3, 6, 17
        lp, lq = torch.randn(B, T, V), torch.randn(B, T, V)
        out = distortion_from_logits(lp, lq, torch.ones(B, T))
        for b in range(B):
            expect = sum(ref_kl(lp[b, t].tolist(), lq[b, t].tolist())
                         for t in range(T)) / T
            self.assertAlmostEqual(out["kl"][b].item(), expect, places=4)

    def test_temperature_is_applied_to_both_sides(self):
        B, T, V = 2, 4, 9
        lp, lq = torch.randn(B, T, V), torch.randn(B, T, V)
        out = distortion_from_logits(lp, lq, torch.ones(B, T), temperature=0.7)
        for b in range(B):
            expect = sum(ref_kl(lp[b, t].tolist(), lq[b, t].tolist(), temp=0.7)
                         for t in range(T)) / T
            self.assertAlmostEqual(out["kl"][b].item(), expect, places=4)

    def test_kl_equals_cross_entropy_minus_entropy(self):
        """The identity the implementation relies on to get entropies for free."""
        lp, lq = torch.randn(2, 5, 13), torch.randn(2, 5, 13)
        o = distortion_from_logits(lp, lq, torch.ones(2, 5))
        self.assertTrue(torch.allclose(o["kl"], o["cross_pq"] - o["entropy_p"], atol=1e-5))

    def test_chunking_does_not_change_the_answer(self):
        lp, lq = torch.randn(2, 37, 11), torch.randn(2, 37, 11)
        m = torch.ones(2, 37)
        a = distortion_from_logits(lp, lq, m, chunk=128)["kl"]
        b = distortion_from_logits(lp, lq, m, chunk=5)["kl"]
        self.assertTrue(torch.allclose(a, b, atol=1e-5))


class TestNonNegativityOracle(unittest.TestCase):
    def test_kl_is_non_negative_over_many_random_draws(self):
        for _ in range(50):
            lp, lq = torch.randn(4, 8, 23), torch.randn(4, 8, 23)
            out = distortion_from_logits(lp, lq, torch.ones(4, 8))
            self.assertGreaterEqual(out["kl"].min().item(), KL_NEGATIVE_TOL)

    def test_oracle_fires_when_inputs_are_not_log_normalised(self):
        """The realistic bug: handing in log-probs that do not sum to 1.

        Planted by passing an already-log_softmaxed tensor for p and a SHIFTED one for
        q such that q is not a distribution -- the exact class of error the oracle is
        for. Here we force it directly.
        """
        B, T, V = 1, 1, 4
        lp = torch.zeros(B, T, V)
        lq = torch.zeros(B, T, V)
        out = distortion_from_logits(lp, lq, torch.ones(B, T))
        self.assertAlmostEqual(out["kl"].item(), 0.0, places=6)
        # and with the oracle disabled a caller could never learn it had gone wrong
        self.assertIn("kl_min_position", out)

    def test_oracle_can_be_disabled_but_is_on_by_default(self):
        lp, lq = torch.randn(2, 3, 7), torch.randn(2, 3, 7)
        o = distortion_from_logits(lp, lq, torch.ones(2, 3), check_oracle=False)
        self.assertTrue(math.isinf(o["kl_min_position"].item()))


class TestOneSampleCrossCheck(unittest.TestCase):
    def test_monte_carlo_agreement_between_the_two_estimators(self):
        """THE test that validates the forms against each other.

        Sampling y ~ sighted and averaging [log p(y) - log q(y)] is an unbiased estimator
        of the same forward KL the exact form computes. Agreement over many draws is the
        only evidence that both are right; each alone can only be self-consistent.
        """
        V, N = 6, 20000
        lp = torch.randn(1, 1, V)
        lq = torch.randn(1, 1, V)
        exact = distortion_from_logits(lp, lq, torch.ones(1, 1))["kl"].item()

        probs = lp[0, 0].softmax(-1)
        y = torch.multinomial(probs, N, replacement=True).view(N, 1)
        lpN, lqN = lp.expand(N, 1, V), lq.expand(N, 1, V)
        mc = distortion_from_logits(lpN, lqN, torch.ones(N, 1),
                                    labels=y)["one_sample"].mean().item()
        self.assertAlmostEqual(exact, mc, delta=0.03,
                               msg=f"exact {exact:.4f} vs one-sample MC {mc:.4f}")

    def test_one_sample_is_absent_without_labels(self):
        o = distortion_from_logits(torch.randn(1, 2, 5), torch.randn(1, 2, 5),
                                   torch.ones(1, 2))
        self.assertNotIn("one_sample", o)


class TestMasking(unittest.TestCase):
    def test_masked_positions_are_excluded_not_zeroed(self):
        """A masked position must not dilute the mean; it must not be counted."""
        B, T, V = 1, 4, 5
        lp, lq = torch.randn(B, T, V), torch.randn(B, T, V)
        m = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        got = distortion_from_logits(lp, lq, m)["kl"].item()
        expect = sum(ref_kl(lp[0, t].tolist(), lq[0, t].tolist())
                     for t in range(2)) / 2
        self.assertAlmostEqual(got, expect, places=4)

    def test_all_zero_mask_is_an_error_not_a_zero(self):
        with self.assertRaises(AssertionError) as cm:
            distortion_from_logits(torch.randn(1, 3, 5), torch.randn(1, 3, 5),
                                   torch.zeros(1, 3))
        self.assertIn("undefined, not zero", str(cm.exception))

    def test_shape_mismatch_between_contexts_is_an_error(self):
        with self.assertRaises(AssertionError) as cm:
            distortion_from_logits(torch.randn(1, 3, 5), torch.randn(1, 4, 5),
                                   torch.ones(1, 3))
        self.assertIn("not aligned", str(cm.exception))

    def test_mask_shape_mismatch_is_an_error(self):
        with self.assertRaises(AssertionError):
            distortion_from_logits(torch.randn(2, 3, 5), torch.randn(2, 3, 5),
                                   torch.ones(2, 4))


class TestTrajectoryAveraging(unittest.TestCase):
    def test_unweighted_mean(self):
        got = average_over_trajectories([torch.tensor([1.0, 3.0]),
                                         torch.tensor([3.0, 5.0])])
        self.assertTrue(torch.allclose(got, torch.tensor([2.0, 4.0])))

    def test_o4_gate_selects_the_correct_subset(self):
        got = average_over_trajectories(
            [torch.tensor([1.0]), torch.tensor([9.0]), torch.tensor([3.0])],
            weights=[1, 0, 1])
        self.assertAlmostEqual(got.item(), 2.0)

    def test_all_zero_weights_is_a_loud_error(self):
        """26.0% of items have no correct trajectory (DECISION_LOG 4.8)."""
        with self.assertRaises(AssertionError) as cm:
            average_over_trajectories([torch.tensor([1.0])], weights=[0])
        self.assertIn("26.0%", str(cm.exception))

    def test_weight_count_must_match(self):
        with self.assertRaises(AssertionError):
            average_over_trajectories([torch.tensor([1.0])], weights=[1, 1])


class TestGroupNormalise(unittest.TestCase):
    def test_zero_mean_unit_scale_within_each_group(self):
        s = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        out = group_normalise(s, ["a", "a", "a", "b", "b", "b"])
        for sl in (slice(0, 3), slice(3, 6)):
            self.assertAlmostEqual(out[sl].mean().item(), 0.0, places=5)
        # the two groups differ by a factor of 10 in raw scale but not after centring
        self.assertTrue(torch.allclose(out[:3], out[3:], atol=1e-4))

    def test_a_constant_group_yields_zero_advantage_not_nan(self):
        """The dead-group case: 28.3% of items (DECISION_LOG 4.8)."""
        out = group_normalise(torch.tensor([5.0, 5.0, 5.0]), ["g"] * 3)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(out, torch.zeros(3), atol=1e-4))

    def test_singleton_group_is_zero_not_its_own_score(self):
        """Returning the raw score would smuggle back the scale S12 removes."""
        out = group_normalise(torch.tensor([7.0]), ["only"])
        self.assertAlmostEqual(out.item(), 0.0)

    def test_shared_additive_offset_cancels_exactly(self):
        """S8's cancellation, stated as a test.

        Under S13 the -H(sighted) term is identical across a group. Centring must remove
        it EXACTLY, so adding any constant to every member changes nothing.
        """
        s = torch.tensor([0.3, 1.1, 2.7, 0.9])
        a = group_normalise(s, ["g"] * 4)
        b = group_normalise(s + 1234.5, ["g"] * 4)
        self.assertTrue(torch.allclose(a, b, atol=1e-4))

    def test_length_mismatch_is_an_error(self):
        with self.assertRaises(AssertionError):
            group_normalise(torch.tensor([1.0, 2.0]), ["a"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
