"""Tests for the R2 group analysis.

The claim being corrected is that a marginal accuracy tells you whether GRPO has a
gradient. It does not. These tests plant the two extreme populations that share a
marginal rate and prove the instrument separates them -- because if it cannot, it is
worth nothing and we would be re-making the original error with more decimal places.
"""

from __future__ import annotations

import unittest

from format_check import group_analysis

N = 8


def recs(spec: list[tuple[int, str, int]]) -> tuple[list[dict], dict[int, str]]:
    """spec: (problem_id, category, n_correct_out_of_N) -> records + gold."""
    out, gold = [], {}
    for pid, cat, k in spec:
        gold[pid] = "RIGHT"
        for j in range(N):
            out.append({"problem_id": pid, "category": cat,
                        "text": "RIGHT" if j < k else "WRONG"})
    return out, gold


def ident(t):          # stand-in for extract_boxed_content
    return t


def grade(a, b):       # stand-in for grade_answer
    return a == b


class TestSeparatesPopulationsWithEqualMarginals(unittest.TestCase):
    """Both populations below sit at marginal 0.5. They are opposite verdicts."""

    def test_homogeneous_population_is_almost_all_live(self):
        r, g = recs([(i, "Math", 4) for i in range(20)])
        out = group_analysis(r, g, ident, grade, N)["overall"]
        self.assertAlmostEqual(out["marginal_accuracy"], 0.5)
        self.assertEqual(out["live_frac"], 1.0)
        self.assertEqual(out["dead_frac"], 0.0)

    def test_polarised_population_at_the_same_marginal_is_entirely_dead(self):
        r, g = recs([(i, "Math", N if i % 2 == 0 else 0) for i in range(20)])
        out = group_analysis(r, g, ident, grade, N)["overall"]
        self.assertAlmostEqual(out["marginal_accuracy"], 0.5)
        self.assertEqual(out["dead_frac"], 1.0, "every group agrees internally")
        self.assertEqual(out["live_frac"], 0.0)

    def test_the_marginal_alone_cannot_tell_them_apart(self):
        """The point of the whole instrument, asserted rather than assumed."""
        a, ga = recs([(i, "Math", 4) for i in range(20)])
        b, gb = recs([(i, "Math", N if i % 2 == 0 else 0) for i in range(20)])
        A = group_analysis(a, ga, ident, grade, N)["overall"]
        B = group_analysis(b, gb, ident, grade, N)["overall"]
        self.assertEqual(A["marginal_accuracy"], B["marginal_accuracy"])
        self.assertNotEqual(A["live_frac"], B["live_frac"])


class TestHeterogeneityGap(unittest.TestCase):
    def test_gap_is_near_zero_for_a_genuinely_iid_population(self):
        # p=0.5 homogeneous: i.i.d. predicts 2*0.5^8 = 0.0078 dead; observed 0.
        r, g = recs([(i, "Math", 4) for i in range(20)])
        out = group_analysis(r, g, ident, grade, N)["overall"]
        self.assertLess(abs(out["heterogeneity_gap"]), 0.02)

    def test_gap_is_large_and_positive_when_items_are_polarised(self):
        r, g = recs([(i, "Math", N if i % 2 == 0 else 0) for i in range(20)])
        out = group_analysis(r, g, ident, grade, N)["overall"]
        self.assertGreater(out["heterogeneity_gap"], 0.9)


class TestO4Reporting(unittest.TestCase):
    def test_at_least_one_correct_counts_items_not_draws(self):
        """O4 over m trajectories fires unless ALL m are wrong."""
        r, g = recs([(0, "Math", 0), (1, "Math", 1), (2, "Math", 8), (3, "Math", 3)])
        out = group_analysis(r, g, ident, grade, N)["overall"]
        self.assertAlmostEqual(out["at_least_one_correct"], 0.75)

    def test_histogram_records_the_shape(self):
        r, g = recs([(0, "Math", 0), (1, "Math", 0), (2, "Math", 5)])
        out = group_analysis(r, g, ident, grade, N)
        self.assertEqual(out["histogram"], {"0": 2, "5": 1})

    def test_per_category_blocks_are_independent(self):
        r, g = recs([(0, "Math", 4), (1, "Chart", 0), (2, "Chart", 8)])
        out = group_analysis(r, g, ident, grade, N)["by_category"]
        self.assertEqual(out["Math"]["live_frac"], 1.0)
        self.assertEqual(out["Chart"]["dead_frac"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
