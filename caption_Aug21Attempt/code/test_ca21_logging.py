"""Tests for the metrics that decide how R1 gets read.

The variance decomposition is tested against CONSTRUCTED cases with known answers, because
it is the one statistic that separates 'the mechanism did nothing' from 'we measured
nothing', and a plausible-looking wrong value would send the whole project down the wrong
branch. Pure Python, no torch, so it runs off-cluster.
"""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from ca21_logging import (  # noqa: E402
    advantage_component_metrics,
    distortion_metrics,
    dump_step_records,
    variance_decomposition,
)


def recs(per_prompt):
    """per_prompt: {uid: {caption_idx: [kl, ...]}}"""
    out = []
    for uid, caps in per_prompt.items():
        for ci, kls in caps.items():
            for ti, kl in enumerate(kls):
                out.append({"uid": uid, "caption_idx": ci, "traj_idx": ti, "kl": kl})
    return out


class TestVarianceDecomposition(unittest.TestCase):
    def test_pure_noise_gives_no_between_signal(self):
        """Captions identical in truth, all spread from trajectories: sigma2_between -> 0.

        This is the case that must NOT be flattered -- it is 'we measured nothing'."""
        random.seed(0)
        caps = {c: [random.gauss(0.0, 1.0) for _ in range(50)] for c in range(8)}
        out = variance_decomposition(recs({"p": caps}))
        self.assertLess(out["sigma2_between"], 0.1)
        self.assertGreater(out["sigma2_within"], 0.5)
        self.assertTrue(out["advantage_is_noise"])

    def test_pure_signal_is_recovered(self):
        """Captions genuinely differ, trajectories agree exactly: sigma2_within = 0."""
        caps = {c: [float(c)] * 5 for c in range(8)}
        out = variance_decomposition(recs({"p": caps}))
        self.assertAlmostEqual(out["sigma2_within"], 0.0, places=9)
        self.assertAlmostEqual(out["sigma2_between"], 6.0, places=6)  # var(0..7), ddof=1
        self.assertFalse(out["advantage_is_noise"])

    def test_the_bias_correction_is_actually_applied(self):
        """THE test. Var_j of the per-caption means is inflated by sigma2_within/m.

        Without subtracting it, a pure-noise population reports substantial 'between'
        variance -- overstating the signal in exactly the number used to decide whether
        the caption advantage is measurable at all.
        """
        random.seed(1)
        m = 4
        caps = {c: [random.gauss(0.0, 1.0) for _ in range(m)] for c in range(8)}
        out = variance_decomposition(recs({"p": caps}))
        means = [sum(v) / len(v) for v in caps.values()]
        mu = sum(means) / len(means)
        raw_between = sum((x - mu) ** 2 for x in means) / (len(means) - 1)
        # The uncorrected quantity is materially larger than the corrected one.
        self.assertGreater(raw_between, out["sigma2_between"])
        self.assertGreater(raw_between - out["sigma2_between"], 0.05)

    def test_never_negative(self):
        random.seed(2)
        caps = {c: [random.gauss(0.0, 1.0) for _ in range(3)] for c in range(4)}
        self.assertGreaterEqual(variance_decomposition(recs({"p": caps}))["sigma2_between"], 0.0)

    def test_singleton_caption_groups_are_skipped_not_counted(self):
        out = variance_decomposition(recs({"p": {0: [1.0, 2.0]}}))
        self.assertEqual(out["n_prompts"], 0)

    def test_multiple_prompts_are_averaged(self):
        caps = {c: [float(c)] * 4 for c in range(8)}
        out = variance_decomposition(recs({"p1": caps, "p2": caps}))
        self.assertEqual(out["n_prompts"], 2)
        self.assertAlmostEqual(out["mean_g_c"], 8.0)
        self.assertAlmostEqual(out["mean_m"], 4.0)


class TestAdvantageComponents(unittest.TestCase):
    def test_rms_of_a_standardised_group_matches_the_closed_form(self):
        """sum(z^2) = G-1 exactly under the n-1 convention, so RMS = sqrt((G-1)/G)."""
        import statistics as st

        vals = [1.0, 2.0, 3.0, 8.0]
        mu, sd = st.mean(vals), st.stdev(vals)
        z = [(v - mu) / sd for v in vals]
        out = advantage_component_metrics(z, z)
        self.assertAlmostEqual(out["adv_answer_rms"], (3 / 4) ** 0.5, places=6)

    def test_ratio_flags_a_scale_mismatch(self):
        out = advantage_component_metrics([1.0, -1.0], [2.0, -2.0])
        self.assertAlmostEqual(out["adv_ratio_caption_over_answer"], 2.0, places=6)

    def test_dead_group_fractions_are_separated(self):
        out = advantage_component_metrics([0.0, 0.0, 1.0, -1.0], [0.5, -0.5, 1.0, -1.0])
        self.assertAlmostEqual(out["adv_answer_zero_frac"], 0.5)
        self.assertAlmostEqual(out["adv_caption_zero_frac"], 0.0)


class TestDistortionMetrics(unittest.TestCase):
    def test_entropy_gap_is_blind_minus_sighted(self):
        out = distortion_metrics([1.0], [2.0], [5.0], [10])
        self.assertAlmostEqual(out["entropy_gap"], 3.0)

    def test_kl_oracle_counts_violations(self):
        out = distortion_metrics([-0.5, 0.1, -1e-6], [0.0] * 3, [0.0] * 3, [1] * 3)
        self.assertEqual(out["kl_oracle_violations"], 1)   # -1e-6 is inside tolerance


class TestDump(unittest.TestCase):
    def test_roundtrip(self):
        import json

        with tempfile.TemporaryDirectory() as d:
            rs = recs({"p": {0: [1.0, 2.0]}})
            p = dump_step_records(d, 7, rs)
            self.assertEqual(p.name, "step_00007.jsonl")
            back = [json.loads(x) for x in Path(p).read_text().splitlines()]
            self.assertEqual(len(back), 2)
            self.assertEqual(back[0]["uid"], "p")


if __name__ == "__main__":
    unittest.main(verbosity=2)
