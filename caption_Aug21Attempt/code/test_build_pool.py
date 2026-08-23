"""Tests for the pool builder. Run locally -- `grade_fn` is injected so nothing
here needs the container's `mathruler`.

These target the properties that would fail SILENTLY in production: a split that
is off by two, an image that sneaks into two splits, a filter applied in the
wrong order, or a "seeded" draw that is not actually reproducible. Every one of
those produces a manifest that looks perfectly reasonable.
"""

from __future__ import annotations

import unittest

from build_pool import CATEGORIES, build, category_of, assert_invariants, derive_trial_smoke


def rows_for(spec: dict[str, int], per_image: int = 1,
             ptype: str = "multiple choice", answer: str = "A") -> list[dict]:
    """Synthesise rows: `spec` maps category -> number of distinct images."""
    out, pid = [], 0
    for cat, n_images in spec.items():
        for i in range(n_images):
            for k in range(per_image):
                out.append({
                    "problem_id": pid,
                    "problem_type": ptype,
                    "path": f"./{cat}/src/img_{i}.png",
                    "data_source": "src",
                    "answer": answer,
                    "shard": "train-0.parquet",
                    "row_in_shard": pid,
                })
                pid += 1
    return out


#: Proportional to the real on-disk distribution, so quotas exercise real ratios.
REALISTIC = {"Knowledge": 2520, "Math": 2480, "Spatial": 2180,
             "Chart": 1900, "General": 920}

ALWAYS = lambda a, b: True  # noqa: E731


class TestCategory(unittest.TestCase):
    def test_parses_leading_dot_slash(self):
        self.assertEqual(category_of("./Chart/MapQA/images/map_437.png"), "Chart")

    def test_parses_without_leading_dot(self):
        self.assertEqual(category_of("Math/GeoQA+/images/5642.png"), "Math")

    def test_rejects_pathless(self):
        with self.assertRaises(ValueError):
            category_of("map_437.png")


class TestSplitSizesExact(unittest.TestCase):
    """The defect the rewrite fixed: double rounding made split sizes drift."""

    def test_sizes_are_exact_not_approximate(self):
        sizes = {"trial": 5000, "eval": 1000, "dev": 300}
        splits, _ = build(rows_for(REALISTIC), ALWAYS, 0, sizes)
        for name, want in sizes.items():
            self.assertEqual(len(splits[name]), want, f"{name} drifted")

    def test_awkward_sizes_still_exact(self):
        sizes = {"trial": 777, "eval": 111, "dev": 37}
        splits, _ = build(rows_for(REALISTIC), ALWAYS, 3, sizes)
        for name, want in sizes.items():
            self.assertEqual(len(splits[name]), want)


class TestNoImageRepetition(unittest.TestCase):
    def test_no_image_twice_anywhere(self):
        rows = rows_for(REALISTIC, per_image=5)   # heavy reuse, as TabMWP has
        splits, _ = build(rows, ALWAYS, 0, {"trial": 500, "eval": 100, "dev": 30})
        assert_invariants(splits)                  # raises if violated
        paths = [r["path"] for v in splits.values() for r in v]
        self.assertEqual(len(paths), len(set(paths)))

    def test_invariant_catches_a_planted_duplicate(self):
        """A gate that cannot fail is not a gate."""
        rows = rows_for({"Chart": 50}, per_image=1)
        splits, _ = build(rows, ALWAYS, 0, {"trial": 10, "eval": 4, "dev": 2})
        splits["eval"].append(splits["trial"][0])   # same image, both splits
        with self.assertRaises(AssertionError):
            assert_invariants(splits)

    def test_collapse_keeps_one_row_per_image(self):
        rows = rows_for({"Chart": 100}, per_image=5)
        splits, rep = build(rows, ALWAYS, 0, {"trial": 60, "eval": 20, "dev": 10})
        self.assertEqual(rep["eligible_rows_after_filters"], 500)
        self.assertEqual(rep["eligible_distinct_images"], 100)


class TestFilters(unittest.TestCase):
    def test_regression_dropped_and_counted(self):
        # Spatial carries BOTH regression and non-regression rows, mirroring the
        # real data. An all-regression category would make the quota infeasible
        # rather than exercise the drop.
        rows = (rows_for({"Chart": 60, "Knowledge": 60, "Math": 60, "General": 60})
                + rows_for({"Spatial": 40})
                + rows_for({"Spatial": 40}, ptype="regression"))
        for i, r in enumerate(rows):          # keep ids unique after concatenation
            r["problem_id"] = i
            if r["problem_type"] == "regression":
                r["path"] = r["path"].replace("img_", "reg_")   # distinct images
        splits, rep = build(rows, ALWAYS, 0, {"trial": 30, "eval": 10, "dev": 5})
        self.assertEqual(rep["dropped_regression"], 40)
        self.assertEqual(rep["dropped_regression_by_category"], {"Spatial": 40})
        kept = [r for v in splits.values() for r in v]
        self.assertTrue(all(r["problem_type"] != "regression" for r in kept))

    def test_category_concentrated_filter_moves_the_target(self):
        """The defect the failing test exposed.

        Spatial is 50% of raw rows but only 33% of eligible ones, because every
        regression row is Spatial. Targeting `raw` must over-sample it relative
        to `eligible`; that gap is precisely what the default is chosen to avoid.
        """
        rows = (rows_for({"Chart": 100, "Knowledge": 100})
                + rows_for({"Spatial": 100})
                + rows_for({"Spatial": 200}, ptype="regression"))
        for i, r in enumerate(rows):
            r["problem_id"] = i
            if r["problem_type"] == "regression":
                r["path"] = r["path"].replace("img_", "reg_")
        sizes = {"trial": 60, "eval": 20, "dev": 10}

        _, rep = build(rows, ALWAYS, 0, sizes, target="eligible")
        self.assertAlmostEqual(rep["raw_share"]["Spatial"], 0.60, delta=0.01)
        self.assertAlmostEqual(rep["eligible_share"]["Spatial"], 1 / 3, delta=0.01)

        el, _ = build(rows, ALWAYS, 0, sizes, target="eligible")
        rw, _ = build(rows, ALWAYS, 0, sizes, target="raw")
        n_el = sum(1 for r in el["trial"] if category_of(r["path"]) == "Spatial")
        n_rw = sum(1 for r in rw["trial"] if category_of(r["path"]) == "Spatial")
        self.assertGreater(n_rw, n_el, "raw target must over-sample the shrunken category")

    def test_ungradeable_dropped(self):
        rows = rows_for({"Chart": 100})
        for r in rows[:30]:
            r["answer"] = "BAD"
        grade = lambda a, b: a != "BAD"  # noqa: E731
        _, rep = build(rows, grade, 0, {"trial": 40, "eval": 20, "dev": 5})
        self.assertEqual(rep["dropped_ungradeable"], 30)

    def test_filters_run_before_collapse(self):
        """An image survives if ANY of its rows is eligible.

        Collapsing first would pick a row at random and then discard the whole
        image when that row happened to be ungradeable -- losing images for no
        reason. Here every image has one good row and four bad ones, so all 50
        images must survive.
        """
        rows = rows_for({"Chart": 50}, per_image=5)
        for r in rows:
            r["answer"] = "OK" if r["problem_id"] % 5 == 0 else "BAD"
        grade = lambda a, b: a != "BAD"  # noqa: E731
        _, rep = build(rows, grade, 0, {"trial": 30, "eval": 10, "dev": 5})
        self.assertEqual(rep["eligible_distinct_images"], 50)

    def test_grader_exception_counts_as_ungradeable(self):
        rows = rows_for({"Chart": 60})
        def grade(a, b):
            if a == "A" and rows[0]["answer"] == "A":
                raise RuntimeError("grader blew up")
            return True
        with self.assertRaises(AssertionError):
            # every row ungradeable -> quota cannot be filled -> loud failure
            build(rows, grade, 0, {"trial": 30, "eval": 10, "dev": 5})


class TestStratification(unittest.TestCase):
    def test_drawn_shares_track_the_raw_distribution(self):
        rows = rows_for(REALISTIC)
        total = len(rows)
        sizes = {"trial": 5000, "eval": 1000, "dev": 300}
        splits, rep = build(rows, ALWAYS, 0, sizes)
        drawn = [r for v in splits.values() for r in v]
        for c in CATEGORIES:
            target = rep["raw_by_category"][c] / total
            got = sum(1 for r in drawn if category_of(r["path"]) == c) / len(drawn)
            self.assertAlmostEqual(got, target, delta=0.005,
                                   msg=f"{c}: drawn {got:.3f} vs target {target:.3f}")

    def test_every_split_is_stratified_not_just_the_union(self):
        rows = rows_for(REALISTIC)
        total = len(rows)
        splits, rep = build(rows, ALWAYS, 0, {"trial": 5000, "eval": 1000, "dev": 300})
        for name in ("trial", "eval"):
            for c in CATEGORIES:
                target = rep["raw_by_category"][c] / total
                got = sum(1 for r in splits[name]
                          if category_of(r["path"]) == c) / len(splits[name])
                self.assertAlmostEqual(got, target, delta=0.02,
                                       msg=f"{name}/{c}: {got:.3f} vs {target:.3f}")

    def test_infeasible_quota_fails_loudly(self):
        rows = rows_for({"Knowledge": 10, "Math": 10, "Spatial": 10,
                         "Chart": 10, "General": 10})
        with self.assertRaises(AssertionError):
            build(rows, ALWAYS, 0, {"trial": 500, "eval": 100, "dev": 30})


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_pool(self):
        rows = rows_for(REALISTIC, per_image=3)
        sizes = {"trial": 300, "eval": 100, "dev": 30}
        a, _ = build(rows, ALWAYS, 7, sizes)
        b, _ = build(rows, ALWAYS, 7, sizes)
        for k in sizes:
            self.assertEqual([r["problem_id"] for r in a[k]],
                             [r["problem_id"] for r in b[k]])

    def test_different_seed_different_pool(self):
        rows = rows_for(REALISTIC, per_image=3)
        sizes = {"trial": 300, "eval": 100, "dev": 30}
        a, _ = build(rows, ALWAYS, 7, sizes)
        b, _ = build(rows, ALWAYS, 8, sizes)
        self.assertNotEqual([r["problem_id"] for r in a["trial"]],
                            [r["problem_id"] for r in b["trial"]])

    def test_row_order_does_not_change_the_draw(self):
        """Parquet row order must not leak into the pool."""
        import random as _r
        rows = rows_for(REALISTIC, per_image=3)
        shuffled = rows[:]
        _r.Random(99).shuffle(shuffled)
        sizes = {"trial": 300, "eval": 100, "dev": 30}
        a, _ = build(rows, ALWAYS, 7, sizes)
        b, _ = build(shuffled, ALWAYS, 7, sizes)
        for k in sizes:
            self.assertEqual(sorted(r["path"] for r in a[k]),
                             sorted(r["path"] for r in b[k]))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestTrialSmoke(unittest.TestCase):
    """trial_smoke is a subset, so the disjointness machinery cannot vet it."""

    def _trial(self, n_per_cat=60):
        rows = []
        pid = 0
        for c in ("Chart", "General", "Knowledge", "Math", "Spatial"):
            for _ in range(n_per_cat):
                rows.append({"problem_id": pid, "path": f"./{c}/{pid}.png",
                             "category": c, "answer": "x", "problem": "q",
                             "data_source": "s", "problem_type": "multiple choice",
                             "shard": "s0.parquet", "row_in_shard": pid})
                pid += 1
        return rows

    def test_is_a_strict_subset_of_trial(self):
        trial = self._trial()
        smoke = derive_trial_smoke(trial, 50, seed=0)
        ids = {r["problem_id"] for r in trial}
        self.assertTrue({r["problem_id"] for r in smoke} <= ids)
        self.assertEqual(len(smoke), 50)

    def test_is_stratified_not_a_head_slice(self):
        """A head slice would be one category -- the job 3168166 failure."""
        trial = self._trial()
        smoke = derive_trial_smoke(trial, 50, seed=0)
        cats = {r["category"] for r in smoke}
        self.assertEqual(len(cats), 5, f"only {cats} represented")
        head = {r["category"] for r in trial[:50]}
        self.assertEqual(len(head), 1, "fixture should reproduce the head-slice hazard")

    def test_deterministic_under_seed(self):
        trial = self._trial()
        a = [r["problem_id"] for r in derive_trial_smoke(trial, 40, seed=0)]
        b = [r["problem_id"] for r in derive_trial_smoke(trial, 40, seed=0)]
        self.assertEqual(a, b)

    def test_oversized_request_is_a_loud_error(self):
        with self.assertRaises(AssertionError) as cm:
            derive_trial_smoke(self._trial(n_per_cat=2), 500, seed=0)
        self.assertIn("exceeds trial", str(cm.exception))
