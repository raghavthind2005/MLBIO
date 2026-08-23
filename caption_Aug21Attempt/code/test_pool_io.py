"""Tests for image-cell decoding.

The bug these exist for (job 3167490) was a schema assumption carried from a
different dataset: `images` was treated as a list of image structs, but
Vision-SR1-47K declares a singular `Image` feature, so the cell IS the struct and
`cell[0]` raised `KeyError: 0`. Cheap to catch here, 55 s of GPU to catch there.
"""

from __future__ import annotations

import collections
import unittest

from pool_io import extract_image_bytes, get_split

PNG = b"\x89PNG\r\n\x1a\n" + b"fake"


class TestExtractImageBytes(unittest.TestCase):
    def test_singular_image_struct(self):
        """Vision-SR1-47K's actual shape."""
        self.assertEqual(extract_image_bytes({"bytes": PNG, "path": "a.png"}), PNG)

    def test_list_of_one_struct(self):
        """ViRL39K's shape -- still supported so the loader is dataset-portable."""
        self.assertEqual(extract_image_bytes([{"bytes": PNG, "path": "a.png"}]), PNG)

    def test_raw_bytes(self):
        self.assertEqual(extract_image_bytes(PNG), PNG)

    def test_multi_image_row_is_a_hard_error_not_a_silent_first_pick(self):
        """R8: one image per item. Silently taking [0] would give `c` an
        undefined referent while every log line still looked healthy."""
        with self.assertRaises(AssertionError) as cm:
            extract_image_bytes([{"bytes": PNG}, {"bytes": PNG}])
        self.assertIn("R8", str(cm.exception))

    def test_empty_list_is_an_error(self):
        with self.assertRaises(AssertionError):
            extract_image_bytes([])

    def test_struct_without_bytes_names_its_keys(self):
        with self.assertRaises(AssertionError) as cm:
            extract_image_bytes({"path": "a.png"})
        self.assertIn("path", str(cm.exception))

    def test_unrecognised_type_is_an_error(self):
        with self.assertRaises(AssertionError):
            extract_image_bytes(42)


def _manifest(counts: dict[str, int]) -> dict:
    """A split stored the way build_pool writes it: categories grouped, not interleaved."""
    items, pid = [], 0
    for cat in sorted(counts):
        for _ in range(counts[cat]):
            items.append({"problem_id": pid, "category": cat,
                          "path": f"{cat}/{pid}.png", "answer": "x"})
            pid += 1
    return {"splits": {"dev": items}}


#: The real dev split that produced the Chart-only measurement in job 3168166.
DEV_COUNTS = {"Chart": 52, "General": 32, "Knowledge": 77, "Math": 91, "Spatial": 48}


class TestGetSplitSampling(unittest.TestCase):
    def test_head_slice_reproduces_the_job_3168166_bug(self):
        """Pin the failure itself, so the regression is a fact and not a memory.

        build_pool sorts each split by image path, which groups categories. A head slice
        of 50 therefore returned 50 Chart rows out of 300 while reporting '50 dev items'.
        """
        got = get_split(_manifest(DEV_COUNTS), "dev", 50, sample="head")
        self.assertEqual({c["category"] for c in got}, {"Chart"})

    def test_stratified_is_the_default(self):
        """The bug was a default, so the fix must be a default."""
        got = get_split(_manifest(DEV_COUNTS), "dev", 50)
        self.assertEqual(len(got), 50)
        self.assertEqual(set(c["category"] for c in got), set(DEV_COUNTS))

    def test_quotas_are_proportional_and_sum_exactly(self):
        got = get_split(_manifest(DEV_COUNTS), "dev", 50)
        seen = collections.Counter(c["category"] for c in got)
        self.assertEqual(sum(seen.values()), 50)
        for cat, n in DEV_COUNTS.items():
            self.assertAlmostEqual(seen[cat], n * 50 / 300, delta=1.0)

    def test_deterministic_under_seed_and_responsive_to_it(self):
        m = _manifest(DEV_COUNTS)
        a = [i["problem_id"] for i in get_split(m, "dev", 50, seed=0)]
        b = [i["problem_id"] for i in get_split(m, "dev", 50, seed=0)]
        c = [i["problem_id"] for i in get_split(m, "dev", 50, seed=7)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_limit_at_or_above_size_returns_everything_untouched(self):
        m = _manifest(DEV_COUNTS)
        self.assertIs(get_split(m, "dev", 300), m["splits"]["dev"])
        self.assertIs(get_split(m, "dev", 999), m["splits"]["dev"])
        self.assertIs(get_split(m, "dev", 0), m["splits"]["dev"])

    def test_returned_items_keep_manifest_order(self):
        """Downstream shard reads walk rows sequentially; scrambling costs re-scans."""
        m = _manifest(DEV_COUNTS)
        got = get_split(m, "dev", 50)
        self.assertEqual([i["problem_id"] for i in got],
                         sorted(i["problem_id"] for i in got))

    def test_awkward_limits_still_sum_exactly(self):
        """Largest remainder must not drift -- the same class of bug build_pool had."""
        m = _manifest(DEV_COUNTS)
        for limit in (1, 7, 13, 49, 101, 299):
            self.assertEqual(len(get_split(m, "dev", limit)), limit, f"limit={limit}")

    def test_unknown_sample_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            get_split(_manifest(DEV_COUNTS), "dev", 10, sample="random")


if __name__ == "__main__":
    unittest.main(verbosity=2)
