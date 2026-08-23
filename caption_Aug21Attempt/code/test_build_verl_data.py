"""Tests for the verl data adapter.

The failure this guards against is the worst kind available to us: a shard/row
misalignment silently pairing image A with question B. Every metric we log would still
look plausible, and the result would read as a perception failure -- which is the exact
phenomenon the project exists to measure. So the alignment gate is tested by planting a
misalignment and proving it fires.

pyarrow is not installed outside the container, so the parquet path is exercised on the
cluster (see runs/build_verl_data.sbatch). What is unit-tested here is the decision
logic: the hash agreement with build_pool, and every gate's ability to fail.
"""

from __future__ import annotations

import json
import unittest

import build_verl_data as B


def items(spec):
    """spec: list of (problem_id, path, problem, answer)."""
    return [{"problem_id": p, "path": q, "problem": r, "answer": s,
             "shard": "s0.parquet", "row_in_shard": i}
            for i, (p, q, r, s) in enumerate(spec)]


class FakeTable:
    """Just enough of a pyarrow Table for verify_split."""

    def __init__(self, d):
        self._d = d
        self.num_rows = len(d["problem_id"])

    def to_pydict(self):
        return self._d


def table_from(its, images=None):
    return FakeTable({
        "problem_id": [r["problem_id"] for r in its],
        "path": [r["path"] for r in its],
        "problem": [r["problem"] for r in its],
        "answer": [r["answer"] for r in its],
        "images": images if images is not None
        else [{"bytes": b"\x89PNG" + bytes([i]), "path": r["path"]}
              for i, r in enumerate(its)],
    })


BASE = items([(10, "Chart/a.png", "How many bars?", "3"),
              (11, "Math/b.png", "What is x?", "7"),
              (12, "Math/c.png", "Which is larger?", "B")])


class TestHashAgreesWithBuildPool(unittest.TestCase):
    def test_reproduces_build_pool_formula_exactly(self):
        """If these drift apart the provenance chain silently stops meaning anything."""
        splits = {"dev": BASE}
        expected = __import__("hashlib").sha256(json.dumps(
            [["dev", i["problem_id"], i["path"]] for i in BASE],
            sort_keys=True).encode()).hexdigest()
        self.assertEqual(B.manifest_selection_hash(splits), expected)

    def test_hash_is_sensitive_to_selection_changes(self):
        a = B.manifest_selection_hash({"dev": BASE})
        b = B.manifest_selection_hash({"dev": BASE[:2]})
        self.assertNotEqual(a, b)

    def test_hash_is_split_aware(self):
        a = B.manifest_selection_hash({"dev": BASE})
        b = B.manifest_selection_hash({"eval": BASE})
        self.assertNotEqual(a, b)


class TestVerifySplit(unittest.TestCase):
    def test_passes_on_a_faithful_table(self):
        out = B.verify_split(table_from(BASE), BASE, "dev")
        self.assertEqual(out, {"Chart/a.png", "Math/b.png", "Math/c.png"})

    def test_fires_on_row_count_mismatch(self):
        with self.assertRaises(AssertionError) as cm:
            B.verify_split(table_from(BASE[:2]), BASE, "dev")
        self.assertIn("2 rows for 3", str(cm.exception))

    def test_fires_on_problem_id_mismatch(self):
        bad = [dict(r) for r in BASE]
        bad[1]["problem_id"] = 999
        with self.assertRaises(AssertionError) as cm:
            B.verify_split(table_from(bad), BASE, "dev")
        self.assertIn("problem_id", str(cm.exception))

    def test_THE_ALIGNMENT_GATE_fires_on_shifted_text(self):
        """Ids and paths line up; the TEXT belongs to a different row.

        This is the off-by-one in `row_in_shard`. Without this gate the run trains on
        mismatched (image, question) pairs and reports it as a perception failure.
        """
        shifted = table_from(BASE)
        d = shifted.to_pydict()
        d["problem"] = [d["problem"][1], d["problem"][2], d["problem"][0]]
        with self.assertRaises(AssertionError) as cm:
            B.verify_split(shifted, BASE, "dev")
        msg = str(cm.exception)
        self.assertIn("problem mismatch at row", msg)
        self.assertIn("locators do not point where the manifest says", msg)

    def test_alignment_gate_also_covers_the_answer_column(self):
        t = table_from(BASE)
        t.to_pydict()["answer"] = ["3", "WRONG", "B"]
        with self.assertRaises(AssertionError) as cm:
            B.verify_split(t, BASE, "dev")
        self.assertIn("answer mismatch", str(cm.exception))

    def test_fires_on_a_multi_image_row(self):
        """R8, re-checked on the written artifact."""
        imgs = [{"bytes": b"x", "path": "a"}, {"bytes": b"y", "path": "b"},
                [{"bytes": b"z"}, {"bytes": b"w"}]]
        with self.assertRaises(AssertionError) as cm:
            B.verify_split(table_from(BASE, images=imgs), BASE, "dev")
        self.assertIn("R8", str(cm.exception))

    def test_fires_on_empty_image_bytes(self):
        imgs = [{"bytes": b"x", "path": "a"}, {"bytes": b"", "path": "b"},
                {"bytes": b"z", "path": "c"}]
        with self.assertRaises(AssertionError) as cm:
            B.verify_split(table_from(BASE, images=imgs), BASE, "dev")
        self.assertIn("empty image bytes", str(cm.exception))

    def test_manifest_order_independence(self):
        """verify_split sorts by problem_id, so manifest order must not matter."""
        shuffled = [BASE[2], BASE[0], BASE[1]]
        B.verify_split(table_from(BASE), shuffled, "dev")


if __name__ == "__main__":
    unittest.main(verbosity=2)
