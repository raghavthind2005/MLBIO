"""Tests for the Pilot-0 grading half (Job A).

The graders are injected, so these run anywhere -- no container, no
``mathruler``. The fakes below mimic ``mathruler``'s contract closely enough to
exercise our logic: ``extract_boxed_content`` returns the innermost ``\\boxed{}``
content (and something useless when there is none), ``grade_answer`` does
normalised string equality.

The container's REAL behaviour on unboxed input is verified separately by
``verify_mathruler.sbatch`` -- these fakes encode our *assumption*, and an
assumption about a third-party function is exactly the thing that must be
checked against the real one rather than trusted.
"""

from __future__ import annotations

import unittest

from pilot_score import (
    fallback_extract,
    grade_strict,
    grade_with_fallback,
    has_boxed,
    score_records,
    wilson_ci,
    check_g_grade,
)

BOXED = __import__("re").compile(r"\\boxed\{([^{}]*)\}")


def fake_extract(response: str) -> str:
    """Stand-in for ``mathruler.grader.extract_boxed_content``."""
    m = BOXED.findall(response or "")
    if not m:
        # mathruler returns the input unchanged when there is no boxed span;
        # the point is that it does NOT yield a gradeable answer.
        return response or ""
    return m[-1].strip()


def fake_grade(pred: str, gold: str) -> bool:
    """Stand-in for ``mathruler.grader.grade_answer``: normalised equality."""
    return (pred or "").strip().lower() == (gold or "").strip().lower()


class TestHasBoxed(unittest.TestCase):
    def test_detects_boxed(self):
        self.assertTrue(has_boxed(r"so the answer is \boxed{3}"))

    def test_absent(self):
        self.assertFalse(has_boxed("the answer is 3"))

    def test_empty(self):
        self.assertFalse(has_boxed(""))


class TestStrict(unittest.TestCase):
    def test_correct(self):
        self.assertTrue(grade_strict(r"\boxed{3}", "3", fake_extract, fake_grade))

    def test_incorrect(self):
        self.assertFalse(grade_strict(r"\boxed{4}", "3", fake_extract, fake_grade))

    def test_unboxed_scores_wrong(self):
        """Vision-SR1's defining property: no box, no credit."""
        self.assertFalse(grade_strict("the answer is 3", "3", fake_extract, fake_grade))

    def test_last_box_wins(self):
        self.assertTrue(grade_strict(r"\boxed{9} ... actually \boxed{3}", "3",
                                     fake_extract, fake_grade))

    def test_grader_exception_scores_zero(self):
        def boom(_p, _g):
            raise ValueError("grader exploded")
        self.assertFalse(grade_strict(r"\boxed{3}", "3", fake_extract, boom))


class TestFallbackExtract(unittest.TestCase):
    def test_letter_whole_line(self):
        self.assertEqual(fallback_extract("reasoning...\nB", "letter"), "B")

    def test_letter_announced(self):
        self.assertEqual(fallback_extract("So the answer is C.", "letter"), "C")

    def test_letter_parenthesised(self):
        self.assertEqual(fallback_extract("Hence (D)", "letter"), "D")

    def test_letter_bold(self):
        self.assertEqual(fallback_extract("final: **A**", "letter"), "A")

    def test_article_a_is_not_an_answer(self):
        """The false positive that would silently inflate letter accuracy."""
        self.assertIsNone(fallback_extract("The shape forms a triangle.", "letter"))

    def test_word_starting_with_letter_not_matched(self):
        self.assertIsNone(fallback_extract("The answer is Apple pie.", "letter"))

    def test_numeric_whole_line(self):
        self.assertEqual(fallback_extract("work...\n42", "numeric"), "42")

    def test_numeric_announced(self):
        self.assertEqual(fallback_extract("Therefore the total is 17", "numeric"), "17")

    def test_numeric_last_resort_takes_final_number(self):
        self.assertEqual(fallback_extract("5 plus 7 gives 12", "numeric"), "12")

    def test_numeric_fraction(self):
        self.assertEqual(fallback_extract("ratio\n3/4", "numeric"), "3/4")

    def test_numeric_negative_and_decimal(self):
        self.assertEqual(fallback_extract("value\n-2.5", "numeric"), "-2.5")

    def test_only_final_line_is_mined(self):
        """Discarded intermediate values must not be harvested."""
        self.assertEqual(
            fallback_extract("First I thought 9\nThen 8\nFinally 7", "numeric"), "7")

    def test_letter_gets_no_cue_free_last_resort(self):
        """Unlike numerics, a stray trailing capital is not treated as a verdict."""
        self.assertIsNone(fallback_extract("Now consider triangle ABC D", "letter"))

    def test_empty(self):
        self.assertIsNone(fallback_extract("", "numeric"))
        self.assertIsNone(fallback_extract("   ", "letter"))


class TestGradeWithFallback(unittest.TestCase):
    def test_boxed_correct_no_fallback(self):
        ok, used = grade_with_fallback(r"\boxed{3}", "3", "numeric", fake_extract, fake_grade)
        self.assertTrue(ok)
        self.assertFalse(used)

    def test_boxed_wrong_is_never_rescued(self):
        """THE safety property: the fallback repairs format, never correctness.

        The response is boxed with 4 but the tail also contains the gold 3. If
        the fallback ran here it would flip a genuinely wrong answer to correct
        and manufacture accuracy out of nothing.
        """
        resp = r"I considered 3 but conclude \boxed{4}" + "\n3"
        ok, used = grade_with_fallback(resp, "3", "numeric", fake_extract, fake_grade)
        self.assertFalse(ok)
        self.assertFalse(used)

    def test_unboxed_recovered(self):
        ok, used = grade_with_fallback("the answer is 3", "3", "numeric",
                                       fake_extract, fake_grade)
        self.assertTrue(ok)
        self.assertTrue(used)

    def test_unboxed_recovered_but_wrong(self):
        ok, used = grade_with_fallback("the answer is 5", "3", "numeric",
                                       fake_extract, fake_grade)
        self.assertFalse(ok)
        self.assertTrue(used)

    def test_unboxed_unrecoverable(self):
        ok, used = grade_with_fallback("I cannot tell.", "C", "letter",
                                       fake_extract, fake_grade)
        self.assertFalse(ok)
        self.assertFalse(used)


class TestWilson(unittest.TestCase):
    def test_zero_n(self):
        self.assertEqual(wilson_ci(0, 0), (0.0, 0.0))

    def test_stays_in_unit_interval_at_extremes(self):
        """Where the normal approximation would report impossible bounds."""
        for k, n in ((0, 10), (10, 10), (0, 1), (1, 1)):
            lo, hi = wilson_ci(k, n)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)
            self.assertLessEqual(lo, hi)

    def test_brackets_the_point_estimate(self):
        lo, hi = wilson_ci(5, 10)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_narrows_with_n(self):
        w_small = (lambda t: t[1] - t[0])(wilson_ci(5, 10))
        w_large = (lambda t: t[1] - t[0])(wilson_ci(500, 1000))
        self.assertLess(w_large, w_small)


class TestScoreRecords(unittest.TestCase):
    def setUp(self):
        self.gold = {1: "3", 2: "A"}
        self.fmt = {1: "numeric", 2: "letter"}

    def _score(self, records):
        return score_records(records, self.gold, self.fmt, fake_extract, fake_grade)

    def test_histogram_counts_items_by_k_correct(self):
        records = [
            {"index": 1, "answer": r"\boxed{3}", "finish_reason": "stop"},
            {"index": 1, "answer": r"\boxed{4}", "finish_reason": "stop"},
            {"index": 2, "answer": r"\boxed{A}", "finish_reason": "stop"},
            {"index": 2, "answer": r"\boxed{A}", "finish_reason": "stop"},
        ]
        rep, _ = self._score(records)
        # item 1 -> 1 of 2 correct; item 2 -> 2 of 2.
        self.assertEqual(rep["strict"]["pass_rate_histogram"], {1: 1, 2: 1})
        self.assertEqual(rep["n_items"], 2)
        self.assertEqual(rep["draws_per_item"], [2])

    def test_fallback_never_below_strict(self):
        records = [
            {"index": 1, "answer": "the answer is 3", "finish_reason": "stop"},
            {"index": 2, "answer": r"\boxed{A}", "finish_reason": "stop"},
        ]
        rep, _ = self._score(records)
        self.assertGreaterEqual(rep["fallback_sensitivity"]["overall"]["accuracy"],
                                rep["strict"]["overall"]["accuracy"])
        self.assertAlmostEqual(rep["diagnostics"]["delta_fallback_minus_strict"], 0.5)

    def test_diagnostics_counts(self):
        records = [
            {"index": 1, "answer": r"\boxed{3}", "finish_reason": "stop"},
            {"index": 1, "answer": "the answer is 3", "finish_reason": "stop"},
            {"index": 2, "answer": "no idea whatsoever", "finish_reason": "length"},
            {"index": 2, "answer": r"\boxed{A}", "finish_reason": "stop"},
        ]
        rep, _ = self._score(records)
        d = rep["diagnostics"]
        self.assertEqual(d["boxed_rate"], 0.5)
        self.assertEqual(d["unboxed_count"], 2)
        self.assertEqual(d["fallback_recovered_count"], 1)
        self.assertEqual(d["fallback_unrecoverable_count"], 1)
        self.assertEqual(d["truncation_rate"], 0.25)

    def test_by_format_breakdown(self):
        records = [
            {"index": 1, "answer": r"\boxed{3}", "finish_reason": "stop"},
            {"index": 2, "answer": r"\boxed{B}", "finish_reason": "stop"},
        ]
        rep, _ = self._score(records)
        bf = rep["strict"]["by_format"]
        self.assertEqual(bf["numeric"]["accuracy"], 1.0)
        self.assertEqual(bf["letter"]["accuracy"], 0.0)

    def test_unknown_index_is_loud(self):
        with self.assertRaises(KeyError):
            self._score([{"index": 99, "answer": r"\boxed{3}", "finish_reason": "stop"}])

    def test_empty_is_loud(self):
        with self.assertRaises(ValueError):
            self._score([])


class TestGGrade(unittest.TestCase):
    def test_passes_on_self_gradeable(self):
        check_g_grade({1: "3", 2: "A"}, fake_grade)  # must not raise

    def test_fails_loudly(self):
        def never(_p, _g):
            return False
        with self.assertRaises(AssertionError):
            check_g_grade({1: "3"}, never)


if __name__ == "__main__":
    unittest.main(verbosity=2)
