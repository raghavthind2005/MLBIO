"""Tests for :mod:`virl_pool`.

Every fixture below is a real string taken from ``PAPO_ViRL39K_train`` (or an
exact-structure reproduction of a format measured there), not an invented
example. The traps these lock down were each found by inspecting the data
before the parser was written:

* ``A.`` is the dominant label style, ``(A)`` the minority.
* Image placeholders can appear *after* the final option.
* Unlabeled bare-line options exist and must be refused, not guessed at.
* Multi-image rows exist.

Run with stdlib only::

    python3 -m unittest discover -s caption_stage1_runs/code -v
"""

from __future__ import annotations

import unittest

from virl_pool import (
    ParseKind,
    build_pool,
    is_gradeable,
    manifest_hash,
    parse_problem,
    stem_leaks_options,
    strip_image_placeholders,
)

# --- real rows -------------------------------------------------------------

GEOMETRY_PAREN = (
    "<image>\nIn the figure, in quadrilateral ABCD, angle A is 70°. Quadrilateral ABCD is "
    "folded so that points D and C coincide with points F and E respectively (points F and E "
    "are both on the line AB). The fold line is MN. What is the measure of angle AMF? \n"
    "(A) 70°\n(B) 40°\n(C) 30°\n(D) 20°"
)

CHOICES_MARKER = "<image>\nIs the dotted line a line of symmetry?\nChoices:\n(A) yes\n(B) no"

DOT_STYLE = (
    "<image>\nWhich object is the largest in the diagram?\n"
    "A. the red cube\nB. the blue sphere\nC. the green cylinder\nD. the yellow cone"
)

# 73 rows in the sample had an image placeholder after the final option.
IMAGE_AFTER_OPTIONS = (
    "Which chart shows the steepest increase?\n"
    "A. the first chart\nB. the second chart\nC. the third chart\nD. the fourth chart\n<image>"
)

BARE_LINE_OPTIONS = (
    "<image>Female preflower is represented by label\nB\nD\nH\nC\n"
    "Please answer the question based on the options mentioned before."
)

FREE_FORM = "<image>\nHow many cans are there?"


class TestImagePlaceholders(unittest.TestCase):
    def test_removes_plain_and_indexed(self):
        self.assertEqual(strip_image_placeholders("<image>abc<image_4>"), "abc")

    def test_collapses_leftover_whitespace(self):
        self.assertEqual(strip_image_placeholders("<image>\n\n\n\nQ?"), "Q?")


class TestParseProblem(unittest.TestCase):
    def test_paren_style(self):
        p = parse_problem(GEOMETRY_PAREN)
        self.assertIs(p.kind, ParseKind.MCQ_LABELED)
        self.assertEqual(p.option_labels, ("A", "B", "C", "D"))
        self.assertEqual(p.option_texts, ("70°", "40°", "30°", "20°"))
        self.assertIn("What is the measure of angle AMF?", p.stem)
        self.assertNotIn("(A)", p.stem)

    def test_dot_style_is_supported(self):
        """The dominant style in the data (451 rows vs 90 for paren style)."""
        p = parse_problem(DOT_STYLE)
        self.assertIs(p.kind, ParseKind.MCQ_LABELED)
        self.assertEqual(p.option_labels, ("A", "B", "C", "D"))
        self.assertEqual(p.option_texts[0], "the red cube")
        for body in p.option_texts:
            self.assertNotIn(body, p.stem)

    def test_choices_marker_kept_out_of_options(self):
        p = parse_problem(CHOICES_MARKER)
        self.assertIs(p.kind, ParseKind.MCQ_LABELED)
        self.assertEqual(p.option_texts, ("yes", "no"))
        self.assertIn("Choices:", p.stem)  # marker belongs to the stem side

    def test_image_after_options_does_not_lose_the_placeholder(self):
        """The trap: naive 'cut to end of string' would delete the image."""
        p = parse_problem(IMAGE_AFTER_OPTIONS)
        self.assertIs(p.kind, ParseKind.MCQ_LABELED)
        self.assertEqual(p.n_image_placeholders, 1)
        self.assertEqual(p.option_labels, ("A", "B", "C", "D"))
        self.assertNotIn("the fourth chart", p.stem)

    def test_bare_line_options_are_refused(self):
        p = parse_problem(BARE_LINE_OPTIONS)
        self.assertIs(p.kind, ParseKind.UNPARSEABLE)
        self.assertIn("bare-line", p.reason)

    def test_free_form_has_nothing_stripped(self):
        p = parse_problem(FREE_FORM)
        self.assertIs(p.kind, ParseKind.NO_OPTIONS)
        self.assertEqual(p.stem, p.full_text)
        self.assertEqual(p.stem, "How many cans are there?")

    def test_non_canonical_labels_refused(self):
        """Options starting at B, not A, are not a run we can trust."""
        p = parse_problem("Q?\nB. one\nC. two\nD. three")
        self.assertIs(p.kind, ParseKind.UNPARSEABLE)
        self.assertIn("canonical", p.reason)

    def test_prose_letters_before_options_do_not_break_the_run(self):
        """Geometry prose ("at point E") emits stray labels before the options.

        Measured on real rows: the raw match list is ['E','A','B','C','D'].
        The trailing canonical run must still be found, or 7.7% of MCQ rows
        would leak their options into the stem.
        """
        raw = (
            "<image>\nDraw ray AO, intersecting BC at point E. Given AB=10 and the area "
            "of triangle ABE is 20, find the length of CE from the choices: "
            "A. 6, B. 5, C. 4, D. 3."
        )
        p = parse_problem(raw)
        self.assertIs(p.kind, ParseKind.MCQ_LABELED)
        self.assertEqual(p.option_labels, ("A", "B", "C", "D"))
        self.assertIn("intersecting BC at point E", p.stem)
        self.assertNotIn("B. 5", p.stem)

    def test_content_after_final_option_is_refused(self):
        p = parse_problem("Q?\nA. one\nB. two\nNow justify your reasoning at length.")
        self.assertIs(p.kind, ParseKind.UNPARSEABLE)
        self.assertIn("after final option", p.reason)

    def test_full_text_keeps_options_for_the_answerer(self):
        p = parse_problem(GEOMETRY_PAREN)
        self.assertIn("70°", p.full_text)
        self.assertIn("(A)", p.full_text)
        self.assertNotIn("<image>", p.full_text)


class TestInlineOptions(unittest.TestCase):
    """Inline single-line options -- 7.7% of letter-answer rows.

    Missing these silently handed the captioner the full option list, which is
    exactly the leak D18 exists to prevent.
    """

    INLINE_REAL = (
        "In trapezoid ABCD, where AD is parallel to BC, and AD + BC is 10 cm, E is the "
        "midpoint of AB, point F is on DC, and EF is parallel to AD, determine the length "
        "of EF. Please choose from the options provided: A. 5 cm B. 10 cm C. 20 cm D. Undetermined."
    )

    INLINE_CHOICES = (
        "<image>\nIn trapezoid ABCD, AD // BC, diagonals AC and BD intersect at O, AD=1, "
        "BC=4. What is the area ratio of triangle AOD to triangle BOC? "
        "Choices: A. 1/2 B. 1/4 C. 1/8 D. 1/16"
    )

    def test_inline_with_marker_is_parsed(self):
        p = parse_problem(self.INLINE_REAL)
        self.assertIs(p.kind, ParseKind.MCQ_LABELED)
        self.assertEqual(p.option_labels, ("A", "B", "C", "D"))
        self.assertNotIn("5 cm", p.stem)
        self.assertIn("determine the length of EF", p.stem)

    def test_inline_choices_marker(self):
        p = parse_problem(self.INLINE_CHOICES)
        self.assertIs(p.kind, ParseKind.MCQ_LABELED)
        self.assertEqual(p.option_labels, ("A", "B", "C", "D"))
        self.assertNotIn("1/16", p.stem)

    def test_ambiguous_prose_is_refused_not_silently_passed(self):
        """Two canonical labels with no marker are ambiguous -> refuse.

        Refusing costs 0.3% of rows (measured over 1,400 real rows) and is the
        safe direction: passing such a row through as NO_OPTIONS would hand the
        captioner an un-stripped option list.
        """
        prose = (
            "<image>\nConsider the path from point A. Then travel onward to point B. "
            "How far is the total journey?"
        )
        p = parse_problem(prose)
        self.assertIs(p.kind, ParseKind.UNPARSEABLE)
        self.assertIn("no trustworthy", p.reason)

    def test_single_stray_label_in_prose_is_free_form(self):
        """One label is not a run; ordinary prose must still parse as free-form."""
        p = parse_problem("<image>\nThe angle at point A. Then compute the area.")
        self.assertIs(p.kind, ParseKind.NO_OPTIONS)

    def test_lowercase_option_list_is_refused(self):
        """Lowercase labels are detected only to refuse (~0.07% of rows)."""
        raw = (
            "If ab is a constant value, what is the relationship that a and b satisfy? "
            "(a). a = 2b (b). a = 3b (c). a = 4b (d). a = 5b"
        )
        self.assertIs(parse_problem(raw).kind, ParseKind.UNPARSEABLE)

    def test_literal_backslash_n_delimiter(self):
        """~1.4% of rows carry literal backslash-n instead of real newlines."""
        raw = "What is the value?\\nChoices:\\nA. one\\nB. two\\nC. three\\nD. four"
        p = parse_problem(raw)
        self.assertIs(p.kind, ParseKind.MCQ_LABELED)
        self.assertEqual(p.option_labels, ("A", "B", "C", "D"))

    def test_image_option_rows_are_refused(self):
        """Options that ARE images cannot be answered blind from a caption."""
        raw = (
            "The graph of the quadratic function is approximately ( )\n"
            "A. \n\n<image><image_1>\nB. \n\n<image><image_2>\n"
            "C. \n\n<image><image_3>\nD. \n\n<image><image_4>"
        )
        p = parse_problem(raw)
        self.assertIs(p.kind, ParseKind.UNPARSEABLE)
        self.assertIn("options are images", p.reason)


class TestLeakDetection(unittest.TestCase):
    def test_clean_stem_has_no_leak(self):
        self.assertFalse(stem_leaks_options(parse_problem(DOT_STYLE)))

    def test_option_text_present_in_stem_is_flagged(self):
        leaky = "Is it the green cylinder shown here?\nA. the green cylinder\nB. something else"
        self.assertTrue(stem_leaks_options(parse_problem(leaky)))

    def test_short_bodies_do_not_false_positive(self):
        """'2' or 'no' occur naturally in stems; they cannot count as leaks."""
        p = parse_problem("How many are there, 2 or more?\nA. 2\nB. 3")
        self.assertFalse(stem_leaks_options(p))


class TestGradeability(unittest.TestCase):
    def test_self_match_required(self):
        self.assertTrue(is_gradeable("A", lambda a, b: a == b))
        self.assertFalse(is_gradeable("A", lambda a, b: False))

    def test_empty_answer_rejected(self):
        self.assertFalse(is_gradeable("   ", lambda a, b: True))

    def test_raising_grader_counts_as_ungradeable(self):
        def boom(a, b):
            raise ValueError("bad latex")

        self.assertFalse(is_gradeable("\\frac{1}{2}", boom))


def _row(i, problem=DOT_STYLE, answer="A", images=("img.jpg",)):
    return {"index": i, "problem": problem, "answer": answer, "images": list(images)}


class TestBuildPool(unittest.TestCase):
    def setUp(self):
        self.grade = lambda a, b: a == b

    def test_drops_and_counts_each_rejection_class(self):
        rows = [_row(i) for i in range(20)]
        rows.append(_row(100, images=("a.jpg", "b.jpg")))          # multi-image
        rows.append(_row(101, problem=BARE_LINE_OPTIONS))          # unparseable
        rows.append(_row(102, answer=""))                          # ungradeable
        items, subset, stats, rejects = build_pool(
            rows, self.grade, n_items=10, n_subset=3, seed=0
        )
        self.assertEqual(stats.n_raw, 23)
        self.assertEqual(stats.n_multi_image, 1)
        self.assertEqual(stats.n_unparseable, 1)
        self.assertEqual(stats.n_ungradeable, 1)
        self.assertEqual(stats.n_eligible, 20)
        self.assertEqual(len(rejects), 3)
        self.assertEqual({r["stage"] for r in rejects},
                         {"multi_image", "unparseable", "ungradeable"})

    def test_sampling_is_deterministic_under_seed(self):
        rows = [_row(i) for i in range(50)]
        a = build_pool(rows, self.grade, n_items=10, n_subset=3, seed=7)[0]
        b = build_pool(rows, self.grade, n_items=10, n_subset=3, seed=7)[0]
        self.assertEqual([i.index for i in a], [i.index for i in b])

    def test_sampling_independent_of_input_order(self):
        """Row order must not change the draw -- only the data and the seed."""
        rows = [_row(i) for i in range(50)]
        a = build_pool(rows, self.grade, n_items=10, n_subset=3, seed=7)[0]
        b = build_pool(list(reversed(rows)), self.grade, n_items=10, n_subset=3, seed=7)[0]
        self.assertEqual([i.index for i in a], [i.index for i in b])

    def test_subset_is_contained_in_sample(self):
        rows = [_row(i) for i in range(50)]
        items, subset, _, _ = build_pool(rows, self.grade, n_items=20, n_subset=5, seed=1)
        self.assertEqual(len(subset), 5)
        self.assertTrue(set(subset).issubset({i.index for i in items}))

    def test_too_small_pool_raises_rather_than_silently_shrinking(self):
        rows = [_row(i) for i in range(5)]
        with self.assertRaises(ValueError):
            build_pool(rows, self.grade, n_items=10, n_subset=2, seed=0)

    def test_manifest_hash_is_stable_and_content_sensitive(self):
        rows = [_row(i) for i in range(30)]
        a = build_pool(rows, self.grade, n_items=10, n_subset=2, seed=3)[0]
        b = build_pool(rows, self.grade, n_items=10, n_subset=2, seed=3)[0]
        self.assertEqual(manifest_hash(a), manifest_hash(b))
        c = build_pool(rows, self.grade, n_items=11, n_subset=2, seed=3)[0]
        self.assertNotEqual(manifest_hash(a), manifest_hash(c))


if __name__ == "__main__":
    unittest.main(verbosity=2)
