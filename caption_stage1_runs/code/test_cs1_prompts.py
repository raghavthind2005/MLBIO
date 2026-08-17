"""Tests for :mod:`cs1_prompts`.

The parity and blindness invariants are the reason this module exists, so the
tests deliberately try to break them: mutated suffixes, images smuggled into the
blind arm, captions that carry option text.

    python3 -m unittest test_cs1_prompts -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cs1_prompts as P

STEM = "Which object is the largest in the diagram?"
FULL = (
    "Which object is the largest in the diagram?\n"
    "A. the red cube\nB. the blue sphere\nC. the green cylinder\nD. the yellow cone"
)
CAPTION = "A red cube sits left of a much larger blue sphere. A small green cylinder is behind them."


class TestSharedTail(unittest.TestCase):
    def test_tail_ends_with_suffix(self):
        self.assertTrue(P.shared_tail(FULL).endswith(P.SHARED_SUFFIX))

    def test_tail_is_deterministic(self):
        self.assertEqual(P.shared_tail(FULL), P.shared_tail(FULL))

    def test_tail_is_whitespace_insensitive_at_edges(self):
        self.assertEqual(P.shared_tail(FULL), P.shared_tail("  " + FULL + "  \n"))


class TestCaptioner(unittest.TestCase):
    def test_has_exactly_one_image(self):
        msgs = P.build_captioner_messages(STEM)
        self.assertEqual(len(P._image_parts(msgs)), 1)

    def test_contains_stem_and_no_answer_instruction(self):
        text = " ".join(P._text_parts(P.build_captioner_messages(STEM)))
        self.assertIn(STEM, text)
        self.assertIn("Do not give the answer", text)
        self.assertNotIn(P.SHARED_SUFFIX, text)

    def test_option_leak_detected(self):
        leaky = P.build_captioner_messages(STEM + " Consider the green cylinder carefully.")
        with self.assertRaises(AssertionError):
            P.assert_captioner_blind_to_options(leaky, ("the green cylinder",))

    def test_clean_stem_passes_option_check(self):
        msgs = P.build_captioner_messages(STEM)
        P.assert_captioner_blind_to_options(msgs, ("the red cube", "the blue sphere"))

    def test_short_option_bodies_do_not_false_positive(self):
        msgs = P.build_captioner_messages("How many are there, 2 or more?")
        P.assert_captioner_blind_to_options(msgs, ("2", "3"))


class TestBlindness(unittest.TestCase):
    def test_answerer_has_no_image_part(self):
        msgs = P.build_answerer_messages(CAPTION, FULL)
        self.assertEqual(P._image_parts(msgs), [])
        P.assert_blind(msgs)

    def test_smuggled_image_part_is_caught(self):
        msgs = P.build_answerer_messages(CAPTION, FULL)
        msgs[0]["content"].insert(0, {"type": "image"})
        with self.assertRaises(AssertionError):
            P.assert_blind(msgs)

    def test_image_placeholder_in_text_is_caught(self):
        msgs = P.build_answerer_messages("<image> a red cube", FULL)
        with self.assertRaises(AssertionError):
            P.assert_blind(msgs)


class TestParity(unittest.TestCase):
    def setUp(self):
        self.ans = P.build_answerer_messages(CAPTION, FULL)
        self.ref = P.build_reference_messages(FULL)

    def test_parity_holds_by_construction(self):
        P.assert_parity(self.ans, self.ref, FULL)

    def test_both_end_with_the_identical_tail(self):
        tail = P.shared_tail(FULL)
        self.assertTrue(" ".join(P._text_parts(self.ans)).endswith(tail))
        self.assertTrue(" ".join(P._text_parts(self.ref)).endswith(tail))

    def test_reference_text_is_exactly_the_tail(self):
        self.assertEqual(" ".join(P._text_parts(self.ref)), P.shared_tail(FULL))

    def test_reference_carries_exactly_one_image(self):
        self.assertEqual(len(P._image_parts(self.ref)), 1)

    def test_divergent_suffix_is_caught(self):
        """The failure this gate exists for: a drifted suffix on one side."""
        bad = P.build_reference_messages(FULL)
        bad[0]["content"][1]["text"] = bad[0]["content"][1]["text"].replace(
            P.SHARED_SUFFIX, "Answer briefly."
        )
        with self.assertRaises(AssertionError):
            P.assert_parity(self.ans, bad, FULL)

    def test_extra_text_on_the_reference_side_is_caught(self):
        bad = P.build_reference_messages(FULL)
        bad[0]["content"].insert(1, {"type": "text", "text": "Think step by step."})
        with self.assertRaises(AssertionError):
            P.assert_parity(self.ans, bad, FULL)

    def test_answerer_residue_must_be_the_caption_span(self):
        bad = P.build_answerer_messages(CAPTION, FULL)
        bad[0]["content"][0]["text"] = "Extra preamble.\n\n" + bad[0]["content"][0]["text"]
        with self.assertRaises(AssertionError):
            P.assert_parity(bad, self.ref, FULL)

    def test_mismatched_question_between_arms_is_caught(self):
        other = P.build_reference_messages("A completely different question?")
        with self.assertRaises(AssertionError):
            P.assert_parity(self.ans, other, FULL)


class TestEvidenceIsTheOnlyDifference(unittest.TestCase):
    def test_removing_evidence_leaves_identical_strings(self):
        """The invariant stated directly: strip evidence, compare remainders."""
        ans = " ".join(P._text_parts(P.build_answerer_messages(CAPTION, FULL)))
        ref = " ".join(P._text_parts(P.build_reference_messages(FULL)))
        tail = P.shared_tail(FULL)
        ans_residue = ans[: -len(tail)]
        ref_residue = ref[: -len(tail)]
        self.assertEqual(ans[-len(tail):], ref[-len(tail):])   # tails identical
        self.assertEqual(ref_residue.strip(), "")              # reference evidence is the image
        self.assertIn(CAPTION, ans_residue)                    # answerer evidence is the caption

    def test_caption_content_never_reaches_the_reference(self):
        ref = " ".join(P._text_parts(P.build_reference_messages(FULL)))
        self.assertNotIn(CAPTION, ref)
        self.assertNotIn(P.CAPTION_PREAMBLE, ref)


class TestFrozenDocMatchesCode(unittest.TestCase):
    """The design doc must not specify prompt strings the code does not use.

    This has now drifted twice: PILOT_0_DESIGN.md carried a superseded
    ``SHARED_SUFFIX`` (the wording that was abandoned for driving 65-75%
    truncation) and a superseded caption instruction, while the code had moved
    on. A frozen document that disagrees with the code on a result-affecting
    string is worse than no document, because it is the one people read when
    deciding what the run *was*. Catch it in CI instead of by eye.
    """

    DOC = Path(__file__).resolve().parent.parent / "docs" / "PILOT_0_DESIGN.md"

    @staticmethod
    def _normalise(text: str) -> str:
        """Collapse whitespace so markdown re-wrapping is not a false positive."""
        return " ".join(text.split())

    def setUp(self):
        if not self.DOC.exists():
            self.skipTest(f"{self.DOC} not present")
        self.doc = self._normalise(self.DOC.read_text())

    def test_shared_suffix_is_the_one_in_the_doc(self):
        self.assertIn(self._normalise(P.SHARED_SUFFIX), self.doc,
                      "PILOT_0_DESIGN.md does not contain the live SHARED_SUFFIX")

    def test_caption_instruction_body_is_in_the_doc(self):
        body = P.CAPTION_INSTRUCTION.split("Question:")[0]
        for para in (p for p in body.split("\n\n") if p.strip()):
            self.assertIn(self._normalise(para), self.doc,
                          f"PILOT_0_DESIGN.md is missing q_cap text: {para[:60]!r}")

    def test_removed_clause_is_gone_from_the_live_instruction(self):
        """The user-approved removal must hold in the string that is actually sent.

        Deliberately asserts nothing about the doc: the doc *should* still name
        the clause when explaining why it was dropped. What must not happen is
        the clause surviving in `CAPTION_INSTRUCTION`. The doc's copy is kept
        honest by :meth:`test_caption_instruction_body_is_in_the_doc`, which
        requires it to match the live paragraphs.
        """
        self.assertNotIn("Describe only what can be seen in the image",
                         P.CAPTION_INSTRUCTION)


if __name__ == "__main__":
    unittest.main(verbosity=2)
