"""Tests for prompt construction and its gates.

The failures these target are all SILENT: a blind arm that sees the image, a
parity break that the KL then reports as perceptual distortion, or a gate that
cannot fail. The previous project shipped a D18 assertion written as
`tuple(...) and ()` -- vacuously true, structurally unable to catch the leak it
named. So every gate here is also shown to FIRE on a planted violation.
"""

from __future__ import annotations

import unittest

import ca21_prompts as P

PROBLEM = "Which city had the highest expatriate population in 2022? Options:\nA. Dubai\nB. Sharjah"
CAPTION = "A ridgeline plot titled 'Expatriate Population Trends'. Dubai's curve is highest throughout."


class TestSharedTail(unittest.TestCase):
    def test_tail_is_problem_then_instruction_single_space(self):
        """Their jinja is `{{ content | trim }} <instruction>` -- one space."""
        tail = P.shared_tail("  Q?  ")
        self.assertEqual(tail, "Q? " + P.SHARED_INSTRUCTION)

    def test_instruction_is_verbatim_vision_r1(self):
        """Byte-check against think_answer.jinja -- the Arm A anchor's prompt."""
        expected = (
            "You FIRST think about the reasoning process as an internal monologue "
            "and then provide the final answer. The reasoning process MUST BE "
            "enclosed within <think> </think> tags. The final answer MUST BE put "
            "in \\boxed{}."
        )
        self.assertEqual(P.SHARED_INSTRUCTION, expected)

    def test_instruction_is_evidence_agnostic(self):
        """It must survive the image->caption swap, so it may not name a modality.

        This is exactly why see_think.jinja ("analyzing an image/video to generate
        a detailed description") is unusable for the scored pair.
        """
        low = P.SHARED_INSTRUCTION.lower()
        for word in ("image", "picture", "photo", "video", "figure", "diagram"):
            self.assertNotIn(word, low, f"instruction mentions {word!r}")


class TestParity(unittest.TestCase):
    def test_parity_holds_for_the_real_builders(self):
        P.assert_parity(P.build_answerer_messages(CAPTION, PROBLEM),
                        P.build_sighted_messages(PROBLEM), PROBLEM)

    def test_both_scored_prompts_end_with_the_identical_tail(self):
        tail = P.shared_tail(PROBLEM)
        ans = P._text_parts(P.build_answerer_messages(CAPTION, PROBLEM))[0]
        sig = P._text_parts(P.build_sighted_messages(PROBLEM))[0]
        self.assertTrue(ans.endswith(tail))
        self.assertEqual(sig, tail)

    def test_stripping_the_evidence_recovers_the_sighted_text_exactly(self):
        """The operational definition of parity."""
        ans = P._text_parts(P.build_answerer_messages(CAPTION, PROBLEM))[0]
        sig = P._text_parts(P.build_sighted_messages(PROBLEM))[0]
        residue_removed = ans.split("\n\n", 1)[1]
        self.assertEqual(residue_removed, sig)

    def test_parity_gate_fires_on_a_planted_suffix(self):
        bad = P.build_answerer_messages(CAPTION, PROBLEM)
        bad[0]["content"][0]["text"] += " Be concise."
        with self.assertRaises(AssertionError):
            P.assert_parity(bad, P.build_sighted_messages(PROBLEM), PROBLEM)

    def test_parity_gate_fires_when_sighted_carries_extra_text(self):
        bad = P.build_sighted_messages(PROBLEM)
        bad[0]["content"].append({"type": "text", "text": "extra"})
        with self.assertRaises(AssertionError):
            P.assert_parity(P.build_answerer_messages(CAPTION, PROBLEM), bad, PROBLEM)

    def test_parity_gate_fires_if_the_evidence_label_changes(self):
        bad = P.build_answerer_messages(CAPTION, PROBLEM)
        bad[0]["content"][0]["text"] = bad[0]["content"][0]["text"].replace(
            P.CAPTION_PREAMBLE, "Here is what I saw:")
        with self.assertRaises(AssertionError):
            P.assert_parity(bad, P.build_sighted_messages(PROBLEM), PROBLEM)


class TestBlindness(unittest.TestCase):
    def test_answerer_carries_no_image(self):
        P.assert_blind(P.build_answerer_messages(CAPTION, PROBLEM))

    def test_blind_gate_fires_on_an_image_part(self):
        bad = P.build_answerer_messages(CAPTION, PROBLEM)
        bad[0]["content"].insert(0, {"type": "image"})
        with self.assertRaises(AssertionError):
            P.assert_blind(bad)

    def test_blind_gate_fires_on_a_textual_image_placeholder(self):
        """Vision-SR1-47K problems carry no <image> tokens (verified, 0/21), but a
        caption could contain one, and a template could render one."""
        bad = P.build_answerer_messages("A chart <image_1> here", PROBLEM)
        with self.assertRaises(AssertionError):
            P.assert_blind(bad)

    def test_sighted_carries_exactly_one_image(self):
        self.assertEqual(len(P._image_parts(P.build_sighted_messages(PROBLEM))), 1)


class TestCaptioner(unittest.TestCase):
    def test_captioner_sees_the_full_question_including_options(self):
        """S2: symmetry. The captioner must not be shown a different x than the
        arm its caption is evaluated on."""
        shown = " ".join(P._text_parts(P.build_captioner_messages(PROBLEM)))
        self.assertIn("A. Dubai", shown)
        self.assertIn("B. Sharjah", shown)

    def test_captioner_has_the_image(self):
        self.assertEqual(len(P._image_parts(P.build_captioner_messages(PROBLEM))), 1)

    def test_captioner_does_not_receive_the_answer_format_instruction(self):
        """q_cap must not demand \\boxed{} -- the caption is not an answer."""
        shown = " ".join(P._text_parts(P.build_captioner_messages(PROBLEM)))
        self.assertNotIn(P.SHARED_INSTRUCTION, shown)

    def test_caption_instruction_carries_its_load_bearing_clauses(self):
        c = P.CAPTION_INSTRUCTION
        self.assertIn("from your description alone", c)   # the sufficiency condition
        self.assertIn("relate to one another", c)         # Set 3's relations lesson
        self.assertIn("Do not give the answer", c)        # the single prohibition (S3)

    def test_caption_instruction_has_exactly_one_prohibition(self):
        """S3: over-restriction is the documented failure (Track T)."""
        low = P.CAPTION_INSTRUCTION.lower()
        self.assertEqual(low.count("do not give"), 1)
        self.assertNotIn("only describe", low)
        self.assertNotIn("do not infer", low)


class TestNoEvidence(unittest.TestCase):
    def test_no_evidence_is_exactly_the_shared_tail(self):
        """It must differ from both scored arms ONLY by removal of evidence."""
        msgs = P.build_no_evidence_messages(PROBLEM)
        self.assertEqual(P._text_parts(msgs), [P.shared_tail(PROBLEM)])
        self.assertEqual(P._image_parts(msgs), [])
        P.assert_no_evidence(msgs)

    def test_no_evidence_gate_fires_on_a_leaked_caption(self):
        bad = P.build_no_evidence_messages(PROBLEM)
        bad[0]["content"][0]["text"] = (
            f"{P.CAPTION_PREAMBLE}\nsomething\n\n" + bad[0]["content"][0]["text"])
        with self.assertRaises(AssertionError):
            P.assert_no_evidence(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
