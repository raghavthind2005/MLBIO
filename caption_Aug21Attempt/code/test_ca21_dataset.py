"""Tests for the question-text preservation gate.

The subclass itself needs verl+datasets, so it is exercised at T0. What is testable here is
`assert_problems_present` -- and it matters, because the failure it catches is silent: a
captioner prompt built from an empty question still yields a fluent caption and a finite
D-hat, so nothing downstream contradicts it.
"""

from __future__ import annotations

import unittest

from ca21_dataset import CA21_PROBLEM_KEY, assert_problems_present


class TestProblemsPresent(unittest.TestCase):
    def test_passes_when_present(self):
        b = {CA21_PROBLEM_KEY: ["q1", "q2"]}
        self.assertTrue(assert_problems_present(b, 2))

    def test_FIRES_when_the_plain_dataset_was_used(self):
        with self.assertRaises(AssertionError) as cm:
            assert_problems_present({"other": [1, 2]}, 2)
        self.assertIn("dataset.py:225", str(cm.exception))

    def test_FIRES_on_count_mismatch(self):
        with self.assertRaises(AssertionError):
            assert_problems_present({CA21_PROBLEM_KEY: ["q1"]}, 2)

    def test_FIRES_on_blank_questions(self):
        """The silent one: a blank question still produces a plausible caption."""
        with self.assertRaises(AssertionError) as cm:
            assert_problems_present({CA21_PROBLEM_KEY: ["q1", "   "]}, 2)
        self.assertIn("blank", str(cm.exception))


try:
    import jinja2 as _jinja2
    _has_jinja = True
except ImportError:                                   # laptop lacks it; container has it
    _has_jinja = False


class _FakeHFDataset:
    """Column access by name, row access by index -- the two shapes verl uses."""

    def __init__(self, problems):
        self._p = list(problems)

    def __getitem__(self, k):
        return list(self._p) if isinstance(k, str) else {"problem": self._p[k]}

    def __len__(self):
        return len(self._p)


class _FakeRLHF:
    """Enough of RLHFDataset for _build_messages: the attributes it reads."""

    image_key = "images"
    prompt_key = "problem"
    format_prompt = None

    def __init__(self):
        self.dataset = _FakeHFDataset(["q"])


class TestBuildMessagesAdaptation(unittest.TestCase):
    """The loader adaptation for Vision-SR1-47K's native schema.

    Both cases here are things the T0c pre-flight caught in production (job 3177011), and
    the second is the one worth testing forever: upstream places the image part only where
    the prompt contains a literal "<image>", our problems have none, and the result is a
    text-only prompt that trains a BLIND model while every metric looks ordinary.
    """

    def _cls(self):
        from ca21_dataset import make_ca21_dataset

        return make_ca21_dataset(_FakeRLHF)

    def _build(self, example, fmt=None):
        ds = self._cls()()
        ds.format_prompt = fmt
        return ds._build_messages(example), example

    def test_singular_image_is_normalised_to_a_list(self):
        """verl does len(images) and iterates it (dataset.py:230-235)."""
        sentinel = object()
        _, ex = self._build({"problem": "How many?", "images": sentinel})
        self.assertEqual(ex["images"], [sentinel],
                         "the singular image must be wrapped, in place, before "
                         "dataset.py:229 pops it")

    def test_existing_list_is_left_alone(self):
        a, b = object(), object()
        _, ex = self._build({"problem": "q", "images": [a, b]})
        self.assertEqual(ex["images"], [a, b])

    def test_image_part_present_without_any_marker(self):
        """The silent-catastrophe case: no "<image>" in the text, image still placed."""
        msgs, _ = self._build({"problem": "How many apples?", "images": object()})
        parts = msgs[0]["content"]
        self.assertEqual(sum(p["type"] == "image" for p in parts), 1)
        self.assertNotIn("<image>", msgs[0]["content"][1]["text"])

    @unittest.skipUnless(_has_jinja, "needs jinja2 (present in the container)")
    def test_format_prompt_is_rendered_into_the_text(self):
        msgs, _ = self._build({"problem": "  q  ", "images": object()},
                              fmt="{{ content | trim }} INSTRUCTION")
        self.assertEqual(msgs[0]["content"][1]["text"], "q INSTRUCTION")

    def test_text_is_unchanged_apart_from_the_template(self):
        """The adaptation must not alter the prompt text -- G-PARITY depends on it."""
        msgs, _ = self._build({"problem": "exact text", "images": object()})
        self.assertEqual(msgs[0]["content"][1]["text"], "exact text")

    def test_gate_rejects_a_text_only_prompt(self):
        from ca21_dataset import assert_exactly_one_image

        with self.assertRaises(AssertionError) as e:
            assert_exactly_one_image([{"role": "user", "content": [
                {"type": "text", "text": "no image here"}]}])
        self.assertIn("trains BLIND", str(e.exception))

    def test_gate_rejects_duplicate_images(self):
        from ca21_dataset import assert_exactly_one_image

        with self.assertRaises(AssertionError):
            assert_exactly_one_image([{"role": "user", "content": [
                {"type": "image"}, {"type": "image"},
                {"type": "text", "text": "q"}]}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
