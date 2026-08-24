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


if __name__ == "__main__":
    unittest.main(verbosity=2)
