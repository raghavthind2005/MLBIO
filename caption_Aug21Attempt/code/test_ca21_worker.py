"""Tests for the parts of the worker that CAN be tested without FSDP/Ray/flash-attn.

Honest scope: `compute_caption_distortion` itself needs a live worker and is validated by
the smoke, not here. What is testable is `build_distortion_batch` -- and it is worth
testing, because the failure it guards against is the quietest one in the whole design.

If the blind context ever received the image, `D` would collapse toward zero for every
caption. Training would proceed, the loss would look excellent, and the conclusion would
be "captions preserve reasoning behaviour perfectly" -- while nothing had been measured at
all. That is not a crash; it is a result. So G-BLIND is enforced at batch construction and
shown here to fire.
"""

from __future__ import annotations

import unittest


class T:
    """Minimal stand-in for a tensor: only `.shape` is used by the code under test."""

    def __init__(self, b, s=4):
        self.shape = (b, s)


def ctx(b=2):
    return {"input_ids": T(b), "attention_mask": T(b), "position_ids": T(b)}


def traj(b=2):
    return {"responses": T(b), "response_mask": T(b)}


from ca21_worker import build_distortion_batch  # noqa: E402


class TestBuildDistortionBatch(unittest.TestCase):
    def test_namespaces_both_contexts(self):
        out = build_distortion_batch(ctx(), ctx(), traj())
        for k in ("input_ids", "attention_mask", "position_ids"):
            self.assertIn(f"sighted_{k}", out)
            self.assertIn(f"blind_{k}", out)
        self.assertIn("responses", out)

    def test_contexts_are_not_merged_into_one_namespace(self):
        """Merging would let a missing tensor fall back to the other context --
        computing KL(sighted || sighted) = 0, which looks like a converged caption."""
        out = build_distortion_batch(ctx(), ctx(), traj())
        self.assertNotIn("input_ids", out)

    def test_G_BLIND_fires_when_the_blind_context_carries_an_image(self):
        """THE silent failure: D collapses to ~0 for every caption and the run looks
        like a triumph."""
        blind = ctx()
        blind["multi_modal_inputs"] = {"pixel_values": T(2)}
        with self.assertRaises(AssertionError) as cm:
            build_distortion_batch(ctx(), blind, traj())
        self.assertIn("G-BLIND", str(cm.exception))

    def test_empty_multi_modal_inputs_on_blind_is_allowed(self):
        blind = ctx()
        blind["multi_modal_inputs"] = {}
        build_distortion_batch(ctx(), blind, traj())

    def test_sighted_keeps_its_image(self):
        sighted = ctx()
        sighted["multi_modal_inputs"] = {"pixel_values": T(2)}
        out = build_distortion_batch(sighted, ctx(), traj())
        self.assertIn("sighted_multi_modal_inputs", out)

    def test_batch_size_mismatch_between_context_and_trajectory_is_an_error(self):
        with self.assertRaises(AssertionError) as cm:
            build_distortion_batch(ctx(2), ctx(2), traj(3))
        self.assertIn("SAME trajectories", str(cm.exception))

    def test_blind_batch_mismatch_is_also_caught(self):
        with self.assertRaises(AssertionError):
            build_distortion_batch(ctx(2), ctx(5), traj(2))

    def test_missing_context_tensor_names_what_is_missing(self):
        bad = ctx()
        del bad["position_ids"]
        with self.assertRaises(KeyError) as cm:
            build_distortion_batch(bad, ctx(), traj())
        self.assertIn("position_ids", str(cm.exception))

    def test_missing_trajectory_is_an_error(self):
        with self.assertRaises(KeyError) as cm:
            build_distortion_batch(ctx(), ctx(), {})
        self.assertIn("shared y", str(cm.exception))

    def test_response_mask_is_optional(self):
        out = build_distortion_batch(ctx(), ctx(), {"responses": T(2)})
        self.assertNotIn("response_mask", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
