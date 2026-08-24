"""Tokenised contexts for the caption pass and the blind pass.

THE RULE THIS MODULE FOLLOWS. Every tensor here is built the way
``verl/utils/dataset.py:222-313`` builds it, using **verl's own helpers**
(``VF.postprocess_data``, ``get_rope_index``, ``VF.get_response_mask``) rather than
hand-rolled equivalents. A hand-rolled padding or rope implementation that is 99% right
produces a model forward that runs, returns finite numbers, and is wrong -- which is the
one failure class this project cannot absorb, because the wrongness would appear as a
caption-distortion value and nothing downstream could reveal it.

THREE THINGS THAT ARE EASY TO GET WRONG HERE, ALL LOAD-BEARING.

1. **mrope.** For Qwen2-VL the dataset takes the ``get_rope_index`` branch whenever the
   PROCESSOR is present -- not whenever an image is present (``dataset.py:271-282``). So the
   blind context, which has no image, still needs 4-row position_ids with
   ``image_grid_thw=None``. Building 1-D position_ids for it would silently feed the model a
   different positional scheme in the blind pass than in the sighted pass, and the resulting
   KL would be a measurement of our own inconsistency.

2. **Left-padded prompt, right-padded response.** verl's layout is
   ``[left-pad][prompt][response][right-pad]`` (``vllm_rollout_spmd.py:252-254``), which is
   what makes ``[:, -T-1:-1]`` select the response logits. The blind sequence is assembled
   the same way so ``ca21_packing`` slices the same window for both contexts.

3. **G-BLIND is structural, not a check.** ``build_blind_batch`` never accepts images and
   returns no ``multi_modal_data`` key at all. There is no code path by which the blind
   context can carry the image, so the invariant cannot be broken by a future edit that
   forgets to call an assertion.
"""

from __future__ import annotations

from typing import Any


def _uses_mrope(processor) -> bool:
    """Mirror of dataset.py:271 -- keyed on the PROCESSOR, not on image presence."""
    return (
        processor is not None
        and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__
    )


def _position_ids(processor, tokenizer, input_ids, attention_mask, model_inputs):
    """Exactly dataset.py:271-289, including the text-only mrope case."""
    import torch

    if _uses_mrope(processor):
        if "Qwen3VLProcessor" in processor.__class__.__name__:
            from verl.models.transformers.qwen3_vl import get_rope_index
        else:
            from verl.models.transformers.qwen2_vl import get_rope_index

        vision_position_ids = get_rope_index(
            processor,
            input_ids=input_ids,
            image_grid_thw=model_inputs.get("image_grid_thw", None),
            video_grid_thw=model_inputs.get("video_grid_thw", None),
            second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
            attention_mask=attention_mask,
        )                                                    # (3, S)
        text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)   # (1, S)
        return torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, S)

    return torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)


def build_prompt_row(processor, tokenizer, messages, images, max_prompt_length: int,
                     min_pixels: int, max_pixels: int, truncation: str = "error") -> dict:
    """One tokenised prompt row, identical in shape to what the dataset yields.

    ``images=None`` gives a text-only row: no ``multi_modal_data`` key is produced, which is
    how G-BLIND is enforced structurally for the blind context.
    """
    import torch
    from verl.utils import torch_functional as VF
    from verl.utils.dataset import process_image

    out: dict[str, Any] = {}
    if images:
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        processed = [process_image(im, min_pixels, max_pixels) for im in images]
        model_inputs = processor(processed, [prompt], add_special_tokens=False,
                                 return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        out["multi_modal_data"] = {"images": images}
    else:
        # Chat template from the TOKENIZER, matching dataset.py:266 for the no-image branch.
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        model_inputs = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]

    position_ids = _position_ids(processor, tokenizer, input_ids, attention_mask,
                                 model_inputs)

    input_ids, attention_mask, position_ids = VF.postprocess_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_length=max_prompt_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,                      # dataset.py:299 -- prompts are LEFT padded
        truncation=truncation,
    )

    raw_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(raw_prompt_ids) > max_prompt_length:
        if truncation == "left":
            raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]
        elif truncation == "right":
            raw_prompt_ids = raw_prompt_ids[:max_prompt_length]
        else:
            raise RuntimeError(
                f"caption/blind prompt is {len(raw_prompt_ids)} tokens, over "
                f"{max_prompt_length}. Truncating it silently would change the prompt the "
                f"policy is scored on.")

    out["input_ids"] = input_ids
    out["attention_mask"] = attention_mask
    out["position_ids"] = position_ids
    out["raw_prompt_ids"] = raw_prompt_ids
    assert not (images is None and "multi_modal_data" in out), "G-BLIND: text-only row leaked an image"
    return out


def append_responses(prompt_input_ids, prompt_attention_mask, prompt_position_ids,
                     responses, response_mask):
    """Glue ``y`` onto a prompt to make the scoreable sequence verl's forward expects.

    Transcribed from ``vllm_rollout_spmd.py:243-258`` -- the same delta-position arithmetic
    generation itself uses, so the blind sequence is positionally identical in construction
    to the sighted one it is compared against. The ONLY difference between the two contexts
    must be the evidence; anything else shows up as distortion that is ours, not the model's.
    """
    import torch

    B, T = responses.shape
    device = prompt_position_ids.device

    delta = torch.arange(1, T + 1, device=device).view(1, -1).expand(B, -1)
    if prompt_position_ids.ndim == 3:                     # (B, 4, S) mrope
        delta = delta.view(B, 1, -1).expand(B, prompt_position_ids.size(1), -1)

    resp_position_ids = prompt_position_ids[..., -1:] + delta
    position_ids = torch.cat([prompt_position_ids, resp_position_ids], dim=-1)
    input_ids = torch.cat([prompt_input_ids, responses], dim=-1)
    attention_mask = torch.cat([prompt_attention_mask, response_mask], dim=-1)
    return input_ids, attention_mask, position_ids


def assert_contexts_comparable(sighted, blind, responses, response_length: int):
    """The two contexts must differ ONLY in evidence and must score the SAME y.

    Every clause here corresponds to a way the caption term could report a confident number
    while measuring nothing:

      * different trajectories  -> not a KL between two conditionals at all
      * blind carrying an image -> D collapses to ~0 and the run looks converged
      * response window mismatch-> the two passes score different token positions
    """
    B, T = responses.shape
    if T != response_length:
        raise AssertionError(
            f"responses have T={T} but response_length={response_length}; "
            f"ca21_packing slices [-T-1:-1] and would take the wrong window")
    for name, ctx in (("sighted", sighted), ("blind", blind)):
        if ctx["input_ids"].shape[0] != B:
            raise AssertionError(
                f"{name} batch {ctx['input_ids'].shape[0]} != {B} trajectories")
        if ctx["input_ids"].shape[-1] < T:
            raise AssertionError(f"{name} sequence shorter than the response it must score")
    if "multi_modal_data" in blind or "multi_modal_inputs" in blind:
        raise AssertionError(
            "G-BLIND violated: the blind context carries image data. Every caption would "
            "score ~0 distortion and the caption term would silently measure nothing.")
    return True
