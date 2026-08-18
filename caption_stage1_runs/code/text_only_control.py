"""No-evidence control: answer from the problem text alone.

WHY THIS EXISTS
---------------
Pilot 0 measured image-conditioned accuracy (a)=0.684 and blind-from-caption
accuracy (c)=0.667, an information loss of only +1.7 points. That was read as
"captions transfer nearly everything the image does".

**That reading is unsafe, because the design has no zero point.** Both arms
supply the model with the full problem statement, and a ViRL39K problem often
states its own quantities::

    Given that QR || TS, QT || RS, and m/_1 = 131 degrees, find m/_8      -> 49

180 - 131 = 49 needs no image and no caption. For such rows (a) and (c) are
*both* high for a reason that has nothing to do with perception, and their
difference is near zero no matter how good or bad the caption is.

This control supplies **no evidence at all** -- no image, no caption, only the
text every arm already shares. It converts an uninterpretable difference into a
decomposition::

    text_only        = what the question gives away by itself
    (c) - text_only  = what the CAPTION actually contributes
    (a) - text_only  = what the IMAGE actually contributes

If ``text_only`` lands near 0.66, then neither the image nor the caption is
doing meaningful work on this substrate, the +1.7 gap is a measurement of
nothing, and ViRL39K cannot support this experiment. If it lands near chance,
(c)=0.667 is real and the captions genuinely carry the scene.

PARITY
------
The prompt is exactly :func:`cs1_prompts.shared_tail` -- the identical string
both scored arms already end with -- so this arm differs from them **only** by
the removal of the evidence span. Any other wording would confound the control
with a prompt change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cs1_prompts as P  # noqa: E402
from pilot_generate import (  # noqa: E402
    DEFAULT_MAX_MODEL_LEN, MAX_PIXELS, MIN_PIXELS, PRESETS, render,
)


def build_text_only_messages(full_text: str) -> list[dict]:
    """The shared tail and nothing else: no image part, no caption span."""
    return [{"role": "user",
             "content": [{"type": "text", "text": P.shared_tail(full_text)}]}]


def assert_no_evidence(messages: list[dict], caption_preamble: str) -> None:
    """This arm must carry neither image nor caption. Verified, not assumed."""
    if P._image_parts(messages):
        raise AssertionError("control: message carries an image part")
    for text in P._text_parts(messages):
        if "<image" in text:
            raise AssertionError("control: text carries an <image> placeholder")
        if caption_preamble in text:
            raise AssertionError("control: caption evidence span leaked in")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-answers", type=int, default=5)
    ap.add_argument("--answer-max-tokens", type=int, default=16384)
    ap.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    ap.add_argument("--preset", default="untruncated", choices=sorted(PRESETS))
    ap.add_argument("--limit-items", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((Path(args.pool) / "pool_manifest.json").read_text())
    items = manifest["items"]
    if args.limit_items:
        items = items[: args.limit_items]
    print(f"[setup] items={len(items)} n_answers={args.n_answers}", flush=True)

    processor = AutoProcessor.from_pretrained(
        args.model, size={"longest_edge": MAX_PIXELS, "shortest_edge": MIN_PIXELS}
    )

    # Gate every prompt before any generation.
    prompts = []
    for it in items:
        msgs = build_text_only_messages(it["full_text"])
        assert_no_evidence(msgs, P.CAPTION_PREAMBLE)
        rendered = render(processor, msgs)
        for marker in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"):
            if marker in rendered:
                raise AssertionError(f"control: rendered prompt contains {marker}")
        prompts.append({"prompt": rendered})
    print(f"[gate] {len(items)} prompts carry no image and no caption", flush=True)

    preset = PRESETS[args.preset]
    print(f"[setup] sampling preset '{args.preset}': {preset}", flush=True)

    llm = LLM(model=args.model, trust_remote_code=True,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.85,
              seed=args.seed, enforce_eager=False, disable_log_stats=True)

    outs = llm.generate(
        prompts, SamplingParams(n=args.n_answers,
                                max_tokens=args.answer_max_tokens, **preset)
    )

    with (out / "answers_text_only.jsonl").open("w") as fh:
        for it, o in zip(items, outs):
            for j, cand in enumerate(o.outputs):
                fh.write(json.dumps({
                    "index": it["index"], "ans_j": j, "answer": cand.text,
                    "token_ids": list(cand.token_ids),
                    "finish_reason": cand.finish_reason,
                }) + "\n")

    (out / "_meta_text_only.json").write_text(json.dumps({
        "model": args.model,
        "pool_manifest_sha256": manifest.get("manifest_sha256"),
        "code_git_sha": __import__("os").environ.get("CS1_GIT_SHA", "unknown"),
        "sampling_preset": args.preset,
        "sampling": {**preset, "seed": args.seed},
        "n_items": len(items), "n_answers": args.n_answers,
        "answer_max_tokens": args.answer_max_tokens,
        "max_model_len": args.max_model_len,
        "arm": "text_only_no_evidence",
    }, indent=2))
    print(f"[done] {len(items) * args.n_answers} text-only answers", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
