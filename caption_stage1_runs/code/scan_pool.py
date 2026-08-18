"""Sharded full-pool scan: build the training pool by measuring each candidate.

TWO STAGES, IN THIS ORDER
-------------------------
::

    stage 1  mode=image      image + question, n=5, ALL candidates
                             -> keep image pass-rate >= 4/5      (D11/D32)
    stage 2  mode=text_only   question alone, n=5, SURVIVORS only
                             -> keep text pass-rate <= 2/5       (vision necessity)

**The order is deliberate and is not the cheaper one.** Screening on text first
would cost marginally less (1.53 vs 1.60 full passes), but ``image >= 4/5`` is
the *settled* criterion -- D11/D32, already approved, and mandatory because
training captions to reproduce the model's own wrong answers is precisely the
defect that disqualified ``D_perc`` under D3. The text threshold is
**provisional**: it rests on 3-31 items per bucket in Pilot 0 and is the number
most likely to move. Screening on the settled criterion first means a later
change to the text threshold costs a re-score, not a re-generation of rows we
never imaged.

WHY A TEXT-ONLY STAGE EXISTS AT ALL
-----------------------------------
Pilot 0 measured image=0.684 vs blind-from-caption=0.667 -- an apparent
information loss of +1.7 points, which looked like the objective had no
headroom. The no-evidence control showed why: **34% of ViRL39K rows are fully
solvable from the question text alone** (text-only 5/5), and on those the gap is
structurally zero no matter what the caption says. Excluding them, the caption
loses 12-17 points of the image's information -- real headroom. The text-only
stage exists to find and remove that dilution.

WHAT IS **NOT** SCANNED
-----------------------
The caption -> blind pass is **not** run over the pool. That is what the
training loop generates on the fly, and pre-filtering on it would mean selecting
rows by the very quantity the objective is meant to improve -- circular. It is
run only on a sample, to validate that the filter delivers the expected gap.

SHARDING
--------
Rows are independent, so shards are contiguous slices of the *index-sorted*
candidate list. Deterministic and gap-free: shard ``i`` of ``n`` takes
``candidates[i::n]`` -- strided rather than blocked, so every shard draws a mix
of easy and hard rows and they finish in comparable time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cs1_prompts as P  # noqa: E402
from pilot_generate import (  # noqa: E402
    DEFAULT_MAX_MODEL_LEN, MAX_PIXELS, MIN_PIXELS, PRESETS,
    PX_PER_VISUAL_TOKEN, load_images, render,
)
from text_only_control import assert_no_evidence, build_text_only_messages  # noqa: E402

MODES = ("image", "text_only")


def select_shard(items: list[dict], shard: int, n_shards: int) -> list[dict]:
    """Strided slice. Every shard gets a mix of row difficulties."""
    if not (0 <= shard < n_shards):
        raise ValueError(f"shard {shard} out of range for n_shards={n_shards}")
    return items[shard::n_shards]


def main() -> int:
    ap = argparse.ArgumentParser(description="Sharded pool scan")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--n-answers", type=int, default=5)
    ap.add_argument("--answer-max-tokens", type=int, default=16384)
    ap.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    ap.add_argument("--preset", default="untruncated", choices=sorted(PRESETS))
    ap.add_argument("--candidates", default="",
                    help="optional json list of indices to restrict to (stage 2 "
                         "runs only on stage-1 survivors)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    visual_cap = MAX_PIXELS // PX_PER_VISUAL_TOKEN
    need = args.answer_max_tokens + (visual_cap if args.mode == "image" else 0)
    if need >= args.max_model_len:
        raise SystemExit(
            f"--max-model-len {args.max_model_len} too small: need > {need}")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((Path(args.pool) / "pool_manifest.json").read_text())
    items = sorted(manifest["items"], key=lambda it: it["index"])

    if args.candidates:
        keep = set(json.loads(Path(args.candidates).read_text()))
        before = len(items)
        items = [it for it in items if it["index"] in keep]
        print(f"[setup] candidate restriction: {before} -> {len(items)}", flush=True)

    shard_items = select_shard(items, args.shard, args.n_shards)
    print(f"[setup] mode={args.mode} shard {args.shard}/{args.n_shards} "
          f"-> {len(shard_items)} of {len(items)} rows", flush=True)
    if not shard_items:
        print("[done] empty shard, nothing to do", flush=True)
        return 0

    processor = AutoProcessor.from_pretrained(
        args.model, size={"longest_edge": MAX_PIXELS, "shortest_edge": MIN_PIXELS}
    )

    reqs = []
    if args.mode == "image":
        # `only=` is essential here, not an optimisation: without it every shard
        # decodes the whole ~27,000-row candidate pool to use one eighth of it.
        images = load_images(manifest, Path(args.snapshot),
                             only={it["index"] for it in shard_items})
        missing = [it["index"] for it in shard_items if it["index"] not in images]
        if missing:
            raise SystemExit(f"missing images for {len(missing)} rows, e.g. {missing[:5]}")
        for it in shard_items:
            msgs = P.build_reference_messages(it["full_text"])
            reqs.append({"prompt": render(processor, msgs),
                         "multi_modal_data": {"image": images[it["index"]]}})
    else:
        for it in shard_items:
            msgs = build_text_only_messages(it["full_text"])
            assert_no_evidence(msgs, P.CAPTION_PREAMBLE)
            rendered = render(processor, msgs)
            for marker in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"):
                if marker in rendered:
                    raise AssertionError(f"text_only: rendered prompt contains {marker}")
            reqs.append({"prompt": rendered})
    print(f"[gate] {len(reqs)} prompts built for mode={args.mode}", flush=True)

    preset = PRESETS[args.preset]
    llm = LLM(model=args.model, trust_remote_code=True,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 1} if args.mode == "image" else {},
              mm_processor_kwargs={"size": {"longest_edge": MAX_PIXELS,
                                            "shortest_edge": MIN_PIXELS}},
              seed=args.seed, enforce_eager=False, disable_log_stats=True)

    outs = llm.generate(
        reqs, SamplingParams(n=args.n_answers,
                             max_tokens=args.answer_max_tokens, **preset))

    tag = f"{args.mode}_shard{args.shard:03d}of{args.n_shards:03d}"
    path = out / f"scan_{tag}.jsonl"
    with path.open("w") as fh:
        for it, o in zip(shard_items, outs):
            for j, cand in enumerate(o.outputs):
                fh.write(json.dumps({
                    "index": it["index"], "ans_j": j, "answer": cand.text,
                    "n_tokens": len(cand.token_ids),
                    "finish_reason": cand.finish_reason,
                }) + "\n")

    (out / f"_meta_{tag}.json").write_text(json.dumps({
        "mode": args.mode, "shard": args.shard, "n_shards": args.n_shards,
        "n_rows": len(shard_items), "n_answers": args.n_answers,
        "model": args.model,
        "pool_manifest_sha256": manifest.get("manifest_sha256"),
        "code_git_sha": os.environ.get("CS1_GIT_SHA", "unknown"),
        "sampling_preset": args.preset, "sampling": {**preset, "seed": args.seed},
        "answer_max_tokens": args.answer_max_tokens,
        "max_model_len": args.max_model_len,
    }, indent=2))
    print(f"[done] wrote {path} ({len(shard_items) * args.n_answers} answers)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
