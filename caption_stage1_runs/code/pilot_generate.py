"""Pilot 0, generation half. vLLM only -- no scoring, no HF model resident.

Three passes, writing JSONL that :mod:`pilot_score` later consumes:

  1. captions      G=5 per item, WITH image        -> captions.jsonl
  2. blind answers M=1 (M=3 on the subset), NO image -> answers_blind.jsonl
  3. image answers n=5 per item, WITH image        -> answers_image.jsonl

Kept separate from scoring so vLLM and HuggingFace never co-reside: the PAPO
line lost time to sleep/wake OOM when they did.

Sampling is temperature 1.0, top_p 1.0, top_k -1 -- untruncated, on BOTH roles
(D23). This is a correctness requirement, not a preference: the KL chain-rule
estimator requires trajectories drawn from the true policy, and GRPO's policy
gradient requires caption samples from pi_theta. Truncated sampling silently
biases both.

The reference distribution is never sampled here. It is scored, in pilot_score.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cs1_prompts as P  # noqa: E402

# Decision D24: 4,194,304 px at 1,024 px per visual token (patch 16, merge 2)
# => 4,096 visual tokens. Verified from the model's own preprocessor_config.
MAX_PIXELS = 4_194_304
MIN_PIXELS = 262_144
PX_PER_VISUAL_TOKEN = 1024

#: Sampling presets, defined HERE and imported by every other script, so the
#: numbers cannot drift between files (the same failure `_env.sh` was created
#: to stop). `untruncated` is D23 and the only one any training or scored run
#: may use; the rest are diagnostics.
PRESETS = {
    # D23. Required for the estimator: the chain-rule identity holds only for
    # y sampled from the true policy, so any truncation biases D-hat.
    "untruncated": dict(temperature=1.0, top_p=1.0, top_k=-1),
    # Qwen3-VL's own generation_config.json (identical on 2B and 4B). Recorded
    # for provenance; measured indistinguishable from untruncated at 4B
    # (job 3105710), so it is not a fallback, only a control.
    "model_card": dict(temperature=0.7, top_p=0.8, top_k=20),
    # Vision-SR1's rollout setting (`vision_sr1/config.yaml`): untruncated but
    # for a 1% tail clip. Tests whether long answers are tail-excursion
    # artifacts; the documented fallback if degeneration ever appears.
    "vision_sr1": dict(temperature=1.0, top_p=0.99, top_k=-1),
}

#: vLLM context window. Must exceed prompt + answer: an image may contribute up
#: to MAX_PIXELS // PX_PER_VISUAL_TOKEN = 4,096 visual tokens, so a 16k answer
#: budget does NOT fit the 16,384 default -- it silently clips instead.
DEFAULT_MAX_MODEL_LEN = 32_768


def load_images(manifest: dict, snapshot_dir: Path,
                only: "set[int] | None" = None) -> dict[int, "object"]:
    """Pull only the images the caller actually needs, by ``<shard>#<row>`` locator.

    The parquet embeds image bytes; we never copied them (2.7 GB) so they are
    fetched here.

    ``only`` restricts materialisation to a specific set of pool indices, and is
    **required at scan scale**. Without it this decodes every row in the
    manifest: for the 200-row pilot that is harmless, but the full candidate
    pool has ~27,000 rows, so each of the 8 scan shards would decode ~39 GB of
    PIL images to use one eighth of them -- 8x redundant work, on top of reading
    every parquet shard on every node. A 200-row smoke cannot expose this.
    """
    import pyarrow.parquet as pq
    from PIL import Image

    entries = [it for it in manifest["items"]
               if only is None or it["index"] in only]

    wanted: dict[str, dict[int, int]] = defaultdict(dict)  # shard -> {row: index}
    for it in entries:
        shard, row = it["image_paths"][0].split("#")
        wanted[shard][int(row)] = it["index"]

    images: dict[int, object] = {}
    for shard_name, rows in wanted.items():
        pf = pq.ParquetFile(snapshot_dir / shard_name)
        row_in_shard = 0
        for batch in pf.iter_batches(batch_size=256, columns=["images"]):
            col = batch.column("images").to_pylist()
            for cell in col:
                if row_in_shard in rows:
                    payload = cell[0]
                    raw = payload["bytes"] if isinstance(payload, dict) else payload
                    images[rows[row_in_shard]] = Image.open(io.BytesIO(raw)).convert("RGB")
                row_in_shard += 1
    missing = {it["index"] for it in entries} - set(images)
    if missing:
        raise RuntimeError(f"could not load {len(missing)} pool images, e.g. {sorted(missing)[:5]}")
    return images


def render(processor, messages: list[dict]) -> str:
    """Apply the chat template. Text only -- images are passed separately."""
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def check_pixel_budget(processor, image, verbose: bool = True) -> int:
    """G-PIXELS: prove the configured resolution cap is actually in force.

    The Qwen3-VL processor expresses its cap as ``size={longest_edge,
    shortest_edge}`` in pixels, not the older ``max_pixels``. Setting the wrong
    key fails silently and we would run at the model default (16,777,216 px,
    4x our intended budget) while believing D24 held. So the token count is
    measured rather than assumed.
    """
    out = processor.image_processor(images=[image], return_tensors="pt")
    grid = out["image_grid_thw"][0].tolist()
    merge = getattr(processor.image_processor, "merge_size", 2)
    n_tokens = (grid[0] * grid[1] * grid[2]) // (merge * merge)
    if verbose:
        print(f"  [G-PIXELS] grid_thw={grid} -> {n_tokens} visual tokens "
              f"(cap {MAX_PIXELS // PX_PER_VISUAL_TOKEN})", flush=True)
    return n_tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--g-captions", type=int, default=5)
    ap.add_argument("--m-answers", type=int, default=1)
    ap.add_argument("--m-answers-subset", type=int, default=3)
    ap.add_argument("--n-image-answers", type=int, default=5)
    ap.add_argument("--caption-max-tokens", type=int, default=1024)
    ap.add_argument("--answer-max-tokens", type=int, default=48)
    ap.add_argument("--limit-items", type=int, default=0, help="smoke: cap item count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", default="untruncated", choices=sorted(PRESETS),
                    help="sampling preset; D23 mandates 'untruncated' for anything scored")
    ap.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    args = ap.parse_args()

    # Fail before loading a model rather than silently clipping mid-run: the
    # image alone can occupy 4,096 tokens of the window.
    visual_cap = MAX_PIXELS // PX_PER_VISUAL_TOKEN
    need = max(args.caption_max_tokens, args.answer_max_tokens) + visual_cap
    if need >= args.max_model_len:
        raise SystemExit(
            f"--max-model-len {args.max_model_len} is too small: an image may use "
            f"{visual_cap} visual tokens and the largest generation budget is "
            f"{max(args.caption_max_tokens, args.answer_max_tokens)} "
            f"(need > {need}). Raise --max-model-len."
        )

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((Path(args.pool) / "pool_manifest.json").read_text())
    items = manifest["items"]
    subset = set(manifest["m3_subset_indices"])
    if args.limit_items:
        items = items[: args.limit_items]
        subset = {it["index"] for it in items[: max(1, args.limit_items // 4)]}
    by_index = {it["index"]: it for it in items}

    print(f"[setup] items={len(items)} subset={len(subset)}", flush=True)

    processor = AutoProcessor.from_pretrained(
        args.model,
        size={"longest_edge": MAX_PIXELS, "shortest_edge": MIN_PIXELS},
    )

    print("[setup] loading pool images", flush=True)
    images = load_images(manifest, Path(args.snapshot))
    images = {k: v for k, v in images.items() if k in by_index}
    print(f"[setup] images loaded: {len(images)}", flush=True)

    # G-PIXELS on the largest image we hold -- the one most likely to hit the cap.
    biggest = max(images.values(), key=lambda im: im.size[0] * im.size[1])
    print(f"[gate] largest image {biggest.size}", flush=True)
    n_tok = check_pixel_budget(processor, biggest)
    cap = MAX_PIXELS // PX_PER_VISUAL_TOKEN
    if n_tok > cap:
        raise AssertionError(f"G-PIXELS FAILED: {n_tok} visual tokens exceeds cap {cap}")

    # Gates on the prompts themselves, before a single token is generated.
    print("[gate] G-BLIND / G-PARITY / D18 over every item", flush=True)
    from virl_pool import parse_problem  # noqa: E402

    VISION_MARKERS = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")
    for it in items:
        cap_msgs = P.build_captioner_messages(it["stem"])
        ans_msgs = P.build_answerer_messages("PLACEHOLDER CAPTION", it["full_text"])
        ref_msgs = P.build_reference_messages(it["full_text"])
        P.assert_blind(ans_msgs)
        P.assert_parity(ans_msgs, ref_msgs, it["full_text"])

        # Option bodies are NOT stored in the manifest, so re-derive them from
        # full_text. An earlier version passed an empty tuple here, which made
        # this assertion silently vacuous -- it could never have caught a leak.
        bodies = parse_problem(it["full_text"]).option_texts
        P.assert_captioner_blind_to_options(cap_msgs, bodies)

        # Token-level blindness: the message-level check cannot see what the
        # chat template emits. If a vision marker survives templating, the
        # "blind" arm is not blind.
        rendered = render(processor, ans_msgs)
        for marker in VISION_MARKERS:
            if marker in rendered:
                raise AssertionError(f"G-BLIND: rendered answerer prompt contains {marker}")
    print(f"[gate] all {len(items)} items pass (incl. rendered-prompt vision check)", flush=True)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"size": {"longest_edge": MAX_PIXELS, "shortest_edge": MIN_PIXELS}},
        seed=args.seed,
        enforce_eager=False,
        disable_log_stats=True,
    )

    # Sampling. D23's `untruncated` is the DEFAULT and the only preset we train
    # under -- that is a correctness constraint (truncated sampling draws from
    # p-tilde != p, biasing every D-hat), not a preference. The other presets
    # exist so nuisance-parameter checks can be run on the same code path
    # instead of a fork of it; selecting one is a diagnostic act.
    #
    # No per-request seed. Reproducibility comes from LLM(seed=...); setting the
    # SAME SamplingParams.seed on every request would give every prompt an
    # identical random stream, correlating which tokens get drawn across items --
    # a subtle way to under-sample the policy while looking deterministic.
    preset = PRESETS[args.preset]
    print(f"[setup] sampling preset '{args.preset}': {preset}", flush=True)
    if args.preset != "untruncated":
        print("[setup] WARNING: off-D23 preset -- diagnostic only, not a training run",
              flush=True)

    def sp(n: int, max_tokens: int) -> "SamplingParams":
        return SamplingParams(n=n, max_tokens=max_tokens, **preset)

    # ---------------- pass 1: captions (with image) ----------------
    print(f"[pass 1] captions, G={args.g_captions}", flush=True)
    reqs = [{"prompt": render(processor, P.build_captioner_messages(it["stem"])),
             "multi_modal_data": {"image": images[it["index"]]}} for it in items]
    outs = llm.generate(reqs, sp(args.g_captions, args.caption_max_tokens))

    with (out / "captions.jsonl").open("w") as fh:
        for it, o in zip(items, outs):
            for k, cand in enumerate(o.outputs):
                fh.write(json.dumps({
                    "index": it["index"], "cap_k": k, "caption": cand.text,
                    "n_tokens": len(cand.token_ids), "finish_reason": cand.finish_reason,
                }) + "\n")
    captions = [json.loads(l) for l in (out / "captions.jsonl").read_text().splitlines()]
    trunc = sum(1 for c in captions if c["finish_reason"] == "length")
    print(f"[pass 1] {len(captions)} captions, truncated {trunc} ({trunc/len(captions):.1%})", flush=True)

    # ---------------- pass 2: blind answers (NO image) ----------------
    print(f"[pass 2] blind answers, M={args.m_answers} (subset M={args.m_answers_subset})", flush=True)
    blind_reqs, blind_meta = [], []
    for c in captions:
        it = by_index[c["index"]]
        msgs = P.build_answerer_messages(c["caption"], it["full_text"])
        P.assert_blind(msgs)
        m = args.m_answers_subset if c["index"] in subset else args.m_answers
        blind_reqs.append({"prompt": render(processor, msgs)})
        blind_meta.append((c["index"], c["cap_k"], m))

    # One request per distinct M so sampling counts stay honest.
    with (out / "answers_blind.jsonl").open("w") as fh:
        for m_val in sorted({m for _, _, m in blind_meta}):
            sel = [(r, meta) for r, meta in zip(blind_reqs, blind_meta) if meta[2] == m_val]
            if not sel:
                continue
            o = llm.generate([r for r, _ in sel], sp(m_val, args.answer_max_tokens))
            for (_, (idx, cap_k, _)), res in zip(sel, o):
                for j, cand in enumerate(res.outputs):
                    fh.write(json.dumps({
                        "index": idx, "cap_k": cap_k, "ans_j": j,
                        "answer": cand.text, "token_ids": list(cand.token_ids),
                        "finish_reason": cand.finish_reason,
                    }) + "\n")
    n_blind = sum(1 for _ in (out / "answers_blind.jsonl").open())
    print(f"[pass 2] {n_blind} blind answers", flush=True)

    # ---------------- pass 3: image-conditioned answers ----------------
    print(f"[pass 3] image answers, n={args.n_image_answers}", flush=True)
    ref_reqs = [{"prompt": render(processor, P.build_reference_messages(it["full_text"])),
                 "multi_modal_data": {"image": images[it["index"]]}} for it in items]
    outs = llm.generate(ref_reqs, sp(args.n_image_answers, args.answer_max_tokens))
    with (out / "answers_image.jsonl").open("w") as fh:
        for it, o in zip(items, outs):
            for j, cand in enumerate(o.outputs):
                fh.write(json.dumps({
                    "index": it["index"], "ans_j": j, "answer": cand.text,
                    "token_ids": list(cand.token_ids), "finish_reason": cand.finish_reason,
                }) + "\n")
    print(f"[pass 3] {len(items) * args.n_image_answers} image answers", flush=True)

    (out / "_meta_generate.json").write_text(json.dumps({
        "model": args.model,
        "pool_manifest_sha256": manifest["manifest_sha256"],
        "code_git_sha": os.environ.get("CS1_GIT_SHA", "unknown"),
        # Recorded from what actually ran. Previously hardcoded, which would
        # have logged untruncated even on a diagnostic preset -- provenance
        # that lies is worse than none.
        "sampling_preset": args.preset,
        "sampling": {**preset, "seed": args.seed},
        "max_model_len": args.max_model_len,
        "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS,
        "visual_tokens_largest_image": n_tok,
        "g_captions": args.g_captions, "m_answers": args.m_answers,
        "m_answers_subset": args.m_answers_subset, "n_image_answers": args.n_image_answers,
        "caption_max_tokens": args.caption_max_tokens,
        "answer_max_tokens": args.answer_max_tokens,
        "caption_truncation_rate": trunc / max(len(captions), 1),
        "n_items": len(items),
    }, indent=2))
    print("[done] wrote _meta_generate.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
