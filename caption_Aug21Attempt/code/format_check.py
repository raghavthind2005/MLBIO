"""Does Qwen2.5-VL-3B-Instruct actually comply with the sighted prompt?

This is the ONE premise behind S7 that rests on inference rather than
measurement. Vision-SR1 trains this backbone from base with this exact prompt and
gets 35.5 -> 47.1, which implies non-zero compliance (GRPO cannot bootstrap from
a group that scores zero everywhere) -- but that is an inference from their
training curve, not a number from our model on our pool. The previous project
asserted "answers are ~8 tokens" and measured 3,072.

THE TWO REQUIREMENTS ARE NOT EQUALLY LOAD-BEARING, so they are reported apart:

  \\boxed{}      ESSENTIAL. R(y) is grade_answer(extract_boxed_content(...)).
                 No box, no reward -- J_success would be scoring formatting
                 rather than perception.
  <think></think> INHERITED CONVENTION. Qwen2.5-VL has no native think tokens, so
                 these are pure instruction-following. Nothing in the design needs
                 them: the chain is whatever precedes the box, and \\boxed{}
                 already locates the answer span if the KL is later restricted to
                 it. They matter only if a format reward is added.

Reporting one merged "format rate" would hide exactly the distinction that
decides whether the prompt is usable.

Accuracy is reported but is NOT a selection criterion. Choosing a prompt on
accuracy optimises the control arm and contaminates the Arm A anchor.

Sampling is Vision-SR1's rollout setting (temperature 1.0, top_p 0.99) rather
than the model card's temperature 1e-6. The card is effectively greedy, which
would make every rollout in a GRPO group identical -- zero variance, zero
advantage, no gradient. It is also unrepresentative of training-time behaviour,
which is what this check exists to characterise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ca21_prompts as P  # noqa: E402
from container_gate import assert_vit_attn_patch  # noqa: E402
from pool_io import check_pixel_budget, get_split, load_images, load_manifest  # noqa: E402

# Vision-SR1's config.yaml: max_pixels 4194304, max_model_len 16896.
MAX_PIXELS = 4_194_304
MIN_PIXELS = 3_136          # Qwen2.5-VL-3B-Instruct preprocessor default
MAX_MODEL_LEN = 16_896
SAMPLING = dict(temperature=1.0, top_p=0.99)

#: Qwen2.5-VL's VISION tower has head_dim = 1280/16 = 80, and the flash_attn build
#: in this container refuses any head_dim that is not a multiple of 32:
#:
#:   RuntimeError: This flash attention build does not support headdim not being
#:   a multiple of 32.        (jobs 3167519 AND 3167568, in vit_flash_attn_wrapper)
#:
#: Setting this alone is NOT sufficient. vLLM 0.11.2 accepts the value, echoes it
#: back in its own non-default-args log line, and then reverts it internally: the
#: CUDA branch of maybe_get_vit_flash_attn_backend never consults
#: attn_backend_override, though the ROCm branch ten lines above does. That is why
#: job 3167568 failed IDENTICALLY to 3167519 despite carrying this setting.
#:
#: It works only alongside the patched layer.py mounted by runs/ca21_vllm.toml, and
#: gate G-VITATTN below proves the pair is actually in force rather than trusting
#: the log line that misled us the first time.
#:
#: RESULT-NEUTRAL: SDPA and FlashAttention both compute exact softmax attention;
#: they differ in kernel and summation order, so outputs differ only at rounding
#: level. TORCH_SDPA is also vLLM's own class default for Qwen2_5_VisionAttention.
#: Full rationale: patches/vllm_0_11_2/README.md
MM_ENCODER_ATTN_BACKEND = "TORCH_SDPA"

#: sha256 of our patched layer.py, so G-VITATTN can check identity as well as
#: behaviour. Kept in sync with patches/vllm_0_11_2/PATCHED.sha256 by a test.
PATCHED_LAYER_SHA256 = "d47643a080a09be7db1b1f1bbeadc9153b43912156e96d1abf588162bc51377a"

THINK_OPEN = re.compile(r"<think>")
THINK_CLOSE = re.compile(r"</think>")
BOXED = re.compile(r"\\boxed\{")


def group_analysis(records: list[dict], gold: dict[int, str],
                   extract_fn, grade_fn, n_draws: int) -> dict:
    """R2, measured properly: GRPO's signal lives in WITHIN-item variance.

    THE ERROR THIS EXISTS TO CORRECT. Job 3168210 measured 30.7% accuracy at one draw
    per item and it was recorded as "R2 satisfied". That inference does not hold. A 30.7%
    *marginal* rate is equally consistent with

      - every item at p ~ 0.307          -> almost no dead groups, healthy gradient
      - 30.7% of items at p ~ 1, rest 0  -> EVERY group dead, no gradient at all

    and one draw cannot tell them apart. GRPO's advantage is group-relative: a group whose
    rollouts all agree contributes exactly nothing, whatever the marginal rate is.

    So this reports the distribution of per-item correct counts, and the honest headline
    is `live_frac` -- the fraction of items that would actually produce a gradient.

    Also reports what an i.i.d. model would have predicted from the marginal alone. The
    gap between predicted and observed dead fractions IS the item heterogeneity, made
    visible rather than assumed away.
    """
    by_item: dict[int, list[bool]] = defaultdict(list)
    for r in records:
        ok = False
        try:
            ok = bool(grade_fn(extract_fn(r["text"]), gold[r["problem_id"]]))
        except Exception:
            pass
        by_item[r["problem_id"]].append(ok)

    cat_of = {r["problem_id"]: r["category"] for r in records}
    counts = {pid: sum(v) for pid, v in by_item.items()}
    n_items = len(counts)
    if n_items == 0:
        return {}

    def block(pids: list[int]) -> dict:
        m = len(pids)
        c = [counts[p] for p in pids]
        all_wrong = sum(1 for x in c if x == 0)
        all_right = sum(1 for x in c if x == n_draws)
        marginal = sum(c) / (m * n_draws)
        # What i.i.d. draws at the observed marginal WOULD have given.
        iid_dead = marginal ** n_draws + (1 - marginal) ** n_draws
        return {
            "n_items": m,
            "marginal_accuracy": marginal,
            "all_wrong": all_wrong / m,
            "all_correct": all_right / m,
            "dead_frac": (all_wrong + all_right) / m,
            "live_frac": 1 - (all_wrong + all_right) / m,
            "at_least_one_correct": 1 - all_wrong / m,   # O4 with m trajectories
            "iid_predicted_dead_frac": iid_dead,
            "heterogeneity_gap": (all_wrong + all_right) / m - iid_dead,
            "mean_correct_per_item": sum(c) / m,
        }

    by_cat: dict[str, list[int]] = defaultdict(list)
    for pid in counts:
        by_cat[cat_of[pid]].append(pid)

    return {
        "n_draws": n_draws,
        "histogram": {str(k): v for k, v in sorted(Counter(counts.values()).items())},
        "overall": block(list(counts)),
        "by_category": {c: block(p) for c, p in sorted(by_cat.items())},
    }


def summarise(records: list[dict], gold: dict[int, str],
              extract_fn, grade_fn) -> dict:
    n = len(records)
    lens = [r["n_tokens"] for r in records]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    def rates(rs: list[dict]) -> dict:
        m = len(rs)
        correct = 0
        for r in rs:
            try:
                correct += bool(grade_fn(extract_fn(r["text"]), gold[r["problem_id"]]))
            except Exception:
                pass
        return {
            "n": m,
            "think_open": sum(r["has_think_open"] for r in rs) / m,
            "think_close": sum(r["has_think_close"] for r in rs) / m,
            "boxed": sum(r["has_boxed"] for r in rs) / m,
            "eos": sum(r["finish_reason"] == "stop" for r in rs) / m,
            "truncated": sum(r["finish_reason"] == "length" for r in rs) / m,
            "accuracy": correct / m,
        }

    ordered = sorted(lens)
    def pct(p: float) -> int:
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    return {
        "n": n,
        "overall": rates(records),
        "by_category": {c: rates(rs) for c, rs in sorted(by_cat.items())},
        "length": {
            "mean": round(statistics.mean(lens), 1),
            "p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99), "max": max(lens),
            "over_4096": sum(1 for x in lens if x > 4096) / n,
        },
        "finish_reasons": dict(Counter(r["finish_reason"] for r in records)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--provenance", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = the whole split. Any limit is drawn STRATIFIED by "
                         "category; see pool_io.get_split and job 3168166.")
    ap.add_argument("--sample", choices=("stratified", "head"), default="stratified")
    ap.add_argument("--n-draws", type=int, default=1)
    # Deliberately NON-BINDING: measuring at Vision-SR1's 4096 and concluding
    # "4096 suffices" is circular, because a censored sample cannot report its
    # own tail. We measure at 8192 and then read off what 4096 would have cost.
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from mathruler.grader import extract_boxed_content, grade_answer

    # FIRST, before any other work. Two jobs have now died ~90 s in, during the
    # profile pass, on a ViT kernel that was mis-selected at import time. This
    # decides the same question in about a second.
    print("[gate] G-VITATTN: is the ViT attention patch in force?", flush=True)
    vit_gate = assert_vit_attn_patch(expect_sha256=PATCHED_LAYER_SHA256)

    prov = json.loads(Path(args.provenance).read_text())
    snapshot = Path(prov["snapshot_path"])

    manifest = load_manifest(Path(args.pool))
    items = get_split(manifest, args.split, args.limit,
                      sample=args.sample, seed=args.seed)
    gold = {it["problem_id"]: it["answer"] for it in items}
    # Printed BEFORE any generation. Job 3168166 measured 50 Chart rows and nothing
    # else while reporting "50 items from split 'dev'"; the skew only surfaced in the
    # by-category table at the very end, after the GPU time had been spent.
    composition = Counter(it["category"] for it in items)
    full = Counter(it["category"] for it in manifest["splits"][args.split])
    print(f"[setup] {len(items)} items from split '{args.split}' "
          f"(sample={args.sample}, seed={args.seed})", flush=True)
    print(f"[setup] composition: {dict(sorted(composition.items()))}", flush=True)
    print(f"[setup] full split:  {dict(sorted(full.items()))}", flush=True)
    if len(composition) < len(full):
        missing = sorted(set(full) - set(composition))
        raise AssertionError(
            f"sample covers only {sorted(composition)} but the split has {sorted(full)}; "
            f"missing {missing}. A single-category measurement is not a measurement of "
            f"this pool -- this is the job 3168166 failure.")

    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

    print("[setup] decoding images", flush=True)
    images = load_images(snapshot, items)

    biggest = max(images.values(), key=lambda im: im.size[0] * im.size[1])
    print(f"[gate] largest image {biggest.size}", flush=True)
    check_pixel_budget(processor, biggest, MAX_PIXELS)

    # Gates on the prompts themselves, before a single GPU token is spent.
    print("[gate] G-PARITY / G-BLIND over every item", flush=True)
    for it in items:
        prob = it["problem"] if "problem" in it else None
        if prob is None:
            raise KeyError("manifest items carry no 'problem' text")
        P.assert_parity(P.build_answerer_messages("PLACEHOLDER", prob),
                        P.build_sighted_messages(prob), prob)
    print(f"[gate] all {len(items)} items pass", flush=True)

    llm = LLM(model=args.model, trust_remote_code=True,
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 1},
              mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
              mm_encoder_attn_backend=MM_ENCODER_ATTN_BACKEND,
              seed=args.seed, enforce_eager=False, disable_log_stats=True)

    reqs = [{"prompt": processor.apply_chat_template(
                 P.build_sighted_messages(it["problem"]),
                 tokenize=False, add_generation_prompt=True),
             "multi_modal_data": {"image": images[it["problem_id"]]}}
            for it in items]

    outs = llm.generate(reqs, SamplingParams(
        n=args.n_draws, max_tokens=args.max_tokens, **SAMPLING))

    records = []
    for it, o in zip(items, outs):
        for k, cand in enumerate(o.outputs):
            t = cand.text
            records.append({
                "problem_id": it["problem_id"], "category": it["category"],
                "draw": k, "text": t, "n_tokens": len(cand.token_ids),
                "finish_reason": cand.finish_reason,
                "has_think_open": bool(THINK_OPEN.search(t)),
                "has_think_close": bool(THINK_CLOSE.search(t)),
                "has_boxed": bool(BOXED.search(t)),
            })

    report = summarise(records, gold, extract_boxed_content, grade_answer)
    if args.n_draws > 1:
        report["groups"] = group_analysis(
            records, gold, extract_boxed_content, grade_answer, args.n_draws)
    report["_meta"] = {
        "model": args.model,
        "model_revision": os.environ.get("CA21_MODEL_REV", "unknown"),
        "pool_manifest_sha256": manifest.get("manifest_sha256"),
        "dataset_revision": prov.get("revision"),
        "code_git_sha": os.environ.get("CA21_GIT_SHA", "unknown"),
        "split": args.split, "limit": args.limit, "n_draws": args.n_draws,
        "sample": args.sample, "n_items": len(items),
        "composition": dict(sorted(composition.items())),
        "sampling": {**SAMPLING, "seed": args.seed},
        "max_tokens": args.max_tokens, "max_model_len": MAX_MODEL_LEN,
        "max_pixels": MAX_PIXELS,
        "mm_encoder_attn_backend": MM_ENCODER_ATTN_BACKEND,
        # The kernel the ViT actually ran, as measured -- not as requested.
        "vit_attn_gate": vit_gate,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    # H2: persist EVERY record. Job 3168210 saved 20 of 300, so 3 of its 32 \boxed{}
    # failures could be characterised rather than all 32. Full responses are needed for
    # L1 anyway, and at these lengths the file is a few MB.
    out.with_suffix(".samples.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records))

    o = report["overall"]
    L = report["length"]
    print("\n=== SIGHTED PROMPT COMPLIANCE (S7 gates) ===")
    print(f"  n = {report['n']} generations\n")
    print(f"  \\boxed{{}} present   {o['boxed']:6.1%}   <- ESSENTIAL (gate >= 90%)")
    print(f"  <think>  present   {o['think_open']:6.1%}   <- convention (gate >= 90%)")
    print(f"  </think> present   {o['think_close']:6.1%}   <- convention")
    print(f"  EOS reached        {o['eos']:6.1%}   <- ESSENTIAL (gate >= 95%)")
    print(f"  truncated          {o['truncated']:6.1%}")
    print(f"\n  length  mean {L['mean']}  p50 {L['p50']}  p90 {L['p90']}  "
          f"p99 {L['p99']}  max {L['max']}")
    print(f"  would exceed a 4096 budget: {L['over_4096']:.1%}")
    print(f"\n  accuracy {o['accuracy']:.1%}   (REPORTED ONLY -- never a selection criterion)")
    print(f"\n  by category:")
    for c, r in report["by_category"].items():
        print(f"    {c:<10} n={r['n']:<4} boxed {r['boxed']:5.1%}  eos {r['eos']:5.1%}  "
              f"acc {r['accuracy']:5.1%}")
    if "groups" in report:
        g = report["groups"]
        go = g["overall"]
        print(f"\n=== R2: GROUP-LEVEL SIGNAL (n_draws={g['n_draws']}) ===")
        print("  This, not the marginal accuracy above, is what decides whether GRPO")
        print("  has a gradient. Groups whose rollouts all agree contribute nothing.\n")
        print(f"  items                     {go['n_items']}")
        print(f"  all {g['n_draws']} wrong             {go['all_wrong']:6.1%}   <- dead")
        print(f"  all {g['n_draws']} correct           {go['all_correct']:6.1%}   <- dead")
        print(f"  DEAD groups               {go['dead_frac']:6.1%}")
        print(f"  LIVE groups               {go['live_frac']:6.1%}   <- the gradient lives here")
        print(f"\n  >=1 correct trajectory    {go['at_least_one_correct']:6.1%}   <- O4 with m trajectories")
        print(f"  mean correct per item     {go['mean_correct_per_item']:.2f} / {g['n_draws']}")
        print(f"\n  i.i.d. would predict dead {go['iid_predicted_dead_frac']:6.1%}")
        print(f"  heterogeneity gap         {go['heterogeneity_gap']:+6.1%}   "
              f"(observed dead - i.i.d. prediction)")
        print(f"\n  correct-count histogram:  {g['histogram']}")
        print(f"\n  by category:")
        for c, r in g["by_category"].items():
            print(f"    {c:<10} n={r['n_items']:<4} live {r['live_frac']:5.1%}  "
                  f"dead {r['dead_frac']:5.1%}  >=1ok {r['at_least_one_correct']:5.1%}")

    print(f"\n[done] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
