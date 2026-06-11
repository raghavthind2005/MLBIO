#!/usr/bin/env python3
"""
HallusionBench evaluation with FORCED image re-examination.

Every sample gets exactly two turns regardless of what the model answers in turn 0:
  Turn 0: [system][user: image + Q] → <think>...</think> → initial answer (recorded but ignored)
  Turn 1: [... full image re-injected ...] → <think>...</think> → FINAL answer

This gives a clean controlled comparison:
  - Does seeing the image twice improve accuracy on edited/illusion samples?
  - Does the model change its answer after re-examination (and in which direction)?
  - How does attention to visual_turn1 (re-examined) compare to visual_turn0 (original)?

Extra fields vs normal run:
  - answer_turn0:         what the model would have said after turn 0 alone
  - answer_changed:       True if final answer differs from turn0 answer
  - thinking_turn0_chars: thinking length in turn 0
  - thinking_turn1_chars: thinking length in turn 1
  - thinking_per_stage:   [turn0_thinking, turn1_thinking]
  - stages:               per-turn token counts (compatible with extract_attention.py)
  - total_image_tokens:   always 512 (2 × 256) for image samples
  - n_tool_calls:         always 1 (for compatibility with extract_attention.py)

Usage:
  python run_eval_forced.py --port 30000 --dry-run 5
  python run_eval_forced.py --port 30000 --fraction 0.30 --out results_forced/forced_results.jsonl
"""

import argparse
import base64
import json
import random
import re
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "data" / "hallusionbench"
JSON_PATH  = DATA_DIR / "HallusionBench.json"
IMG_BASE   = DATA_DIR / "data"

# ─── Config ─────────────────────────────────────────────────────────────────
DEFAULT_PORT     = 30000
DEFAULT_FRACTION = 0.30
DEFAULT_SEED     = 42
MAX_NEW_TOKENS   = 16384
IMAGE_TOKENS     = 256   # Gemma-4 tokens per image injection

SYSTEM_PROMPT = (
    "You are a helpful visual question answering assistant. "
    "After your reasoning, answer ONLY with the single word 'Yes' or 'No'."
)

# Injected as the user turn between turn 0 and turn 1.
# Explicit Yes/No reminder combats the format drift seen in multi-turn.
REEXAMINE_TEXT = (
    "Here is the image again. Re-examine it carefully before giving your final answer.\n\n"
    "Answer ONLY with the single word 'Yes' or 'No'."
)


# ─── Image helpers ───────────────────────────────────────────────────────────

def encode_image(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def image_content(b64: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def parse_answer(text: str) -> str | None:
    """Return '1' (Yes) / '0' (No) / None if unparseable."""
    t = text.strip().lower()
    if t.startswith("yes"):         return "1"
    if t.startswith("no"):          return "0"
    if re.search(r'\byes\b', t):    return "1"
    if re.search(r'\bno\b',  t):    return "0"
    return None


# ─── Stratified sample ───────────────────────────────────────────────────────

def stratified_sample(data: list, fraction: float, seed: int) -> list:
    rng    = random.Random(seed)
    groups: dict[tuple, list] = defaultdict(list)
    for s in data:
        groups[(s["visual_input"], s["category"], s["subcategory"])].append(s)
    result = []
    for grp in groups.values():
        k = max(1, round(len(grp) * fraction))
        result.extend(rng.sample(grp, min(k, len(grp))))
    return result


# ─── Single-sample two-turn forced inference ─────────────────────────────────

def run_sample(
    url: str,
    model: str,
    question: str,
    img_path: Path | None,
    max_tokens: int = MAX_NEW_TOKENS,
    verbose: bool = False,
) -> dict:
    b64 = encode_image(img_path) if img_path else None

    # ── Turn 0: initial reasoning ─────────────────────────────────────────────
    user0_content = (
        [image_content(b64), {"type": "text", "text": question}] if b64 else question
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user0_content},
    ]

    t_start = time.time()

    r0 = requests.post(url, json={
        "model":    model,
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": True},
        "max_tokens":  max_tokens,
        "temperature": 1.0,
        "top_k": 64,
        "top_p": 0.95,
    }, timeout=600)
    r0.raise_for_status()
    resp0  = r0.json()
    ch0    = resp0["choices"][0]["message"]
    usage0 = resp0.get("usage", {})

    thinking0  = (ch0.get("reasoning_content") or "").strip()
    answer0    = (ch0.get("content") or "").strip()
    finish0    = resp0["choices"][0].get("finish_reason", "")
    comp_tok0  = usage0.get("completion_tokens") or 0

    stage0 = {
        "turn":              0,
        "prompt_tokens":     usage0.get("prompt_tokens"),
        "completion_tokens": comp_tok0,
        "finish_reason":     finish0,
        "thinking_chars":    len(thinking0),
        "answer_raw":        answer0,
    }

    if verbose:
        print(f"\n  ── Turn 0 ──────────────────────────────────")
        print(f"  finish_reason : {finish0}")
        print(f"  prompt_tokens : {usage0.get('prompt_tokens')}")
        print(f"  comp_tokens   : {comp_tok0}")
        print(f"  thinking      : {thinking0[:300]}{'...' if len(thinking0) > 300 else ''}")
        print(f"  answer_raw    : {answer0!r}  → parse={parse_answer(answer0)!r}")

    # ── Turn 1: forced re-examination ────────────────────────────────────────
    # Append turn-0 response and new user turn with the image re-injected.
    messages.append({"role": "assistant", "content": answer0})

    user1_content = (
        [image_content(b64), {"type": "text", "text": REEXAMINE_TEXT}]
        if b64 else REEXAMINE_TEXT
    )
    messages.append({"role": "user", "content": user1_content})

    r1 = requests.post(url, json={
        "model":    model,
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": True},
        "max_tokens":  max_tokens,
        "temperature": 1.0,
        "top_k": 64,
        "top_p": 0.95,
    }, timeout=600)
    r1.raise_for_status()
    resp1  = r1.json()
    ch1    = resp1["choices"][0]["message"]
    usage1 = resp1.get("usage", {})

    thinking1     = (ch1.get("reasoning_content") or "").strip()
    final_answer  = (ch1.get("content") or "").strip()
    finish1       = resp1["choices"][0].get("finish_reason", "")
    comp_tok1     = usage1.get("completion_tokens") or 0

    stage1 = {
        "turn":              1,
        "prompt_tokens":     usage1.get("prompt_tokens"),
        "completion_tokens": comp_tok1,
        "finish_reason":     finish1,
        "thinking_chars":    len(thinking1),
        "answer_raw":        final_answer,
    }

    if verbose:
        print(f"\n  ── Turn 1 (forced re-examination) ──────────")
        print(f"  finish_reason : {finish1}")
        print(f"  prompt_tokens : {usage1.get('prompt_tokens')}")
        print(f"  comp_tokens   : {comp_tok1}")
        print(f"  thinking      : {thinking1[:300]}{'...' if len(thinking1) > 300 else ''}")
        print(f"  answer_raw    : {final_answer!r}  → parse={parse_answer(final_answer)!r}")

    elapsed           = time.time() - t_start
    total_comp_tokens = comp_tok0 + comp_tok1
    all_thinking      = len(thinking0) + len(thinking1)
    total_img_tokens  = (IMAGE_TOKENS * 2) if img_path else 0
    final_prompt_tok  = stage1["prompt_tokens"]

    pred0  = parse_answer(answer0)
    pred1  = parse_answer(final_answer)

    return {
        # model output
        "model_prediction":        pred1,
        "answer_text":             final_answer,
        "answer_turn0":            answer0,
        "pred_turn0":              pred0,
        "answer_changed":          (pred0 != pred1) if (pred0 and pred1) else None,

        # thinking
        "thinking_per_stage":      [thinking0, thinking1],
        "thinking_turn0_chars":    len(thinking0),
        "thinking_turn1_chars":    len(thinking1),
        "all_thinking_chars":      all_thinking,
        "thinking_chars":          all_thinking,   # alias for analyze.py

        # forced re-examination always counts as 1 tool call for extract_attention compat
        "n_tool_calls":            1 if img_path else 0,
        "tool_calls":              [{
            "call_index":            0,
            "region":                "full",
            "thinking_chars_before": len(thinking0),
            "image_tokens_added":    IMAGE_TOKENS if img_path else 0,
        }] if img_path else [],
        "total_image_tokens":      total_img_tokens,

        # token accounting
        "stages":                  [stage0, stage1],
        "total_completion_tokens": total_comp_tokens,
        "final_prompt_tokens":     final_prompt_tok,
        "visual_token_ratio_approx": (
            total_img_tokens / final_prompt_tok if final_prompt_tok else None
        ),

        # timing
        "inference_time_s":  round(elapsed, 3),
        "tokens_per_second": round(
            total_comp_tokens / elapsed if elapsed > 0 else 0, 2
        ),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",       type=int,   default=DEFAULT_PORT)
    ap.add_argument("--model",      default=None)
    ap.add_argument("--fraction",   type=float, default=DEFAULT_FRACTION)
    ap.add_argument("--seed",       type=int,   default=DEFAULT_SEED)
    ap.add_argument("--max-tokens", type=int,   default=MAX_NEW_TOKENS)
    ap.add_argument("--out",        default=None)
    ap.add_argument("--dry-run",    type=int,   default=0, metavar="N",
                    help="Run N samples (preferring edited images) verbosely. No file written.")
    args = ap.parse_args()

    base_url = f"http://localhost:{args.port}"
    chat_url = f"{base_url}/v1/chat/completions"

    model_name = args.model
    if model_name is None:
        models     = requests.get(f"{base_url}/v1/models", timeout=10).json()
        model_name = models["data"][0]["id"]
        print(f"Auto-detected model: {model_name}")

    with open(JSON_PATH) as f:
        data = json.load(f)
    for s in data:
        s["image_path"] = (
            str(IMG_BASE / s["filename"].lstrip("./")) if s.get("filename") else None
        )

    data = [s for s in data if s["visual_input"] != "0"]

    if args.dry_run:
        edited   = [s for s in data if s["visual_input"] == "2"]
        original = [s for s in data if s["visual_input"] == "1"]
        rng      = random.Random(args.seed)
        subset   = rng.sample(edited + original, min(args.dry_run, len(edited) + len(original)))
        print(f"DRY RUN — {len(subset)} samples\n" + "=" * 60)
    else:
        subset = stratified_sample(data, args.fraction, args.seed)

    out_dir  = SCRIPT_DIR / "results_forced"
    out_dir.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / "forced_results.jsonl"

    n_correct = n_total = n_parse_fail = 0
    n_changed = n_changed_correct = n_changed_wrong = 0

    fout = open(out_path, "w") if not args.dry_run else None

    try:
        for i, sample in enumerate(subset):
            tag = (f"[{i+1:3d}/{len(subset)}] "
                   f"{sample['category']}/{sample['subcategory']:8s} vi={sample['visual_input']}")
            print(f"{tag}", end="  ", flush=True)

            try:
                img_path = Path(sample["image_path"]) if sample["image_path"] else None
                result   = run_sample(
                    chat_url, model_name, sample["question"], img_path,
                    max_tokens=args.max_tokens,
                    verbose=bool(args.dry_run),
                )

                pred       = result["model_prediction"]
                is_correct = int(pred == sample["gt_answer"]) if pred is not None else None

                record = {
                    "sample_id":   (f"{sample['category']}_{sample['set_id']}_"
                                    f"{sample['figure_id']}_{sample['question_id']}"),
                    "category":    sample["category"],
                    "subcategory": sample["subcategory"],
                    "set_id":      sample["set_id"],
                    "figure_id":   sample["figure_id"],
                    "question_id": sample["question_id"],
                    "visual_input":sample["visual_input"],
                    "question":    sample["question"],
                    "gt_answer":   sample["gt_answer"],
                    "image_path":  str(img_path) if img_path else None,
                    "is_correct":  is_correct,
                    **result,
                }

                if fout:
                    fout.write(json.dumps(record) + "\n")
                    fout.flush()

                if is_correct is not None:
                    n_correct += is_correct
                    n_total   += 1
                    if result["answer_changed"] is True:
                        n_changed += 1
                        if is_correct == 1 and result["pred_turn0"] == "0":
                            n_changed_correct += 1   # wrong→right
                        elif is_correct == 0 and result["pred_turn0"] == "1":
                            n_changed_wrong   += 1   # right→wrong
                else:
                    n_parse_fail += 1

                sym = "✓" if is_correct else ("✗" if is_correct == 0 else "?")
                chg = "↑" if (result["answer_changed"] and is_correct == 1
                               and result["pred_turn0"] == "0") else (
                      "↓" if (result["answer_changed"] and is_correct == 0
                               and result["pred_turn0"] == "1") else (
                      "~" if result["answer_changed"] else " "))
                print(
                    f"{sym}{chg} "
                    f"t0={result['thinking_turn0_chars']:5d}ch "
                    f"t1={result['thinking_turn1_chars']:5d}ch "
                    f"img={result['total_image_tokens']}tok "
                    f"{result['inference_time_s']:.1f}s"
                )

                if args.dry_run:
                    assert result["total_image_tokens"] == IMAGE_TOKENS * 2, \
                        f"expected {IMAGE_TOKENS*2} image tokens, got {result['total_image_tokens']}"
                    assert result["n_tool_calls"] == 1
                    assert len(result["stages"]) == 2
                    assert len(result["thinking_per_stage"]) == 2
                    print(f"  turn0_answer={result['answer_turn0']!r}  "
                          f"final_answer={result['answer_text']!r}  "
                          f"changed={result['answer_changed']}")
                    print(f"  stages: "
                          + "  ".join(f"t{s['turn']}: p={s['prompt_tokens']} "
                                      f"c={s['completion_tokens']} fin={s['finish_reason']}"
                                      for s in result["stages"]))
                    print()

            except Exception as exc:
                import traceback
                print(f"ERROR: {exc}")
                if args.dry_run:
                    traceback.print_exc()
                if fout:
                    fout.write(json.dumps({
                        "sample_id": (f"{sample['category']}_{sample['set_id']}_"
                                      f"{sample['figure_id']}_{sample['question_id']}"),
                        "error": str(exc),
                    }) + "\n")
                    fout.flush()

    finally:
        if fout:
            fout.close()

    print("\n" + "─" * 60)
    if n_total > 0:
        print(f"qAcc          : {n_correct}/{n_total} = {n_correct/n_total:.4f}")
        print(f"parse failures: {n_parse_fail}")
        print(f"answer changed: {n_changed}/{n_total} = {n_changed/n_total:.2f}")
        print(f"  wrong→right (↑): {n_changed_correct}")
        print(f"  right→wrong (↓): {n_changed_wrong}")
        print(f"  neutral change : {n_changed - n_changed_correct - n_changed_wrong}")
    if not args.dry_run:
        print(f"Results       : {out_path}")


if __name__ == "__main__":
    main()
