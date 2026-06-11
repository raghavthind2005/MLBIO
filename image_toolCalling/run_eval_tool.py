#!/usr/bin/env python3
"""
HallusionBench evaluation with voluntary mid-reasoning image re-examination.

The model is given a tool: respond with "LOOK_AGAIN: <region>" as its ANSWER
(not inside thinking) to request the image before giving a final answer.
This preserves the sglang reasoning parser (no mid-think interruption) and
keeps Turn-1 thinking vs Turn-2 thinking cleanly separated for analysis.

Flow per sample:
  Turn 0: [system][user: image + Q]  → <think>...</think>  → "Yes"/"No" OR "LOOK_AGAIN: full"
  Turn 1: [... injected image ...]   → <think>...</think>  → "Yes"/"No" OR "LOOK_AGAIN: ..."
  ...up to MAX_TOOL_CALLS turns

Data saved:
  - All normal eval fields (is_correct, answer, token counts, timing)
  - n_tool_calls:           0..MAX_TOOL_CALLS
  - tool_calls:             [{region, thinking_chars_at_decision, image_tokens_added}, ...]
  - thinking_per_stage:     [turn0_thinking, turn1_thinking, ...]   (separate, not merged)
  - stages:                 [{prompt_tokens, completion_tokens, finish_reason}, ...]
  - total_image_tokens:     256 × (1 + n_tool_calls) for image samples
  - total_completion_tokens

Usage:
  # Dry-run: 5 edited samples, verbose, no file written
  python run_eval_tool.py --port 30000 --dry-run 5

  # Full run
  python run_eval_tool.py --port 30000 --fraction 0.30 --out results/tool_results.jsonl
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

# ─── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "data" / "hallusionbench"
JSON_PATH  = DATA_DIR / "HallusionBench.json"
IMG_BASE   = DATA_DIR / "data"

# ─── Config ─────────────────────────────────────────────────────────────────
DEFAULT_PORT      = 30000
DEFAULT_FRACTION  = 0.30
DEFAULT_SEED      = 42
MAX_NEW_TOKENS    = 16384
MAX_TOOL_CALLS    = 3     # max re-examinations per sample before forcing final answer
IMAGE_TOKENS      = 256   # Gemma-4 standard resolution per image injection

LOOK_AGAIN_RE = re.compile(
    r"LOOK_AGAIN\s*:\s*(full|top-left|top-right|bottom-left|bottom-right|top|bottom|left|right)",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are a careful visual question answering assistant.

TOOL AVAILABLE:
If you need to re-examine the image before giving your final answer, respond with
EXACTLY one of:
  LOOK_AGAIN: full
  LOOK_AGAIN: top-left
  LOOK_AGAIN: top-right
  LOOK_AGAIN: bottom-left
  LOOK_AGAIN: bottom-right

You will immediately receive the requested image view and can then reason further.
Use this tool when visual details are unclear or when your reasoning depends on
a specific region you cannot verify from memory.

After all reasoning, answer ONLY with the single word Yes or No."""


# ─── Image helpers ──────────────────────────────────────────────────────────

REGION_BOXES = {
    "top-left":     (0.0, 0.0, 0.5, 0.5),
    "top-right":    (0.5, 0.0, 1.0, 0.5),
    "bottom-left":  (0.0, 0.5, 0.5, 1.0),
    "bottom-right": (0.5, 0.5, 1.0, 1.0),
    "top":          (0.0, 0.0, 1.0, 0.5),
    "bottom":       (0.0, 0.5, 1.0, 1.0),
    "left":         (0.0, 0.0, 0.5, 1.0),
    "right":        (0.5, 0.0, 1.0, 1.0),
    "full":         (0.0, 0.0, 1.0, 1.0),
}


def encode_image(path: Path, region: str = "full") -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    rx1, ry1, rx2, ry2 = REGION_BOXES.get(region.lower(), REGION_BOXES["full"])
    box = (int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h))
    img = img.crop(box)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def parse_answer(text: str) -> str | None:
    """Return '1' (Yes) / '0' (No) / None if unparseable."""
    t = text.strip().lower()
    if t.startswith("yes"):  return "1"
    if t.startswith("no"):   return "0"
    if "yes" in t[:50]:      return "1"
    if "no"  in t[:50]:      return "0"
    return None


def parse_look_again(text: str) -> str | None:
    """Extract region from a LOOK_AGAIN answer, or None if not a tool call."""
    m = LOOK_AGAIN_RE.search(text)
    return m.group(1).lower() if m else None


# ─── Stratified sample (visual-only) ────────────────────────────────────────

def stratified_sample(data: list, fraction: float, seed: int) -> list:
    """Stratify by (visual_input, category, subcategory), keep only image samples."""
    rng = random.Random(seed)
    groups: dict[tuple, list] = defaultdict(list)
    for s in data:
        groups[(s["visual_input"], s["category"], s["subcategory"])].append(s)
    result = []
    for grp in groups.values():
        k = max(1, round(len(grp) * fraction))
        result.extend(rng.sample(grp, min(k, len(grp))))
    return result


# ─── Single-sample multi-turn inference ─────────────────────────────────────

def run_sample(
    url: str,
    model: str,
    question: str,
    img_path: Path | None,
    verbose: bool = False,
) -> dict:
    """
    Run multi-turn inference with optional tool use.
    Returns a dict of all fields to be merged into the final record.
    """

    def image_content(b64: str) -> dict:
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

    b64_original = encode_image(img_path) if img_path else None

    # Initial messages
    user_content: list | str
    if b64_original:
        user_content = [image_content(b64_original), {"type": "text", "text": question}]
    else:
        user_content = question

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    thinking_per_stage: list[str] = []
    stages: list[dict]            = []
    tool_calls: list[dict]        = []
    total_completion_tokens: int  = 0
    final_answer: str             = ""
    t_start = time.time()

    for turn in range(MAX_TOOL_CALLS + 1):
        payload = {
            "model":  model,
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": True},
            "max_tokens":  MAX_NEW_TOKENS,
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.95,
        }

        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        resp   = r.json()
        choice = resp["choices"][0]["message"]
        usage  = resp.get("usage", {})

        thinking   = (choice.get("reasoning_content") or "").strip()
        answer_raw = (choice.get("content") or "").strip()
        finish     = resp["choices"][0].get("finish_reason", "")

        thinking_per_stage.append(thinking)
        comp_tok = usage.get("completion_tokens") or 0
        total_completion_tokens += comp_tok

        stages.append({
            "turn":               turn,
            "prompt_tokens":      usage.get("prompt_tokens"),
            "completion_tokens":  comp_tok,
            "finish_reason":      finish,
            "thinking_chars":     len(thinking),
            "answer_raw":         answer_raw,
        })

        if verbose:
            print(f"\n  ── Turn {turn} ──────────────────────────────")
            print(f"  finish_reason : {finish}")
            print(f"  prompt_tokens : {usage.get('prompt_tokens')}")
            print(f"  comp_tokens   : {comp_tok}")
            print(f"  thinking      : {thinking[:300]}{'...' if len(thinking) > 300 else ''}")
            print(f"  answer_raw    : {answer_raw!r}")

        # ── Tool call? ───────────────────────────────────────────────────
        region = parse_look_again(answer_raw)

        if region is None or turn >= MAX_TOOL_CALLS:
            # Normal answer or last allowed turn — accept this as the final answer
            final_answer = answer_raw
            break

        # ── Process tool call ────────────────────────────────────────────
        chars_at_decision = sum(len(t) for t in thinking_per_stage)

        tool_calls.append({
            "call_index":            len(tool_calls),
            "region":                region,
            "thinking_chars_before": chars_at_decision,
            "image_tokens_added":    IMAGE_TOKENS if img_path else 0,
        })

        if verbose:
            print(f"\n  *** TOOL CALL {len(tool_calls)-1}: region='{region}' "
                  f"at total_thinking_chars={chars_at_decision} ***")

        # Append model turn (for context) and new user turn with injected image
        messages.append({"role": "assistant", "content": answer_raw})

        b64_look = encode_image(img_path, region) if img_path else b64_original
        tool_user_content: list | str
        if b64_look:
            tool_user_content = [
                image_content(b64_look),
                {"type": "text",
                 "text": f"Here is the image view you requested ({region}). "
                         "Continue your reasoning and give a final answer."},
            ]
        else:
            tool_user_content = "No image available. Please give your final answer."

        messages.append({"role": "user", "content": tool_user_content})

    elapsed = time.time() - t_start

    # Compute totals
    n_tool_calls      = len(tool_calls)
    total_image_tok   = (IMAGE_TOKENS if img_path else 0) + sum(
        tc["image_tokens_added"] for tc in tool_calls
    )
    pred              = parse_answer(final_answer)
    all_thinking_chars = sum(len(t) for t in thinking_per_stage)

    return {
        # model output
        "model_prediction":        pred,
        "answer_text":             final_answer,
        "thinking_per_stage":      thinking_per_stage,
        "all_thinking_chars":      all_thinking_chars,

        # tool use
        "n_tool_calls":            n_tool_calls,
        "tool_calls":              tool_calls,
        "total_image_tokens":      total_image_tok,

        # token accounting
        "stages":                  stages,
        "total_completion_tokens": total_completion_tokens,
        "final_prompt_tokens":     stages[-1]["prompt_tokens"] if stages else None,

        # timing
        "inference_time_s": round(elapsed, 3),
        "tokens_per_second": round(
            total_completion_tokens / elapsed if elapsed > 0 else 0, 2
        ),
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",     type=int,   default=DEFAULT_PORT)
    ap.add_argument("--model",    default=None)
    ap.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    ap.add_argument("--seed",     type=int,   default=DEFAULT_SEED)
    ap.add_argument("--out",      default=None)
    ap.add_argument("--dry-run",  type=int,   default=0, metavar="N",
                    help="Run N samples (preferring edited images) with verbose "
                         "output for verification. Does NOT write results.")
    args = ap.parse_args()

    base_url = f"http://localhost:{args.port}"
    chat_url = f"{base_url}/v1/chat/completions"

    model_name = args.model
    if model_name is None:
        models = requests.get(f"{base_url}/v1/models", timeout=10).json()
        model_name = models["data"][0]["id"]
        print(f"Auto-detected model: {model_name}")

    with open(JSON_PATH) as f:
        data = json.load(f)
    for s in data:
        s["image_path"] = (
            str(IMG_BASE / s["filename"].lstrip("./")) if s.get("filename") else None
        )

    # Visual-only
    data = [s for s in data if s["visual_input"] != "0"]

    if args.dry_run:
        # Prefer visual_input="2" (edited/hard) for dry run — most likely to trigger tool
        edited  = [s for s in data if s["visual_input"] == "2"]
        original = [s for s in data if s["visual_input"] == "1"]
        rng = random.Random(args.seed)
        pool = edited + original
        subset = rng.sample(pool, min(args.dry_run, len(pool)))
        print(f"DRY RUN — {len(subset)} samples (preferring edited images)\n")
        print("=" * 60)
    else:
        subset = stratified_sample(data, args.fraction, args.seed)
        print(f"Subset: {len(subset)} visual samples ({args.fraction*100:.0f}%)")

    out_dir  = SCRIPT_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / "tool_results.jsonl"

    n_correct = n_total = n_tool_uses = n_parse_fail = 0

    fout = open(out_path, "w") if not args.dry_run else None

    try:
        for i, sample in enumerate(subset):
            tag = f"[{i+1:3d}/{len(subset)}] {sample['category']}/{sample['subcategory']:8s} vi={sample['visual_input']}"
            print(f"{tag}", end="  ", flush=True)

            try:
                img_path = Path(sample["image_path"]) if sample["image_path"] else None
                result   = run_sample(
                    chat_url, model_name, sample["question"], img_path,
                    verbose=bool(args.dry_run),
                )

                pred       = result["model_prediction"]
                is_correct = int(pred == sample["gt_answer"]) if pred is not None else None

                record = {
                    # identity
                    "sample_id":    (f"{sample['category']}_{sample['set_id']}_"
                                     f"{sample['figure_id']}_{sample['question_id']}"),
                    "category":     sample["category"],
                    "subcategory":  sample["subcategory"],
                    "set_id":       sample["set_id"],
                    "figure_id":    sample["figure_id"],
                    "question_id":  sample["question_id"],
                    "visual_input": sample["visual_input"],
                    "question":     sample["question"],
                    "gt_answer":    sample["gt_answer"],
                    "image_path":   str(img_path) if img_path else None,
                    "is_correct":   is_correct,
                    **result,
                }

                if fout:
                    fout.write(json.dumps(record) + "\n")
                    fout.flush()

                if is_correct is not None:
                    n_correct   += is_correct
                    n_total     += 1
                    n_tool_uses += result["n_tool_calls"]
                else:
                    n_parse_fail += 1

                sym = "✓" if is_correct else ("✗" if is_correct == 0 else "?")
                print(
                    f"{sym}  tools={result['n_tool_calls']}  "
                    f"think={result['all_thinking_chars']:5d}ch  "
                    f"img_tok={result['total_image_tokens']:4d}  "
                    f"{result['inference_time_s']:.1f}s"
                )

                if args.dry_run:
                    print(f"\n  VERIFICATION CHECK:")
                    print(f"    n_tool_calls       = {result['n_tool_calls']}")
                    print(f"    total_image_tokens = {result['total_image_tokens']}"
                          f"  (expected {256 * (1 + result['n_tool_calls'])})")
                    assert result["total_image_tokens"] == 256 * (1 + result["n_tool_calls"]), \
                        "image token count mismatch!"
                    for j, tc in enumerate(result["tool_calls"]):
                        print(f"    tool_call[{j}]: region={tc['region']}  "
                              f"thinking_chars_before={tc['thinking_chars_before']}  "
                              f"img_tokens_added={tc['image_tokens_added']}")
                    print(f"    stages:")
                    for st in result["stages"]:
                        print(f"      turn {st['turn']}: "
                              f"prompt={st['prompt_tokens']}  "
                              f"comp={st['completion_tokens']}  "
                              f"finish={st['finish_reason']}")
                    if len(result["stages"]) > 1:
                        delta = (result["stages"][-1]["prompt_tokens"] or 0) - \
                                (result["stages"][0]["prompt_tokens"] or 0)
                        print(f"    prompt_token increase per tool call: "
                              f"+{delta} (includes injected image ~256 tokens)")
                    print()

            except AssertionError as exc:
                print(f"\n  ASSERTION FAILED: {exc}")
            except Exception as exc:
                print(f"ERROR: {exc}")
                import traceback; traceback.print_exc()
                if fout:
                    sid = (f"{sample['category']}_{sample['set_id']}_"
                           f"{sample['figure_id']}_{sample['question_id']}")
                    fout.write(json.dumps({"sample_id": sid, "error": str(exc)}) + "\n")
                    fout.flush()

    finally:
        if fout:
            fout.close()

    print(f"\n{'─' * 60}")
    if n_total:
        print(f"qAcc          : {n_correct}/{n_total} = {n_correct/n_total:.4f}")
        print(f"tool calls    : {n_tool_uses} total, {n_tool_uses/n_total:.2f} avg/sample")
        print(f"  samples w/ tool use : {sum(1 for _ in range(n_total))}")
    print(f"parse failures: {n_parse_fail}")
    if not args.dry_run:
        print(f"Results       : {out_path}")


if __name__ == "__main__":
    main()
