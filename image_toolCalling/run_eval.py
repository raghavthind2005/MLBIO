#!/usr/bin/env python3
"""
HallusionBench evaluation via sglang Gemma-4 server.

Saves per-sample: response, thinking trace, token counts, timing.
Attention extraction is done separately (extract_attention.py).

Usage:
    python run_eval.py [--port 30000] [--fraction 0.30] [--seed 42]
"""

import argparse
import base64
import json
import random
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "data" / "hallusionbench"
JSON_PATH  = DATA_DIR / "HallusionBench.json"
IMG_BASE   = DATA_DIR / "data"

DEFAULT_PORT     = 30000
DEFAULT_FRACTION = 0.30
DEFAULT_SEED     = 42
MAX_NEW_TOKENS   = 2048

SYSTEM_PROMPT = (
    "You are a helpful visual question answering assistant. "
    "After your reasoning, answer ONLY with the single word 'Yes' or 'No'."
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def encode_image(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def parse_answer(text: str) -> str | None:
    """Return '1' (Yes) or '0' (No), or None if unparseable."""
    t = text.strip().lower()
    if t.startswith("yes"):
        return "1"
    if t.startswith("no"):
        return "0"
    first50 = t[:50]
    if "yes" in first50:
        return "1"
    if "no" in first50:
        return "0"
    return None


def stratified_sample(data: list, fraction: float, seed: int) -> list:
    """Preserve (visual_input, category, subcategory) distribution."""
    rng = random.Random(seed)
    groups: dict[tuple, list] = defaultdict(list)
    for s in data:
        groups[(s["visual_input"], s["category"], s["subcategory"])].append(s)
    result = []
    for grp in groups.values():
        k = max(1, round(len(grp) * fraction))
        result.extend(rng.sample(grp, min(k, len(grp))))
    return result


def call_model(
    url: str, model: str, question: str, img_path: Path | None
) -> tuple[dict, float]:
    content = []
    if img_path is not None:
        b64 = encode_image(img_path)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": question})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
        "chat_template_kwargs": {"enable_thinking": True},
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
    }

    t0 = time.time()
    r  = requests.post(url, json=payload, timeout=600)
    elapsed = time.time() - t0
    r.raise_for_status()
    return r.json(), elapsed


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",     type=int,   default=DEFAULT_PORT)
    ap.add_argument("--model",    default=None, help="Model name (auto-detected if omitted)")
    ap.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    ap.add_argument("--seed",     type=int,   default=DEFAULT_SEED)
    ap.add_argument("--out",      default=None)
    args = ap.parse_args()

    base_url = f"http://localhost:{args.port}"
    chat_url = f"{base_url}/v1/chat/completions"

    # Auto-detect running model name
    model_name = args.model
    if model_name is None:
        models = requests.get(f"{base_url}/v1/models", timeout=10).json()
        model_name = models["data"][0]["id"]
        print(f"Auto-detected model: {model_name}")

    # Load dataset
    with open(JSON_PATH) as f:
        data = json.load(f)
    for s in data:
        s["image_path"] = str(IMG_BASE / s["filename"].lstrip("./")) if s.get("filename") else None

    subset = stratified_sample(data, args.fraction, args.seed)
    print(f"Subset: {len(subset)}/{len(data)} samples ({args.fraction*100:.0f}%)")
    for vi, cnt in sorted(
        defaultdict(int, {k: sum(1 for s in subset if s["visual_input"] == k) for k in ("0","1","2")}).items()
    ):
        label = {"0": "text-only", "1": "original", "2": "edited"}[vi]
        print(f"  visual_input={vi} ({label}): {cnt}")

    out_dir  = SCRIPT_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / "raw_results.jsonl"
    print(f"Writing to: {out_path}\n")

    n_correct = n_total = n_parse_fail = 0

    with open(out_path, "w") as fout:
        for i, sample in enumerate(subset):
            tag = f"[{i+1:3d}/{len(subset)}] {sample['category']}/{sample['subcategory']:8s}"
            print(tag, end="  ", flush=True)

            try:
                img_path = Path(sample["image_path"]) if sample["image_path"] else None
                resp, elapsed = call_model(chat_url, model_name, sample["question"], img_path)

                msg      = resp["choices"][0]["message"]
                thinking = (msg.get("reasoning_content") or "").strip()
                answer   = (msg.get("content") or "").strip()
                usage    = resp.get("usage", {})

                pred       = parse_answer(answer)
                is_correct = int(pred == sample["gt_answer"]) if pred is not None else None

                # sglang returns prompt_tokens which includes image tokens.
                # Gemma-4 encodes one image as 256 soft tokens at default resolution;
                # treat this as an approximation until we verify via HF tokenizer.
                n_img_tok  = 256 if img_path else 0
                prompt_tok = usage.get("prompt_tokens")
                comp_tok   = usage.get("completion_tokens")

                record = {
                    # identity
                    "sample_id":  f"{sample['category']}_{sample['set_id']}_{sample['figure_id']}_{sample['question_id']}",
                    "category":   sample["category"],
                    "subcategory":sample["subcategory"],
                    "set_id":     sample["set_id"],
                    "figure_id":  sample["figure_id"],
                    "question_id":sample["question_id"],
                    "visual_input": sample["visual_input"],
                    "question":   sample["question"],
                    "gt_answer":  sample["gt_answer"],
                    "image_path": str(img_path) if img_path else None,

                    # model output
                    "model_prediction": pred,
                    "answer_text":      answer,
                    "thinking_content": thinking,
                    "is_correct":       is_correct,

                    # token counts
                    "prompt_tokens":            prompt_tok,
                    "completion_tokens":        comp_tok,
                    "thinking_chars":           len(thinking),
                    "answer_chars":             len(answer),
                    "n_image_tokens_approx":    n_img_tok,
                    "visual_token_ratio_approx": (
                        round(n_img_tok / prompt_tok, 4)
                        if prompt_tok and n_img_tok else None
                    ),

                    # timing
                    "inference_time_s":  round(elapsed, 3),
                    "tokens_per_second": round(comp_tok / elapsed if comp_tok and elapsed > 0 else 0, 2),
                }

                fout.write(json.dumps(record) + "\n")
                fout.flush()

                if is_correct is not None:
                    n_correct += is_correct
                    n_total   += 1
                else:
                    n_parse_fail += 1

                sym = "✓" if is_correct else ("✗" if is_correct == 0 else "?")
                print(f"{sym}  think={len(thinking):5d}ch  comp={comp_tok or '?':>4}tok  {elapsed:.1f}s")

            except Exception as exc:
                print(f"ERROR: {exc}")
                sid = f"{sample['category']}_{sample['set_id']}_{sample['figure_id']}_{sample['question_id']}"
                fout.write(json.dumps({"sample_id": sid, "error": str(exc)}) + "\n")
                fout.flush()

    print(f"\n{'─'*60}")
    if n_total:
        print(f"qAcc : {n_correct}/{n_total} = {n_correct/n_total:.4f}")
    print(f"parse failures : {n_parse_fail}")
    print(f"Results saved  : {out_path}")


if __name__ == "__main__":
    main()
