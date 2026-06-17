#!/usr/bin/env python3
"""
BabyVision standard run — Gemma-4-31B via sglang.
Paper-faithful protocol + comprehensive passive instrumentation.

Protocol (per paper §3.3):
  - No system prompt
  - Prompt suffix: "\\nThink about the question and give your final answer in \\boxed{Answer} format."
  - Thinking ON (highest budget), temperature=1.0, top_k=64, top_p=0.95, max_tokens=65536
  - 3 independent passes, pass@1 mean ± std

Output per pass N:
  results_run{N}.jsonl   — compact: all fields except logprob sequence
  logprobs_run{N}.jsonl  — heavy: full per-token logprob+top_20 (for entropy/confidence/teacher-forcing)

Judge is run SEPARATELY by run_judge.py (fills judge_result / judge_raw in results files).
"""

import argparse
import base64
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

# ── Protocol constants (must match paper exactly) ──────────────────────────────
MAX_TOKENS     = 65536
TEMPERATURE    = 1.0
TOP_K          = 64
TOP_P          = 0.95
N_TOP_LOGPROBS = 20
N_CONCURRENT   = 8
N_PASSES       = 3
PROMPT_SUFFIX  = "\nThink about the question and give your final answer in \\boxed{Answer} format."

# ── Image helpers ──────────────────────────────────────────────────────────────

def encode_image(path: Path) -> str:
    """Data URI preserving original format — no JPEG re-encode (protects fine grids/mazes)."""
    ext = path.suffix.lower()
    with open(path, "rb") as f:
        raw = f.read()
    if ext in (".jpg", ".jpeg"):
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    if ext == ".png":
        return "data:image/png;base64," + base64.b64encode(raw).decode()
    # rare fallback: convert unknown formats to PNG
    from io import BytesIO
    buf = BytesIO()
    Image.open(path).convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def get_image_properties(path: Path) -> dict:
    try:
        img = Image.open(path)
        w, h = img.size
        return {
            "img_width":        w,
            "img_height":       h,
            "img_megapixels":   round(w * h / 1e6, 4),
            "img_aspect_ratio": round(w / h, 3) if h else None,
            "img_mode":         img.mode,
            "img_is_grayscale": img.mode in ("L", "LA"),
            "img_format":       img.format or path.suffix.lstrip(".").upper(),
            "img_file_bytes":   path.stat().st_size,
        }
    except Exception as e:
        return {"img_error": str(e)}

# ── Question formatting ────────────────────────────────────────────────────────

def format_choices(options: list) -> str:
    return "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(options))


def build_question(item: dict) -> tuple[str, str]:
    """Returns (full question as sent to model, ground-truth answer string)."""
    if item["ansType"] == "blank":
        return item["question"] + PROMPT_SUFFIX, item["blankAns"]
    opts = format_choices(item["options"])
    q    = item["question"] + "\nChoices:\n" + opts + PROMPT_SUFFIX
    gt   = chr(65 + int(item["choiceAns"]))
    return q, gt

# ── Answer extraction ──────────────────────────────────────────────────────────

def extract_boxed_answer(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if m:
        return m[-1].strip()
    m = re.findall(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', text, re.DOTALL)
    if m:
        return m[-1].strip()
    return None

# ── Reasoning-structure features ───────────────────────────────────────────────

_RE_CORRECTION = re.compile(r'\b(wait,?|actually,?|hmm,?|hold on|let me reconsider|no,|wait—|oops)\b', re.I)
_RE_IMG_REF    = re.compile(r'\b(the image|in the image|looking at|i can see|from the image|the picture|i see)\b', re.I)
_RE_ENUM       = re.compile(r'(?:^|\n)\s*(?:\d+[\.\)]|step\s+\d+)', re.I | re.M)
_RE_BACKTRACK  = re.compile(r'\b(let me start over|actually let me|wait let me|let me re.?check|let me re.?do)\b', re.I)


def extract_reasoning_features(thinking: str, extracted: Optional[str]) -> dict:
    t = thinking or ""
    boxed_in_thinking = re.findall(r'\\boxed\{([^{}]*)\}', t)
    last_mid = boxed_in_thinking[-1].strip() if boxed_in_thinking else None
    return {
        "n_self_corrections":     len(_RE_CORRECTION.findall(t)),
        "n_image_refs":           len(_RE_IMG_REF.findall(t)),
        "n_enumeration_steps":    len(_RE_ENUM.findall(t)),
        "n_backtracks":           len(_RE_BACKTRACK.findall(t)),
        "n_boxed_in_thinking":    len(boxed_in_thinking),
        "answer_flipped_in_trace": bool(
            last_mid and extracted and last_mid != extracted
        ),
    }

# ── Logprob stats (compact summary saved inline) ───────────────────────────────

def compute_logprob_stats(lp_content: list) -> dict:
    if not lp_content:
        return {}
    lps = [e["logprob"] for e in lp_content if e.get("logprob") is not None]
    if not lps:
        return {}
    mean_lp = sum(lps) / len(lps)
    var_lp  = sum((x - mean_lp) ** 2 for x in lps) / len(lps)

    entropies = []
    for entry in lp_content:
        top = entry.get("top_logprobs") or []
        if len(top) >= 2:
            probs = [math.exp(t["logprob"]) for t in top if t.get("logprob") is not None]
            s = sum(probs)
            if s > 0:
                probs = [p / s for p in probs]
                entropies.append(-sum(p * math.log(p + 1e-12) for p in probs if p > 0))

    return {
        "logprob_mean":  round(mean_lp, 5),
        "logprob_var":   round(var_lp,  5),
        "logprob_min":   round(min(lps), 5),
        "entropy_mean":  round(sum(entropies) / len(entropies), 5) if entropies else None,
        "entropy_max":   round(max(entropies), 5) if entropies else None,
    }

# ── Model API call ─────────────────────────────────────────────────────────────

def call_model(url: str, model: str, question: str, img_uri: str,
               enable_thinking: bool = True) -> tuple[dict, float]:
    payload = {
        "model":    model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img_uri}},
            {"type": "text",      "text": question},
        ]}],
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "max_tokens":  MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_k":       TOP_K,
        "top_p":       TOP_P,
        "logprobs":    True,
        "top_logprobs": N_TOP_LOGPROBS,
    }
    t0 = time.time()
    r  = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=7200)
    elapsed = time.time() - t0
    r.raise_for_status()
    return r.json(), elapsed

# ── Per-sample worker ──────────────────────────────────────────────────────────

def process_sample(
    item:      dict,
    img_uri:   str,
    img_props: dict,
    model:     str,
    url:       str,
    pass_idx:  int,
    compact_fh,
    heavy_fh,
    lock:      threading.Lock,
    counter:   list,  # [n_done, n_total] shared mutable
    enable_thinking: bool = True,
) -> dict:

    task_id = item["taskId"]
    q, gt   = build_question(item)

    try:
        resp, elapsed = call_model(url, model, q, img_uri, enable_thinking)
        choice = resp["choices"][0]
        msg    = choice["message"]
        usage  = resp.get("usage", {})

        thinking  = (msg.get("reasoning_content") or "").strip()
        answer    = (msg.get("content") or "").strip()
        extracted = extract_boxed_answer(answer) or extract_boxed_answer(thinking)
        finish_r  = choice.get("finish_reason")

        prompt_tok = usage.get("prompt_tokens")
        comp_tok   = usage.get("completion_tokens")

        lp_content = (choice.get("logprobs") or {}).get("content") or []
        lp_stats   = compute_logprob_stats(lp_content)
        rsn_feats  = extract_reasoning_features(thinking, extracted)

        # Approximate token split via char proportion
        total_chars      = len(thinking) + len(answer) or 1
        think_tok_approx = round(comp_tok * len(thinking) / total_chars) if comp_tok else None
        ans_tok_approx   = round(comp_tok * len(answer)   / total_chars) if comp_tok else None

        # MCQ-only prelim correctness (exact match on letter) — blank always None until judge
        is_correct_prelim = None
        if item["ansType"] == "choice" and extracted:
            is_correct_prelim = int(extracted.strip().upper() == gt.strip().upper())

        compact = {
            # ── identity ──────────────────────────────────────────────────────
            "taskId":       task_id,
            "pass_idx":     pass_idx,
            "condition":    "a0_nothink" if not enable_thinking else "standard",
            "type":         item["type"],
            "subtype":      item["subtype"],
            "ansType":      item["ansType"],
            "image_file":   item["image"],
            "question_sent": q,
            "options":      item.get("options", []),
            "gt_answer":    gt,
            "gold_coT":     item.get("coT"),
            # ── model output ──────────────────────────────────────────────────
            "thinking_trace":     thinking,
            "answer_text":        answer,
            "extracted_answer":   extracted,
            "finish_reason":      finish_r,
            "is_correct_prelim":  is_correct_prelim,
            "judge_result":       None,   # filled by run_judge.py
            "judge_raw":          None,
            # ── token accounting ──────────────────────────────────────────────
            "prompt_tokens":             prompt_tok,
            "completion_tokens":         comp_tok,
            "n_image_tokens_approx":     260,   # verified ~258-262 in smoke test; exact via HF in Phase 2
            "visual_token_ratio_approx": round(260 / prompt_tok, 4) if prompt_tok else None,
            "reasoning_tokens_approx":   think_tok_approx,
            "answer_tokens_approx":      ans_tok_approx,
            "reasoning_chars":           len(thinking),
            "answer_chars":              len(answer),
            # ── logprob stats (compact summary) ───────────────────────────────
            **lp_stats,
            # ── reasoning-structure features ──────────────────────────────────
            **rsn_feats,
            # ── image properties ──────────────────────────────────────────────
            **img_props,
            # ── timing ────────────────────────────────────────────────────────
            "inference_time_s":  round(elapsed, 3),
            "tokens_per_second": round(comp_tok / elapsed, 2) if comp_tok and elapsed > 0 else None,
        }

        heavy = {
            "taskId":          task_id,
            "pass_idx":        pass_idx,
            "logprobs_content": lp_content,   # full per-token sequence: [{token, logprob, top_logprobs}]
        }

    except Exception as e:
        compact = {
            "taskId": task_id, "pass_idx": pass_idx,
            "type": item["type"], "subtype": item["subtype"], "ansType": item["ansType"],
            "image_file": item["image"], "question_sent": q, "gt_answer": gt,
            "error": str(e), **img_props,
        }
        heavy = {"taskId": task_id, "pass_idx": pass_idx, "error": str(e)}

    with lock:
        compact_fh.write(json.dumps(compact) + "\n")
        compact_fh.flush()
        heavy_fh.write(json.dumps(heavy) + "\n")
        heavy_fh.flush()
        counter[0] += 1
        n_done, n_total = counter
        ok  = "✓" if compact.get("is_correct_prelim") == 1 else (
              "✗" if compact.get("is_correct_prelim") == 0 else "?")
        trunc = "TRUNC" if compact.get("finish_reason") == "length" else ""
        print(
            f"  [{n_done:3d}/{n_total}] id={task_id:4d} {item['subtype'][:22]:22s} "
            f"{ok} ans={str(compact.get('extracted_answer',''))[:8]:8s} "
            f"gt={str(gt)[:8]:8s} "
            f"think={len(thinking)//1000}k "
            f"comp={comp_tok or '?'}tok "
            f"{compact.get('inference_time_s','?')}s {trunc}",
            flush=True,
        )

    return compact

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",     type=int, default=30000)
    ap.add_argument("--model",    default=None, help="auto-detected if omitted")
    ap.add_argument("--data-dir", required=True, help="dir with meta_data.jsonl + images/")
    ap.add_argument("--out-dir",  required=True)
    ap.add_argument("--n-passes", type=int, default=N_PASSES)
    ap.add_argument("--no-thinking", action="store_true",
                    help="A0 condition: enable_thinking=False (direct answer, no reasoning trace).")
    args = ap.parse_args()
    enable_thinking = not args.no_thinking

    url = f"http://localhost:{args.port}"
    model = args.model or requests.get(f"{url}/v1/models", timeout=10).json()["data"][0]["id"]
    print(f"Model : {model}")
    print(f"OutDir: {args.out_dir}")
    print(f"Thinking: {'ON (standard)' if enable_thinking else 'OFF (A0 no-think)'}\n")

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = [json.loads(l) for l in open(data_dir / "meta_data.jsonl") if l.strip()]
    print(f"Loaded {len(items)} items  ({sum(1 for i in items if i['ansType']=='blank')} blank, "
          f"{sum(1 for i in items if i['ansType']=='choice')} choice)\n")

    print("Pre-computing image URIs and properties...")
    img_uris  = {item["taskId"]: encode_image(data_dir / item["image"]) for item in items}
    img_props = {item["taskId"]: get_image_properties(data_dir / item["image"]) for item in items}
    print("Done.\n")

    lock = threading.Lock()

    for pass_idx in range(1, args.n_passes + 1):
        compact_path = out_dir / f"results_run{pass_idx}.jsonl"
        heavy_path   = out_dir / f"logprobs_run{pass_idx}.jsonl"

        done_ids = set()
        if compact_path.exists():
            for line in open(compact_path):
                try:
                    r = json.loads(line)
                    if "error" not in r:
                        done_ids.add(r["taskId"])
                except Exception:
                    pass

        todo = [it for it in items if it["taskId"] not in done_ids]
        if not todo:
            print(f"Pass {pass_idx}: all {len(items)} already done, skipping.\n")
            continue

        print(f"{'='*72}")
        print(f"Pass {pass_idx}/{args.n_passes}  —  {len(todo)}/{len(items)} samples to run")
        print(f"{'='*72}")

        counter = [0, len(todo)]
        t_pass  = time.time()

        with open(compact_path, "a") as cfh, open(heavy_path, "a") as hfh:
            with ThreadPoolExecutor(max_workers=N_CONCURRENT) as pool:
                futures = [
                    pool.submit(
                        process_sample,
                        item, img_uris[item["taskId"]], img_props[item["taskId"]],
                        model, url, pass_idx, cfh, hfh, lock, counter,
                        enable_thinking,
                    )
                    for item in todo
                ]
                results = [f.result() for f in futures]

        completed  = [r for r in results if "error" not in r]
        truncated  = [r for r in completed if r.get("finish_reason") == "length"]
        no_extract = [r for r in completed if r.get("extracted_answer") is None]
        mcq_done   = [r for r in completed if r.get("ansType") == "choice"]
        mcq_right  = [r for r in mcq_done  if r.get("is_correct_prelim") == 1]

        print(f"\nPass {pass_idx} done in {(time.time()-t_pass)/60:.1f} min")
        print(f"  completed  : {len(completed)}/{len(todo)}")
        print(f"  truncated  : {len(truncated)}  (finish_reason=length — logged, will score False at judge)")
        print(f"  no boxed   : {len(no_extract)}")
        print(f"  MCQ prelim : {len(mcq_right)}/{len(mcq_done)} correct (exact letter match, MCQ only)")
        print(f"  compact    : {compact_path}")
        print(f"  logprobs   : {heavy_path}\n")

    print("All passes complete. Run run_judge.py next.")


if __name__ == "__main__":
    main()
