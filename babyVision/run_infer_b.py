#!/usr/bin/env python3
"""
BabyVision B1/B2 — two-turn reconsider conditions.

B1 (--reinject):    image present in BOTH turn 1 and turn 2 (re-grounding).
B2 (--no-reinject): image in turn 1 only; turn 2 is text-only reconsider.

Protocol:
  Turn 1: [image] + question → model thinks + gives initial answer
  Turn 2: [image (B1) | text-only (B2)] + "Give your final answer in \\boxed{Answer}."
          → model (optionally re-)thinks + gives FINAL answer

The FULL turn-1 context (thinking trace + initial answer) is preserved in the
turn-2 prompt, so the model has its own prior reasoning in context when it
reconsiders. The only difference between B1 and B2 is whether fresh image tokens
appear during the second pass.

Scientific question:
  Does re-grounding (B1) restore visual attention and lift accuracy vs. no-reinject
  (B2)? Does B1 beat standard (single-turn) on the perception-heavy subtypes where
  single-turn thinking hurts?

Use --spike N to validate the 2-turn prompt format before a full run.
"""

import argparse
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from run_infer import (  # noqa: E402
    PROMPT_SUFFIX,
    TEMPERATURE, TOP_K, TOP_P, N_TOP_LOGPROBS,
    build_question, extract_boxed_answer,
    extract_reasoning_features, get_image_properties,
)

# ── Channel delimiters ────────────────────────────────────────────────────────
THINK_OPEN  = "<|channel>thought\n"   # model opens this itself after <|turn>model\n
THINK_CLOSE = "<channel|>"            # model closes thinking here

# ── Generation budgets ────────────────────────────────────────────────────────
TURN1_BUDGET   = 16384   # thinking + initial answer (generous: median ~7k in standard)
TURN2_BUDGET   = 8192    # re-examination: usually shorter, but give room for re-think
N_CONCURRENT   = 2       # triton SWA attention throws CUDA illegal-memory-access on long
                         # sequences under batching (crashed A3 v2 at concurrency 6). With
                         # 2 serial /generate calls per sample already, keep 2 in flight.

# Turn-2 prompt (same for B1 and B2; only the image presence differs)
TURN2_QUESTION = "Give your final answer in \\boxed{Answer}."


# ── sglang raw /generate ──────────────────────────────────────────────────────

def sgl_generate(base_url, prompt, image_paths, max_new_tokens,
                 stop=None, return_logprob=True):
    payload = {
        "text":        prompt,
        "image_data":  image_paths,
        "sampling_params": {
            "temperature":         TEMPERATURE,
            "top_k":               TOP_K,
            "top_p":               TOP_P,
            "max_new_tokens":      max_new_tokens,
            "stop":                stop or [],
            "skip_special_tokens": False,
        },
        "return_logprob":   return_logprob,
        "top_logprobs_num": N_TOP_LOGPROBS if return_logprob else 0,
        # NB: omit logprob_start_len — output-tokens only (input logprobs → OOM)
    }
    r = requests.post(f"{base_url}/generate", json=payload, timeout=7200)
    r.raise_for_status()
    return r.json()


def _finish(meta):
    fr = meta.get("finish_reason") or {}
    if isinstance(fr, dict):
        return fr.get("type"), fr.get("matched")
    return str(fr), None


# ── Logprob stats (/generate format: [[logprob, token_id, token_text], ...]) ──

def compute_logprob_stats_gen(token_lps, top_lps) -> dict:
    lps = [e[0] for e in token_lps if e and e[0] is not None]
    if not lps:
        return {}
    mean_lp = sum(lps) / len(lps)
    var_lp  = sum((x - mean_lp) ** 2 for x in lps) / len(lps)
    entropies = []
    for cand in (top_lps or []):
        probs = [math.exp(c[0]) for c in cand if c and c[0] is not None]
        s = sum(probs)
        if s > 0:
            probs = [p / s for p in probs]
            entropies.append(-sum(p * math.log(p + 1e-12) for p in probs if p > 0))
    return {
        "logprob_mean": round(mean_lp, 5),
        "logprob_var":  round(var_lp, 5),
        "logprob_min":  round(min(lps), 5),
        "entropy_mean": round(sum(entropies) / len(entropies), 5) if entropies else None,
        "entropy_max":  round(max(entropies), 5) if entropies else None,
    }


# ── Prompt building ───────────────────────────────────────────────────────────

def build_turn1_prompt(processor, question, enable_thinking=True):
    """Single-turn user prompt ending with <|turn>model\\n (generation trigger)."""
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": question},
    ]}]
    try:
        return processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )


def build_turn2_prompt(processor, question, t1_thinking, t1_answer,
                       reinject_image, enable_thinking=True):
    """Build full 2-turn prompt for /generate.

    The turn-1 model response (with thinking markers) is passed as the
    assistant message; apply_chat_template wraps it in <|turn>model…<turn|>.
    The turn-2 user message optionally includes a second image placeholder.

    Returns (prompt_string, n_images_total).
    """
    # Reconstruct turn-1 model output as the model would have generated it.
    # The model opens THINK_OPEN itself; we include it so the context is faithful.
    t1_model_content = f"{THINK_OPEN}{t1_thinking}{THINK_CLOSE}\n{t1_answer}"

    turn2_user_content = []
    if reinject_image:
        turn2_user_content.append({"type": "image"})
    turn2_user_content.append({"type": "text", "text": TURN2_QUESTION})

    messages = [
        {"role": "user",      "content": [{"type": "image"},
                                           {"type": "text", "text": question}]},
        {"role": "assistant", "content": t1_model_content},
        {"role": "user",      "content": turn2_user_content},
    ]
    try:
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
    n_images = 2 if reinject_image else 1
    return prompt, n_images


def split_thinking_answer(text: str):
    """Parse raw /generate output → (thinking, answer).

    Expected format:
      [THINK_OPEN]{thinking}[THINK_CLOSE]\\n{answer}[<turn|>?]
    THINK_OPEN may or may not be in the text (model opens it itself after the
    prompt ends with <|turn>model\\n, so it may not be echoed in the output).
    """
    text = text.rstrip()
    if text.endswith("<turn|>"):
        text = text[:-len("<turn|>")].rstrip()
    if THINK_CLOSE in text:
        left, right = text.split(THINK_CLOSE, 1)
        thinking = left.removeprefix(THINK_OPEN).strip()
        answer   = right.strip()
    else:
        thinking = ""
        answer   = text.strip()
    return thinking, answer


# ── Core 2-turn protocol ──────────────────────────────────────────────────────

def two_turn(base_url, processor, question, img_path, reinject_image,
             verbose=False):
    """Execute the 2-turn protocol; return raw fields (no exception handling)."""
    # ── Turn 1 ────────────────────────────────────────────────────────────────
    t1_prompt = build_turn1_prompt(processor, question, enable_thinking=True)
    t1_out    = sgl_generate(base_url, t1_prompt, [str(img_path)],
                             max_new_tokens=TURN1_BUDGET)
    t1_meta   = t1_out["meta_info"]
    t1_thinking, t1_answer = split_thinking_answer(t1_out["text"])

    if verbose:
        ft, _ = _finish(t1_meta)
        print(f"    T1 {t1_meta.get('completion_tokens')}tok  finish={ft}")
        print(f"       think_tail …{t1_thinking[-120:]!r}")
        print(f"       answer     …{t1_answer[:120]!r}")

    # ── Turn 2 ────────────────────────────────────────────────────────────────
    t2_prompt, n_imgs = build_turn2_prompt(processor, question,
                                            t1_thinking, t1_answer,
                                            reinject_image, enable_thinking=True)
    img_paths = [str(img_path)] * n_imgs

    if verbose:
        print(f"    T2 prompt tail …{t2_prompt[-300:]!r}  (n_imgs={n_imgs})")

    t2_out  = sgl_generate(base_url, t2_prompt, img_paths,
                           max_new_tokens=TURN2_BUDGET)
    t2_meta = t2_out["meta_info"]
    t2_thinking, t2_answer = split_thinking_answer(t2_out["text"])

    if verbose:
        ft, _ = _finish(t2_meta)
        print(f"    T2 {t2_meta.get('completion_tokens')}tok  finish={ft}")
        print(f"       think_tail …{t2_thinking[-120:]!r}")
        print(f"       answer     …{t2_answer[:120]!r}")

    return {
        "t1_thinking":    t1_thinking,   "t1_answer":   t1_answer,
        "t2_thinking":    t2_thinking,   "t2_answer":   t2_answer,
        "t1_tokens":      t1_meta.get("completion_tokens") or 0,
        "t2_tokens":      t2_meta.get("completion_tokens") or 0,
        "t1_finish":      _finish(t1_meta)[0],
        "t2_finish":      _finish(t2_meta)[0],
        "t1_token_lps":   t1_meta.get("output_token_logprobs") or [],
        "t1_top_lps":     t1_meta.get("output_top_logprobs") or [],
        "t2_token_lps":   t2_meta.get("output_token_logprobs") or [],
        "t2_top_lps":     t2_meta.get("output_top_logprobs") or [],
    }


# ── Per-sample worker ─────────────────────────────────────────────────────────

def process_sample_b(item, img_path, img_props, processor, base_url,
                     reinject_image, compact_fh, heavy_fh, lock, counter):
    task_id   = item["taskId"]
    q, gt     = build_question(item)
    condition = "b1_reinject" if reinject_image else "b2_noreinject"
    try:
        t0  = time.time()
        res = two_turn(base_url, processor, q, img_path, reinject_image)
        elapsed = time.time() - t0

        t1_extracted = (extract_boxed_answer(res["t1_answer"]) or
                        extract_boxed_answer(res["t1_thinking"]))
        extracted    = (extract_boxed_answer(res["t2_answer"]) or
                        extract_boxed_answer(res["t2_thinking"]) or
                        t1_extracted)

        comp_tok = res["t1_tokens"] + res["t2_tokens"]
        lp_stats = compute_logprob_stats_gen(
            res["t1_token_lps"] + res["t2_token_lps"],
            res["t1_top_lps"]   + res["t2_top_lps"],
        )
        rsn_feats = extract_reasoning_features(
            res["t1_thinking"] + res["t2_thinking"], extracted)

        is_correct_prelim = None
        if item["ansType"] == "choice" and extracted:
            is_correct_prelim = int(extracted.strip().upper() == gt.strip().upper())

        compact = {
            "taskId":      task_id,  "pass_idx": 1,  "condition": condition,
            "type":        item["type"],   "subtype":  item["subtype"],
            "ansType":     item["ansType"],"image_file": item["image"],
            "question_sent": q,  "options": item.get("options", []),
            "gt_answer":   gt,   "gold_coT": item.get("coT"),
            # Turn 1
            "turn1_thinking":          res["t1_thinking"],
            "turn1_answer_text":       res["t1_answer"],
            "turn1_extracted":         t1_extracted,
            "turn1_completion_tokens": res["t1_tokens"],
            "turn1_finish":            res["t1_finish"],
            # Turn 2 (canonical fields that judge + analyze expect)
            "thinking_trace":   res["t2_thinking"],
            "answer_text":      res["t2_answer"],
            "extracted_answer": extracted,
            "finish_reason":    res["t2_finish"],
            "is_correct_prelim": is_correct_prelim,
            "judge_result": None, "judge_raw": None,
            # Token accounting
            "completion_tokens":    comp_tok,
            "turn1_tokens":         res["t1_tokens"],
            "turn2_tokens":         res["t2_tokens"],
            "reinject_image":       reinject_image,
            "n_image_tokens_turn1": 260,
            "n_image_tokens_turn2": 260 if reinject_image else 0,
            "reasoning_chars":      len(res["t1_thinking"]) + len(res["t2_thinking"]),
            "answer_chars":         len(res["t2_answer"]),
            **lp_stats, **rsn_feats, **img_props,
            "inference_time_s":  round(elapsed, 3),
            "tokens_per_second": round(comp_tok / elapsed, 2) if comp_tok and elapsed else None,
        }
        heavy = {
            "taskId": task_id, "pass_idx": 1,
            "turn1_token_logprobs": res["t1_token_lps"],
            "turn1_top_logprobs":   res["t1_top_lps"],
            "turn2_token_logprobs": res["t2_token_lps"],
            "turn2_top_logprobs":   res["t2_top_lps"],
        }
    except Exception as e:
        compact = {
            "taskId": task_id, "pass_idx": 1, "condition": condition,
            "type": item["type"], "subtype": item["subtype"], "ansType": item["ansType"],
            "image_file": item["image"], "question_sent": q, "gt_answer": gt,
            "error": str(e), **img_props,
        }
        heavy = {"taskId": task_id, "pass_idx": 1, "error": str(e)}

    with lock:
        compact_fh.write(json.dumps(compact) + "\n"); compact_fh.flush()
        heavy_fh.write(json.dumps(heavy) + "\n");     heavy_fh.flush()
        counter[0] += 1
        n_done, n_total = counter
        ok = ("✓" if compact.get("is_correct_prelim") == 1 else
              "✗" if compact.get("is_correct_prelim") == 0 else "?")
        t1_extracted_str = str(compact.get("turn1_extracted") or "")
        t2_extracted_str = str(compact.get("extracted_answer") or "")
        print(f"  [{n_done:3d}/{n_total}] id={task_id:4d} {item['subtype'][:20]:20s} {ok} "
              f"t1={compact.get('turn1_tokens','?')} t2={compact.get('turn2_tokens','?')} "
              f"t1ans={t1_extracted_str[:6]:6s} t2ans={t2_extracted_str[:6]:6s} "
              f"gt={str(gt)[:8]:8s} {compact.get('inference_time_s','?')}s", flush=True)
    return compact


# ── Spike ─────────────────────────────────────────────────────────────────────

def run_spike(base_url, processor, items, n, reinject_image):
    label = "B1 reinject" if reinject_image else "B2 no-reinject"
    print(f"\n{'='*70}\nSPIKE — 2-turn protocol ({label}) on {n} samples\n{'='*70}")
    ok_t1 = ok_t2 = 0
    for i, item in enumerate(items[:n]):
        q, gt = build_question(item)
        print(f"\n[{i+1}/{n}] id={item['taskId']}  {item['subtype']}  gt={gt!r}")
        if i == 0:
            t1p = build_turn1_prompt(processor, q)
            print(f"  Turn-1 prompt tail: …{t1p[-200:]!r}")
        res = two_turn(base_url, processor, q, item["_img_path"], reinject_image, verbose=True)
        t1a = extract_boxed_answer(res["t1_answer"]) or extract_boxed_answer(res["t1_thinking"])
        t2a = extract_boxed_answer(res["t2_answer"]) or extract_boxed_answer(res["t2_thinking"])
        print(f"  → t1_extracted={t1a!r}   t2_extracted={t2a!r}")
        if t1a is not None: ok_t1 += 1
        if t2a is not None: ok_t2 += 1
    print(f"\n{'='*70}\nSPIKE SUMMARY ({label})")
    print(f"  turn-1 boxed answer : {ok_t1}/{n}")
    print(f"  turn-2 boxed answer : {ok_t2}/{n}")
    ok = ok_t2 >= max(1, n // 2)
    print(f"  VERDICT : {'PASS — safe to launch full run' if ok else 'FAIL — inspect above'}")
    print(f"{'='*70}\n")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",       type=int, default=30000)
    ap.add_argument("--model-path", required=True, help="HF path for AutoProcessor")
    ap.add_argument("--data-dir",   required=True)
    ap.add_argument("--out-dir",    default=None)
    ap.add_argument("--spike",      type=int, default=0, metavar="N")
    # Condition selector
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--reinject",    action="store_true",
                     help="B1: include image in turn 2 (re-grounding)")
    grp.add_argument("--no-reinject", action="store_true",
                     help="B2: text-only turn 2 (no re-grounding)")
    args = ap.parse_args()

    reinject_image = args.reinject
    base_url = f"http://localhost:{args.port}"

    from transformers import AutoProcessor
    print(f"Loading processor: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)

    data_dir = Path(args.data_dir)
    items = [json.loads(l) for l in open(data_dir / "meta_data.jsonl") if l.strip()]
    for it in items:
        it["_img_path"] = data_dir / it["image"]
    label = "B1 (reinject)" if reinject_image else "B2 (no-reinject)"
    print(f"Model at :{args.port}  |  {len(items)} items  |  condition={label}")

    if args.spike:
        ok = run_spike(base_url, processor, items, args.spike, reinject_image)
        sys.exit(0 if ok else 2)

    if not args.out_dir:
        ap.error("--out-dir required for a full run (omit only with --spike)")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compact_path = out_dir / "results_run1.jsonl"
    heavy_path   = out_dir / "logprobs_run1.jsonl"

    # Resume: skip taskIds already successfully written
    done = set()
    if compact_path.exists():
        for line in open(compact_path):
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add(r["taskId"])
            except Exception:
                pass
    todo = [it for it in items if it["taskId"] not in done]
    print(f"To run: {len(todo)}/{len(items)} (resume skipped {len(done)})\n")
    if not todo:
        print("Nothing to do."); return

    print("Pre-building prompts and image properties...")
    base_prompts = {it["taskId"]: build_turn1_prompt(processor, build_question(it)[0])
                    for it in todo}
    img_props    = {it["taskId"]: get_image_properties(it["_img_path"]) for it in todo}
    print("Done.\n")

    lock    = threading.Lock()
    counter = [0, len(todo)]
    t_start = time.time()

    with open(compact_path, "a") as cfh, open(heavy_path, "a") as hfh:
        with ThreadPoolExecutor(max_workers=N_CONCURRENT) as pool:
            futures = [
                pool.submit(process_sample_b, it, it["_img_path"],
                            img_props[it["taskId"]], processor, base_url,
                            reinject_image, cfh, hfh, lock, counter)
                for it in todo
            ]
            results = [f.result() for f in futures]

    completed = [r for r in results if "error" not in r]
    no_ans    = [r for r in completed if r.get("extracted_answer") is None]
    print(f"\n{label} done in {(time.time()-t_start)/60:.1f} min")
    print(f"  completed : {len(completed)}/{len(todo)}")
    print(f"  no boxed  : {len(no_ans)}")
    print(f"  results   : {compact_path}\n  logprobs  : {heavy_path}")
    print("Run run_judge.py on this dir next (--passes 1).")


if __name__ == "__main__":
    main()
