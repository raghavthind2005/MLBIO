#!/usr/bin/env python3
"""
BabyVision A3 — forced-LONG condition (s1-style single-trace budget forcing).

Goal: grow the reasoning trace at a FIXED input, then read out (Phase 2) whether
visual grounding decays as the single continuous <|channel>thought block lengthens
("think-longer-see-less"). This is NOT re-examination — the image is shown once,
never re-injected; we only force the model to keep thinking before it answers.

Mechanism (faithful to s1, Muennighoff et al. 2025):
  - Drive sglang's raw /generate (NOT chat) so we own the channel markers.
  - Generate the thought channel with stop="<channel|>".
  - Each time the model tries to CLOSE the thought channel before the token floor,
    suppress the close, append " Wait", and continue the SAME trace.
  - After the floor is reached (or MAX_FORCES hit, or ceiling), append "<channel|>"
    to close the channel and force the final answer.

Why raw /generate and not chat: the chat endpoint + gemma4 reasoning-parser
auto-manage the channel markers, so there is no seam to suppress the close and
continue one continuous trace. /generate gives us the raw string; we parse the
channels ourselves (delimiters confirmed from the rethink trace format:
opener "<|channel>thought\\n", closer "<channel|>").

Output schema mirrors run_infer.py (results_run1.jsonl + logprobs_run1.jsonl) so
run_judge.py and the analysis scripts work unchanged. Single pass only.

SPIKE FIRST:  python run_infer_a3.py --spike 10 --model-path <path>   (verbose, asserts mechanism)
FULL:         python run_infer_a3.py --data-dir ... --out-dir ... --model-path <path>
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

# Reuse validated helpers from the baseline script (same directory).
sys.path.insert(0, str(Path(__file__).parent))
from run_infer import (  # noqa: E402
    PROMPT_SUFFIX,
    TEMPERATURE, TOP_K, TOP_P, N_TOP_LOGPROBS,
    build_question, extract_boxed_answer,
    extract_reasoning_features, get_image_properties,
)

# ── Channel delimiters (this model's thinking format) ───────────────────────────
# Confirmed from the 2556704 spike: after "<|turn>model\n" the model thinks directly
# (no per-response open token in the template — turn delims are <|turn> / <turn|>) and
# closes the thought with "<channel|>". Only THINK_CLOSE matters for the logic: we
# stop on it, suppress it to inject "Wait", and re-append it to force the answer.
THINK_OPEN  = "<|channel>thought\n"   # cosmetic only (not present in template)
THINK_CLOSE = "<channel|>"            # real thinking-close; stop=[THINK_CLOSE] fires here

# ── Budget-forcing knobs ────────────────────────────────────────────────────────
# v2 (genuine long arm): standard's natural thinking median is ~7000 tok, so the old
# 4000 floor left 251/388 samples un-forced and the trace capped at one 8192 segment.
# To make A3 *exceed* standard, the floor is lifted well above standard's median and
# the loop now continues across segments (instead of breaking at the first 8192-token
# segment) until the floor is reached or the ceiling hit. MAX_FORCES is raised so
# stubborn early-closers get enough "Wait" injections to actually reach the floor.
MIN_THINKING_TOKENS = 12000   # keep forcing/continuing until the trace reaches this floor
MAX_FORCES          = 30      # safety cap on number of "Wait" injections (was 8)
MAX_THINKING_TOKENS = 32768   # hard ceiling on the thought channel
SEGMENT_MAX         = 8192    # max_new_tokens per /generate segment
ANSWER_BUDGET       = 4096    # tokens for the final answer after the channel closes
WAIT_STR            = " Wait"

N_CONCURRENT = 6              # multi-segment + multimodal → a bit lighter than baseline


# ── sglang raw /generate ─────────────────────────────────────────────────────────

def sgl_generate(base_url, prompt, image_paths, max_new_tokens,
                 stop=None, return_logprob=True):
    payload = {
        "text":        prompt,
        "image_data":  image_paths,           # list of absolute file paths
        "sampling_params": {
            "temperature":         TEMPERATURE,
            "top_k":               TOP_K,
            "top_p":               TOP_P,
            "max_new_tokens":      max_new_tokens,
            "stop":                stop or [],
            "skip_special_tokens": False,      # we need the channel markers in the text
        },
        "return_logprob":   return_logprob,
        "top_logprobs_num": N_TOP_LOGPROBS if return_logprob else 0,
        # NB: do NOT set logprob_start_len=0 — that forces logprobs over the entire
        # INPUT prompt. On the answer step the input is the full grown trace (~7k+
        # tokens) × ~262k vocab → a 14.6 GiB logits all-gather → CUDA OOM that kills
        # the server. Omitting it returns OUTPUT-token logprobs only (computed cheaply
        # during decode), which is exactly the generated thinking+answer we want.
    }
    r = requests.post(f"{base_url}/generate", json=payload, timeout=7200)
    r.raise_for_status()
    return r.json()


def _finish(meta):
    fr = meta.get("finish_reason") or {}
    if isinstance(fr, dict):
        return fr.get("type"), fr.get("matched")
    return str(fr), None


# ── Logprob stats over /generate format (output_token_logprobs / output_top_logprobs)
# /generate returns [[logprob, token_id, token_text], ...] and top is a list of those.

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


# ── Prompt construction (HF chat template → raw string for /generate) ────────────

def build_base_prompt(processor, question, enable_thinking=True):
    """Returns the raw prompt string up to the start of the model's turn.
    Image is a placeholder here; the real pixels go via image_data to sglang."""
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
        # processor doesn't accept enable_thinking — fall back
        return processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )


# ── Core: single-trace forced-long generation ────────────────────────────────────

def forced_long(base_url, base_prompt, image_paths, verbose=False):
    """Grow one continuous thought channel via Wait-injection, then force the answer."""
    cur_prompt   = base_prompt
    thinking     = ""
    forces       = 0
    think_tokens = 0
    seg_finishes = []
    think_token_lps, think_top_lps = [], []

    while True:
        remaining = MAX_THINKING_TOKENS - think_tokens
        if remaining <= 0:
            seg_finishes.append("ceiling")
            break

        out  = sgl_generate(base_url, cur_prompt, image_paths,
                            max_new_tokens=min(remaining, SEGMENT_MAX),
                            stop=[THINK_CLOSE])
        meta = out["meta_info"]
        seg  = out["text"]
        ftype, matched = _finish(meta)

        thinking     += seg
        cur_prompt   += seg
        think_tokens += meta.get("completion_tokens") or 0
        think_token_lps += meta.get("output_token_logprobs") or []
        think_top_lps   += meta.get("output_top_logprobs") or []
        seg_finishes.append(f"{ftype}:{matched}")

        if verbose:
            print(f"      seg{len(seg_finishes)}: +{meta.get('completion_tokens')}tok "
                  f"fin={ftype}/{matched!r} think_total={think_tokens} forces={forces}")
            print(f"        tail: ...{seg[-160:]!r}")

        closed = (ftype == "stop" and matched and THINK_CLOSE in str(matched))

        if closed:
            if think_tokens < MIN_THINKING_TOKENS and forces < MAX_FORCES:
                thinking   += WAIT_STR        # suppress close, push the SAME trace longer
                cur_prompt += WAIT_STR
                forces     += 1
                continue
            break                              # floor reached (or forces exhausted) → close
        if ftype == "length":
            # Segment filled without the model closing the channel. If we're still
            # below the floor, keep generating the SAME trace across the next segment
            # (this is what makes A3 genuinely long); otherwise stop and force the answer.
            if think_tokens < MIN_THINKING_TOKENS:
                continue
            break
        # any other stop (e.g. EOS without closing) → stop forcing
        break

    # Force the answer: close the channel, then generate the final content.
    answer_prompt = cur_prompt + THINK_CLOSE
    ans_out  = sgl_generate(base_url, answer_prompt, image_paths,
                           max_new_tokens=ANSWER_BUDGET, stop=None)
    ans_meta = ans_out["meta_info"]
    answer   = ans_out["text"]
    ans_ftype, _ = _finish(ans_meta)

    all_token_lps = think_token_lps + (ans_meta.get("output_token_logprobs") or [])
    all_top_lps   = think_top_lps   + (ans_meta.get("output_top_logprobs") or [])

    return {
        "thinking":        thinking.strip(),
        "answer":          answer.strip(),
        "n_forces":        forces,
        "thinking_tokens": think_tokens,
        "answer_tokens":   ans_meta.get("completion_tokens") or 0,
        "seg_finishes":    seg_finishes,
        "answer_finish":   ans_ftype,
        "token_lps":       all_token_lps,
        "top_lps":         all_top_lps,
    }


# ── Per-sample worker ─────────────────────────────────────────────────────────────

def process_sample(item, img_path, img_props, base_prompt, base_url,
                   compact_fh, heavy_fh, lock, counter):
    task_id = item["taskId"]
    q, gt   = build_question(item)
    try:
        t0  = time.time()
        res = forced_long(base_url, base_prompt, [str(img_path)])
        elapsed = time.time() - t0

        thinking  = res["thinking"]
        answer    = res["answer"]
        extracted = extract_boxed_answer(answer) or extract_boxed_answer(thinking)
        lp_stats  = compute_logprob_stats_gen(res["token_lps"], res["top_lps"])
        rsn_feats = extract_reasoning_features(thinking, extracted)

        is_correct_prelim = None
        if item["ansType"] == "choice" and extracted:
            is_correct_prelim = int(extracted.strip().upper() == gt.strip().upper())

        comp_tok = res["thinking_tokens"] + res["answer_tokens"]
        compact = {
            "taskId":      task_id, "pass_idx": 1, "condition": "a3_forced_long",
            "type":        item["type"], "subtype": item["subtype"],
            "ansType":     item["ansType"], "image_file": item["image"],
            "question_sent": q, "options": item.get("options", []),
            "gt_answer":   gt, "gold_coT": item.get("coT"),
            "thinking_trace": thinking, "answer_text": answer,
            "extracted_answer": extracted,
            "finish_reason": res["answer_finish"],
            "is_correct_prelim": is_correct_prelim,
            "judge_result": None, "judge_raw": None,
            # ── A3-specific budget-forcing accounting ─────────────────────────
            "n_forces":             res["n_forces"],
            "thinking_tokens_a3":   res["thinking_tokens"],
            "answer_tokens_a3":     res["answer_tokens"],
            "completion_tokens":    comp_tok,
            "seg_finishes":         res["seg_finishes"],
            "n_image_tokens_approx": 260,
            "reasoning_chars":      len(thinking),
            "answer_chars":         len(answer),
            **lp_stats, **rsn_feats, **img_props,
            "inference_time_s":  round(elapsed, 3),
            "tokens_per_second": round(comp_tok / elapsed, 2) if comp_tok and elapsed > 0 else None,
        }
        heavy = {
            "taskId": task_id, "pass_idx": 1,
            "token_logprobs": res["token_lps"],   # [[logprob, token_id, token_text], ...]
            "top_logprobs":   res["top_lps"],
        }
    except Exception as e:
        compact = {
            "taskId": task_id, "pass_idx": 1, "condition": "a3_forced_long",
            "type": item["type"], "subtype": item["subtype"], "ansType": item["ansType"],
            "image_file": item["image"], "question_sent": q, "gt_answer": gt,
            "error": str(e), **img_props,
        }
        heavy = {"taskId": task_id, "pass_idx": 1, "error": str(e)}

    with lock:
        compact_fh.write(json.dumps(compact) + "\n"); compact_fh.flush()
        heavy_fh.write(json.dumps(heavy) + "\n");      heavy_fh.flush()
        counter[0] += 1
        n_done, n_total = counter
        ok = "✓" if compact.get("is_correct_prelim") == 1 else (
             "✗" if compact.get("is_correct_prelim") == 0 else "?")
        print(f"  [{n_done:3d}/{n_total}] id={task_id:4d} {item['subtype'][:20]:20s} {ok} "
              f"forces={compact.get('n_forces','?')} think={compact.get('thinking_tokens_a3','?')}tok "
              f"ans={str(compact.get('extracted_answer',''))[:8]:8s} gt={str(gt)[:8]:8s} "
              f"{compact.get('inference_time_s','?')}s", flush=True)
    return compact


# ── Spike: validate the mechanism on a few samples (verbose, no full output) ──────

def run_spike(base_url, processor, items, n, enable_thinking):
    print(f"\n{'='*72}\nSPIKE — validating single-trace budget forcing on {n} samples\n{'='*72}")
    sample = items[:n]
    ok_long = ok_answer = 0
    floor_target = int(0.75 * MIN_THINKING_TOKENS)   # "long enough" = reached ~3/4 of floor
    for i, item in enumerate(sample):
        q, gt = build_question(item)
        base_prompt = build_base_prompt(processor, q, enable_thinking)
        if i == 0:
            print(f"\n--- base_prompt tail (last 500 chars) ---\n...{base_prompt[-500:]}\n"
                  f"--- (THINK_OPEN present in template? {THINK_OPEN.strip() in base_prompt}) ---")
        print(f"\n[{i+1}/{n}] id={item['taskId']} {item['subtype']}  gt={gt!r}")
        res = forced_long(base_url, base_prompt, [str(item['_img_path'])], verbose=True)
        extracted = extract_boxed_answer(res["answer"]) or extract_boxed_answer(res["thinking"])
        print(f"    → forces={res['n_forces']}  think_tokens={res['thinking_tokens']}  "
              f"answer_tokens={res['answer_tokens']}  finish={res['answer_finish']}")
        print(f"    → extracted={extracted!r}   answer_tail=...{res['answer'][-160:]!r}")
        # Genuine long arm: the trace must actually approach the floor (via Wait OR
        # segment-continuation), not merely have fired ≥1 Wait.
        if res["thinking_tokens"] >= floor_target:  ok_long += 1
        if extracted is not None:                   ok_answer += 1

    print(f"\n{'='*72}\nSPIKE SUMMARY")
    print(f"  reached ≥{floor_target} think tok : {ok_long}/{n}   (trace genuinely long, floor={MIN_THINKING_TOKENS})")
    print(f"  boxed answer            : {ok_answer}/{n}   (answer extractable after forced close)")
    ok = ok_long >= max(1, n // 2) and ok_answer >= max(1, n // 2)
    print(f"  VERDICT        : {'PASS — safe to launch full run' if ok else 'FAIL — inspect above before full run'}")
    print(f"{'='*72}\n")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",       type=int, default=30000)
    ap.add_argument("--model",      default=None, help="served model id (auto-detected)")
    ap.add_argument("--model-path", required=True, help="HF path for AutoProcessor (prompt building)")
    ap.add_argument("--data-dir",   required=True)
    ap.add_argument("--out-dir",    default=None, help="required unless --spike")
    ap.add_argument("--spike",      type=int, default=0, metavar="N",
                    help="Validate the mechanism on N samples (verbose, no output files).")
    ap.add_argument("--no-thinking-template", action="store_true",
                    help="Build the base prompt without enable_thinking (debug only).")
    args = ap.parse_args()

    base_url = f"http://localhost:{args.port}"
    model = args.model or requests.get(f"{base_url}/v1/models", timeout=10).json()["data"][0]["id"]
    enable_thinking = not args.no_thinking_template

    from transformers import AutoProcessor
    print(f"Loading processor: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)

    data_dir = Path(args.data_dir)
    items = [json.loads(l) for l in open(data_dir / "meta_data.jsonl") if l.strip()]
    for it in items:
        it["_img_path"] = data_dir / it["image"]
    print(f"Model: {model}  |  Loaded {len(items)} items  |  thinking_template={enable_thinking}")

    if args.spike:
        ok = run_spike(base_url, processor, items, args.spike, enable_thinking)
        sys.exit(0 if ok else 2)

    if not args.out_dir:
        ap.error("--out-dir is required for a full run (omit only with --spike)")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    compact_path = out_dir / "results_run1.jsonl"
    heavy_path   = out_dir / "logprobs_run1.jsonl"

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

    # Pre-build prompts + image properties once.
    print("Pre-building prompts and image properties...")
    base_prompts = {it["taskId"]: build_base_prompt(processor, build_question(it)[0], enable_thinking)
                    for it in todo}
    img_props    = {it["taskId"]: get_image_properties(it["_img_path"]) for it in todo}
    print("Done.\n")

    lock    = threading.Lock()
    counter = [0, len(todo)]
    t_start = time.time()
    with open(compact_path, "a") as cfh, open(heavy_path, "a") as hfh:
        with ThreadPoolExecutor(max_workers=N_CONCURRENT) as pool:
            futures = [pool.submit(process_sample, it, it["_img_path"], img_props[it["taskId"]],
                                   base_prompts[it["taskId"]], base_url, cfh, hfh, lock, counter)
                       for it in todo]
            results = [f.result() for f in futures]

    completed = [r for r in results if "error" not in r]
    forced    = [r for r in completed if (r.get("n_forces") or 0) > 0]
    no_ans    = [r for r in completed if r.get("extracted_answer") is None]
    print(f"\nA3 done in {(time.time()-t_start)/60:.1f} min")
    print(f"  completed : {len(completed)}/{len(todo)}")
    print(f"  forced ≥1 : {len(forced)}  (Wait actually injected)")
    print(f"  no boxed  : {len(no_ans)}")
    print(f"  results   : {compact_path}\n  logprobs  : {heavy_path}")
    print("Run run_judge.py on this dir next (--passes 1).")


if __name__ == "__main__":
    main()
