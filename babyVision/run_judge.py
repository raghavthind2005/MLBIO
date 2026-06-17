#!/usr/bin/env python3
"""
BabyVision judge — Qwen3-32B via sglang (text-only server).
Reads results_run{N}.jsonl files produced by run_infer.py,
fills judge_result (bool) and judge_raw (str), writes results_run{N}_judged.jsonl.

Judge model: Qwen3-32B  (thinking OFF — deterministic True/False output)
Judge prompt: exact LLM_JUDGE_PROMPT from paper (utils.py)
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

# ── Judge prompt (verbatim from paper's utils.py) ─────────────────────────────
LLM_JUDGE_PROMPT = """You are a careful and strict evaluator. You will be given:

1. **Question**
2. **Ground Truth Answer** (correct answer)
3. **Model Output** (answer from another model)

**Your goal:** Determine if the Model Output **accurately matches** the Ground Truth Answer in meaning.

* Matching means: the facts, entities, and key details are equivalent, even if phrasing differs.
* Not matching means: the Model Output is wrong, incomplete, contains extra incorrect facts, or changes the meaning.

**Process (internal reasoning):**

1. Read and understand the Question, Ground Truth Answer, and Model Output.
2. Ignore small wording differences, formatting, or synonyms.
3. If all factual content matches, conclude `1`. Otherwise, conclude `0`.

**Important:**

* Think through your decision step-by-step **internally** before responding.
* In your final output, return **only** True or False, with no extra text or explanation.

**Output format:**

True

or

False

**Input:**

Question: {question},
Ground Truth Answer: {groundtruth},
Model Output: {modeloutput}
"""

N_CONCURRENT = 16   # judge calls are cheap; more parallelism is fine


def call_judge(url: str, model: str, question: str, gt: str, extracted) -> tuple[bool, str]:
    """Returns (judge_result: bool, raw_response: str)."""
    prompt = LLM_JUDGE_PROMPT.format(
        question=question,
        groundtruth=gt,
        modeloutput=str(extracted) if extracted is not None else "NO ANSWER",
    )
    payload = {
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},  # Qwen3 thinking must be OFF
        "max_tokens":  64,    # True/False only
        "temperature": 0.0,   # deterministic
    }
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    raw = (r.json()["choices"][0]["message"]["content"] or "").strip().lower()

    if "true" in raw:
        return True, raw
    if "false" in raw:
        return False, raw
    print(f"  ⚠ Judge unparseable: {raw!r}")
    return False, raw  # conservative: unparseable → wrong


def judge_record(
    rec:     dict,
    url:     str,
    model:   str,
    lock:    threading.Lock,
    counter: list,
) -> dict:
    if "error" in rec:
        rec["judge_result"] = False
        rec["judge_raw"]    = "SKIP_ERROR"
        return rec

    # Already judged (resume support)
    if rec.get("judge_result") is not None:
        return rec

    try:
        result, raw = call_judge(
            url, model,
            rec["question_sent"],
            rec["gt_answer"],
            rec.get("extracted_answer"),
        )
        rec["judge_result"] = result
        rec["judge_raw"]    = raw
    except Exception as e:
        rec["judge_result"] = False
        rec["judge_raw"]    = f"ERROR:{e}"

    with lock:
        counter[0] += 1
        n_done, n_total = counter
        ok = "✓" if rec["judge_result"] else "✗"
        print(
            f"  [{n_done:4d}/{n_total}] id={rec['taskId']:4d} "
            f"{ok}  ans={str(rec.get('extracted_answer',''))[:10]:10s}  "
            f"gt={str(rec.get('gt_answer',''))[:10]:10s}  raw={rec['judge_raw'][:20]!r}",
            flush=True,
        )

    return rec


def judge_file(results_path: Path, url: str, model: str) -> None:
    out_path = results_path.with_name(results_path.stem + "_judged.jsonl")

    records = [json.loads(l) for l in open(results_path) if l.strip()]
    print(f"\n  {results_path.name}  →  {out_path.name}  ({len(records)} records)")

    lock    = threading.Lock()
    counter = [0, len(records)]

    with ThreadPoolExecutor(max_workers=N_CONCURRENT) as pool:
        futures = [pool.submit(judge_record, rec, url, model, lock, counter) for rec in records]
        judged  = [f.result() for f in futures]

    with open(out_path, "w") as f:
        for rec in judged:
            f.write(json.dumps(rec) + "\n")

    correct = sum(1 for r in judged if r.get("judge_result") is True)
    total   = sum(1 for r in judged if "error" not in r)
    print(f"\n  Accuracy: {correct}/{total} = {correct/total:.4f}" if total else "  No valid records.")
    print(f"  Written : {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",       type=int, default=30001, help="Qwen3-32B judge server port")
    ap.add_argument("--model",      default=None)
    ap.add_argument("--results-dir", required=True, help="dir containing results_run*.jsonl")
    ap.add_argument("--passes",     type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    url = f"http://localhost:{args.port}"
    model = args.model or requests.get(f"{url}/v1/models", timeout=10).json()["data"][0]["id"]
    print(f"Judge model : {model}")

    results_dir = Path(args.results_dir)
    for pass_idx in args.passes:
        path = results_dir / f"results_run{pass_idx}.jsonl"
        if not path.exists():
            print(f"  Skipping pass {pass_idx} — {path} not found.")
            continue
        judge_file(path, url, model)

    print("\nAll passes judged.")


if __name__ == "__main__":
    main()
