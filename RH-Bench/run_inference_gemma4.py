"""
Phase 1: Gemma-4-31B-it inference on RH-Bench.

Paper: "More Thinking, Less Seeing?" arXiv:2505.21523
Model:  google/gemma-4-31B-it (architecture: gemma4, native sglang support)
Reasoning: sglang auto-detects gemma4 reasoning parser — thinking is ON by default.

Usage:
  python run_inference_gemma4.py              # full 1000-sample run
  python run_inference_gemma4.py --limit 10   # test on first 10 samples (5 halu + 5 reason)
"""
import json, base64, re, os, sys, time, requests, argparse
from pathlib import Path

# --- Config ---
DATASET_DIR = "/capstor/store/cscs/swissai/a0174/benchmarks/RH-Bench"
VLM_URL     = "http://localhost:30000/v1/chat/completions"
OUT_DIR     = "/iopsstor/scratch/cscs/raghavthind/rh-bench/results"
OUTPUT      = f"{OUT_DIR}/vlm_responses_gemma4.json"

# Gemma-4 sampling: use model defaults (temperature=1.0 triggers thinking mode)
# top_k=64, top_p=0.95 matches model's generation_config.json
MAX_TOKENS  = 16384
TEMPERATURE = 1.0
TOP_K       = 64
TOP_P       = 0.95

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None,
                    help="Limit samples per subset (e.g. --limit 5 tests 5 halu + 5 reason)")
args = parser.parse_args()

os.makedirs(OUT_DIR, exist_ok=True)


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def extract_pred_letter(text):
    for p in [
        r'\*\*[Aa]nswer[:\s]+\(?([A-E])\)?',
        r'[Aa]nswer[:\s]+\(?([A-E])\)?',
        r'[Tt]he\s+(?:correct\s+)?answer\s+is\s+\(?([A-E])\)?',
        r'[Cc]orrect\s+(?:answer|option)[:\s]+\(?([A-E])\)?',
    ]:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines:
        m = re.match(r'^["\']?\(?([A-E])\)?["\']?[.\s]*$', lines[-1], re.IGNORECASE)
        if m:
            return m.group(1).upper()
    letters = re.findall(r'\b([A-E])\b', text, re.IGNORECASE)
    return letters[-1].upper() if letters else None


# System prompt that triggers Gemma-4's thinking mode.
# Without this, Gemma-4-31B-it answers directly with no <think> traces.
SYSTEM_PROMPT = (
    "You are a careful and analytical visual reasoning assistant. "
    "For every question, think step by step before giving your final answer."
)


def query_vlm(question, image_path, retries=3):
    ext = Path(image_path).suffix[1:].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    b64 = encode_image(image_path)
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": question}
            ]}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        # chat_template_kwargs must be at TOP LEVEL of the JSON body.
        # Using requests.post directly (not OpenAI SDK), so extra_body
        # wrapping does NOT work — this key must be here directly.
        "chat_template_kwargs": {"enable_thinking": True},
    }
    for i in range(retries):
        try:
            r = requests.post(VLM_URL, json=payload, timeout=300)
            resp_json = r.json()

            # Debug: print raw message keys on first call to verify thinking field
            if i == 0 and len(resp_json.get("choices", [])) > 0:
                msg_keys = list(resp_json["choices"][0]["message"].keys())
                # Only print on very first sample overall
                import sys
                if not getattr(query_vlm, "_debug_printed", False):
                    print(f"  [DEBUG] message keys: {msg_keys}", flush=True)
                    query_vlm._debug_printed = True

            msg = resp_json["choices"][0]["message"]
            # sglang gemma4 reasoning parser separates:
            #   reasoning_content = thinking chain
            #   content           = final answer only
            reasoning = msg.get("reasoning_content", "") or ""
            content   = msg.get("content", "") or ""
            full = f"<think>{reasoning}</think>\n{content}" if reasoning else content
            return full, content, bool(reasoning), len(reasoning.split()) if reasoning else 0
        except Exception as e:
            print(f"  Retry {i+1}: {e}")
            time.sleep(5)
    return "", "", False, 0


# Wait for VLM server
print("Waiting for VLM server on port 30000...")
for _ in range(60):
    try:
        requests.get("http://localhost:30000/health", timeout=3)
        print("VLM server ready.")
        break
    except:
        time.sleep(5)

# ── Resume logic ────────────────────────────────────────────────────────────
# Load any existing results and skip items that already have a valid response.
# This lets us safely restart after a server crash without re-doing done work.
existing_results = {}
if os.path.exists(OUTPUT) and not args.limit:
    try:
        saved = json.load(open(OUTPUT))
        for r in saved:
            if r.get("clean_response", "").strip():  # non-empty = completed
                existing_results[(r["subset"], r["id"])] = r
        if existing_results:
            print(f"Resuming: found {len(existing_results)} already-completed samples, skipping them.")
    except Exception as e:
        print(f"Warning: could not load existing results ({e}), starting fresh.")

results = list(existing_results.values())  # seed with already-done items
total_start = time.time()

for subset, fname in [("halu", "halu_data.json"), ("reason", "reason_data.json")]:
    data = json.load(open(f"{DATASET_DIR}/{fname}"))
    for r in data:
        if r.get("question_type") == "free_from":
            r["question_type"] = "free_form"

    if args.limit:
        data = data[:args.limit]

    # Count how many in this subset are already done
    already_done = sum(1 for item in data if (subset, item["id"]) in existing_results)
    todo = len(data) - already_done
    print(f"\n=== {subset}: {len(data)} questions ({already_done} already done, {todo} to run) ===")

    for i, item in enumerate(data):
        # Skip if already completed successfully
        if (subset, item["id"]) in existing_results:
            continue

        img_path = f"{DATASET_DIR}/{item['image']}"
        t0 = time.time()
        raw, clean, has_thinking, think_words = query_vlm(item["question"], img_path)
        elapsed = time.time() - t0

        rec = {
            "id":            item["id"],
            "subset":        subset,
            "question_type": item["question_type"],
            "question":      item["question"],
            "image":         item["image"],
            "gt_answer":     item["answer"],
            "raw_response":  raw,
            "clean_response": clean,
            "has_thinking":  has_thinking,
            "thinking_words": think_words,
            "elapsed_s":     round(elapsed, 1),
        }
        if item["question_type"] == "multi_choice":
            rec["pred_letter"] = extract_pred_letter(clean)
            rec["gt_letter"]   = item["answer"].strip().upper() \
                                  if re.match(r'^[A-E]$', item["answer"].strip(), re.IGNORECASE) else None

        results.append(rec)

        # Per-sample progress line
        think_flag = f"thinking={think_words}w" if has_thinking else "NO_THINKING"
        print(f"  [{i+1}/{len(data)}] {think_flag} | {elapsed:.1f}s | "
              f"clean: {clean[:80].replace(chr(10),' ')!r}")

        # Save checkpoint every 50 samples
        if len(results) % 50 == 0:
            json.dump(results, open(OUTPUT, "w"), indent=2)

json.dump(results, open(OUTPUT, "w"), indent=2)

# Summary stats
think_count  = sum(1 for r in results if r["has_thinking"])
avg_words    = sum(r["thinking_words"] for r in results) / max(1, think_count)
avg_elapsed  = sum(r["elapsed_s"] for r in results) / len(results)
total_time   = time.time() - total_start

print(f"\n{'='*50}")
print(f"Saved {len(results)} results → {OUTPUT}")
print(f"Thinking ON:     {think_count}/{len(results)} ({100*think_count/len(results):.0f}%)")
print(f"Avg think words: {avg_words:.0f}")
print(f"Avg time/sample: {avg_elapsed:.1f}s")
print(f"Total time:      {total_time/60:.1f} min")
print(f"Projected full:  {avg_elapsed*1000/3600:.1f}h for 1000 samples")
print(f"{'='*50}")

# Print sample outputs for quality check
print("\n=== SAMPLE OUTPUTS (first 3) ===")
for rec in results[:3]:
    print(f"\n--- [{rec['subset']}] id={rec['id']} qtype={rec['question_type']} ---")
    print(f"Q: {rec['question'][:120]}")
    print(f"GT: {rec['gt_answer']}")
    if rec["has_thinking"]:
        # Show first 200 chars of thinking
        think_preview = rec["raw_response"].split("</think>")[0].replace("<think>","").strip()[:200]
        print(f"<think> (preview): {think_preview}...")
    print(f"Answer: {rec['clean_response'][:200]}")
    if rec.get("pred_letter"):
        print(f"Extracted letter: {rec['pred_letter']} | GT letter: {rec.get('gt_letter')}")
