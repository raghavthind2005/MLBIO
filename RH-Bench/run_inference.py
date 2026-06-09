"""
Phase 1: Qwen3-VL-4B-Thinking inference on RH-Bench.

Paper: "More Thinking, Less Seeing?" arXiv:2505.21523
Thinking mode: --reasoning-parser qwen3-thinking (sglang flag)
Response format: reasoning_content (thinking) + content (final answer) returned separately
"""
import json, base64, re, os, time, requests
from pathlib import Path

DATASET_DIR = "/capstor/store/cscs/swissai/a0174/benchmarks/RH-Bench"
VLM_URL     = "http://localhost:30000/v1/chat/completions"
OUTPUT      = "/iopsstor/scratch/cscs/raghavthind/rh-bench/results/vlm_responses.json"


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


def query_vlm(question, image_path, retries=3):
    ext = Path(image_path).suffix[1:].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    b64 = encode_image(image_path)
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": question}
        ]}],
        "max_tokens": 16384,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    for i in range(retries):
        try:
            r = requests.post(VLM_URL, json=payload, timeout=300)
            msg = r.json()["choices"][0]["message"]
            # --reasoning-parser qwen3-thinking splits response into two fields:
            #   reasoning_content = thinking chain (inside <think>)
            #   content           = final answer only
            reasoning = msg.get("reasoning_content", "") or ""
            content   = msg.get("content", "") or ""
            # Reconstruct full response for storage
            full = f"<think>{reasoning}</think>\n{content}" if reasoning else content
            return full, content, bool(reasoning), len(reasoning.split()) if reasoning else 0
        except Exception as e:
            print(f"  Retry {i+1}: {e}")
            time.sleep(5)
    return "", "", False, 0


# Wait for VLM server
for _ in range(60):
    try:
        requests.get("http://localhost:30000/health", timeout=3)
        print("VLM server ready.")
        break
    except:
        time.sleep(5)

results = []
for subset, fname in [("halu", "halu_data.json"), ("reason", "reason_data.json")]:
    data = json.load(open(f"{DATASET_DIR}/{fname}"))
    for item in data:
        if item.get("question_type") == "free_from":
            item["question_type"] = "free_form"
    print(f"\n=== {subset}: {len(data)} questions ===")
    for i, item in enumerate(data):
        img_path = f"{DATASET_DIR}/{item['image']}"
        raw, clean, has_thinking, think_tokens = query_vlm(item["question"], img_path)
        rec = {
            "id": item["id"],
            "subset": subset,
            "question_type": item["question_type"],
            "question": item["question"],
            "image": item["image"],
            "gt_answer": item["answer"],
            "raw_response": raw,
            "clean_response": clean,
            "has_thinking": has_thinking,
            "thinking_tokens": think_tokens,
        }
        if item["question_type"] == "multi_choice":
            rec["pred_letter"] = extract_pred_letter(clean)
            rec["gt_letter"] = item["answer"].strip().upper() if re.match(r'^[A-E]$', item["answer"].strip(), re.IGNORECASE) else None
        results.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(data)} done")
            json.dump(results, open(OUTPUT, "w"), indent=2)

json.dump(results, open(OUTPUT, "w"), indent=2)
print(f"\nSaved {len(results)} results → {OUTPUT}")

think_count = sum(1 for r in results if r.get("has_thinking"))
avg_think = sum(r.get("thinking_tokens", 0) for r in results) / max(1, think_count)
print(f"Thinking chains: {think_count}/{len(results)} ({100*think_count/len(results):.0f}%)")
print(f"Avg thinking tokens: {avg_think:.0f}")
