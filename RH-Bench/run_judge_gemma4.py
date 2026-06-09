"""
Phase 2: Judge evaluation for Gemma-4-31B RH-Bench run.

Judge model: Qwen3-32B (text only, no image needed).
Judge port:  30001 (started separately after VLM inference completes).

Usage:
  python run_judge_gemma4.py              # judge all samples in vlm_responses_gemma4.json
  python run_judge_gemma4.py --limit 10   # judge first 10 (for test runs)
"""
import json, re, time, requests, argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None)
args = parser.parse_args()

BASE   = "/iopsstor/scratch/cscs/raghavthind/rh-bench/results"
INPUT  = f"{BASE}/vlm_responses_gemma4.json"
OUTPUT = f"{BASE}/judged_responses_gemma4.json"
JUDGE_URL = "http://localhost:30001/v1/chat/completions"

SYS_REASON = "You are an impartial evaluator. Evaluate answers strictly based on the provided criteria."
SYS_PERCEP = "You are an impartial evaluator. Evaluate multiple-choice answers carefully considering both the selected option and any provided reasoning."

REASON_MC_PROMPT = """You are an impartial evaluator assessing the correctness of a model's answer to a multiple-choice question.

Question: {question}
Model's Answer: {model_answer}
Correct Answer: {ground_truth}

Please evaluate whether the model's answer is correct by considering:
1. Whether the model's answer matches the correct answer exactly (e.g., same option letter).
2. If the model's answer is a value, whether it matches the value of the correct option.
3. Whether the model's reasoning (if provided) supports its answer.

Your response should be a JSON object with the following structure:
{{
    "is_correct": <true or false>,
    "reason": "<brief explanation of your evaluation>"
}}"""

REASON_FF_PROMPT = """You are an impartial evaluator assessing the correctness of a model's answer to a free-form question requiring a numerical answer.

Question: {question}
Model's Answer: {model_answer}
Correct Answer: {ground_truth}

Please evaluate whether the model's answer is correct by considering:
1. Whether the model's answer matches the correct answer exactly.
2. If not exact, whether the model's answer is numerically equivalent to the correct answer.
3. Whether the model's reasoning (if provided) supports its answer.

Your response should be a JSON object with the following structure:
{{
    "is_correct": <true or false>,
    "reason": "<brief explanation of your evaluation>"
}}"""

PERCEP_FF_PROMPT = """Please act as an impartial and objective judge to evaluate the presence and severity of hallucination in the response provided by a Large Multimodal Model (LMM) to the user question.

Evaluation Guidelines:
- Compare the LMM response with the standard human-generated answer.
- Identify any claims in the LMM response that are not supported by the image or question.
- The standard human-generated answer may not describe all objects or details comprehensively, so focus on whether the LMM introduces false claims rather than omitting details.
- Detailed reasoning or analysis by the LMM that is consistent with the image contents is acceptable and should not be considered hallucination.

Assign a rating from the following six levels:
Rating 5: No hallucination (response is fully consistent with image contents).
Rating 4: Minimal hallucination (minor, negligible false detail).
Rating 3: Mild hallucination (small false claim, non-critical).
Rating 2: Moderate hallucination (clear false claim affecting accuracy).
Rating 1: Severe hallucination (significant false claim).
Rating 0: Extreme hallucination (largely detached from image contents).

Question: {question}
Standard Human-Generated Answer: {ground_truth}
LMM Response to Evaluate: {model_answer}

Your response should be a JSON object with the following structure:
{{
    "hallucination_score": <integer from 0 to 5>,
    "reason": "<brief explanation of your evaluation>"
}}"""

PERCEP_MC_PROMPT = """Please evaluate whether the model's answer to the multiple-choice question is correct by considering:
1. Whether the model's answer matches the correct answer exactly (same option letter).
2. If the model's answer is a value, whether it matches the value of the correct option.
3. Whether the model's reasoning (if provided) supports its answer.

Question: {question}
Correct Answer: {ground_truth}
Model's Answer: {model_answer}

Your evaluation should be a JSON object with the following structure:
{{
    "is_correct": <boolean>,
    "reason": "<explanation of your evaluation>",
    "model_answer_extracted": "<the extracted answer from the model's response>"
}}"""


def strip_thinking(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def parse_json_response(text):
    text = strip_thinking(text).strip()
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    return {}


def query_judge(system_prompt, user_prompt, retries=3):
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    for i in range(retries):
        try:
            r = requests.post(JUDGE_URL, json=payload, timeout=90)
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  Retry {i+1}: {e}")
            time.sleep(5)
    return ""


# Wait for judge server on port 30001
print("Waiting for judge server on port 30001...")
for _ in range(120):
    try:
        requests.get("http://localhost:30001/health", timeout=3)
        print("Judge server ready.")
        break
    except:
        time.sleep(5)

results = json.load(open(INPUT))
for r in results:
    if r.get("question_type") == "free_from":
        r["question_type"] = "free_form"

if args.limit:
    results = results[:args.limit]

print(f"Judging {len(results)} entries...")
judged = 0

for i, rec in enumerate(results):
    qtype    = rec["question_type"]
    subset   = rec["subset"]
    model_ans = rec["clean_response"]
    gt       = rec["gt_answer"]
    question = rec["question"]

    t0 = time.time()

    if subset == "reason":
        if qtype == "multi_choice":
            prompt = REASON_MC_PROMPT.format(question=question, model_answer=model_ans, ground_truth=gt)
        else:
            prompt = REASON_FF_PROMPT.format(question=question, model_answer=model_ans, ground_truth=gt)
        raw = query_judge(SYS_REASON, prompt)
        parsed = parse_json_response(raw)
        rec["judge_raw"]        = raw
        rec["is_correct"]       = bool(parsed.get("is_correct", False))
        rec["evaluation_reason"] = parsed.get("reason", "")

    else:  # perception / halu
        if qtype == "multi_choice":
            prompt = PERCEP_MC_PROMPT.format(question=question, ground_truth=gt, model_answer=model_ans)
            raw = query_judge(SYS_PERCEP, prompt)
            parsed = parse_json_response(raw)
            rec["judge_raw"]              = raw
            rec["is_correct"]             = bool(parsed.get("is_correct", False))
            rec["model_answer_extracted"] = parsed.get("model_answer_extracted", "")
            rec["evaluation_reason"]      = parsed.get("reason", "")
            rec["is_positive"]            = rec["is_correct"]
        else:
            prompt = PERCEP_FF_PROMPT.format(question=question, ground_truth=gt, model_answer=model_ans)
            raw = query_judge(SYS_PERCEP, prompt)
            parsed = parse_json_response(raw)
            score = int(parsed.get("hallucination_score", 0))
            rec["judge_raw"]          = raw
            rec["hallucination_score"] = score
            rec["evaluation_reason"]  = parsed.get("reason", "")
            rec["is_positive"]        = score >= 3
            rec["is_correct"]         = rec["is_positive"]

    elapsed = time.time() - t0
    verdict = "✓" if rec["is_correct"] else "✗"
    print(f"  [{i+1}/{len(results)}] {verdict} {subset}/{qtype} | {elapsed:.1f}s | {rec['evaluation_reason'][:80]}")

    judged += 1
    if judged % 50 == 0:
        json.dump(results, open(OUTPUT, "w"), indent=2)

json.dump(results, open(OUTPUT, "w"), indent=2)
print(f"\nDone. Saved {judged} judged results → {OUTPUT}")

# Print sample judge outputs for quality check
print("\n=== SAMPLE JUDGE OUTPUTS (first 3) ===")
for rec in results[:3]:
    print(f"\n--- [{rec['subset']}] id={rec['id']} qtype={rec['question_type']} ---")
    print(f"Model answer (clean): {rec['clean_response'][:150]}")
    print(f"GT: {rec['gt_answer']}")
    print(f"Judge verdict: {'CORRECT' if rec['is_correct'] else 'WRONG'}")
    print(f"Judge reason: {rec.get('evaluation_reason', '')[:200]}")
    if rec.get("hallucination_score") is not None:
        print(f"Hallucination score: {rec['hallucination_score']}/5")
