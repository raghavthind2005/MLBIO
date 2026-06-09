"""
Phase 2: Judge evaluation using official RH-Bench prompts.

Paper: "More Thinking, Less Seeing?" arXiv:2505.21523
Official judge: GPT-4o (Azure). Substitute used: Qwen3-32B via sglang port 30001.
NOTE: Qwen3.6-27B was unavailable (symlink to restricted path). Qwen3-32B used instead.
NOTE: Results with Qwen3-32B are NOT directly comparable to published GPT-4o scores.

Prompts are verbatim from the official eval scripts:
  evaluation_rhbench_reason.py
  evaluation_rhbench_perception.py
  (github.com/MLRM-Halu/MLRM-Halu)
"""
import json, re, time, requests

INPUT  = "/iopsstor/scratch/cscs/raghavthind/rh-bench/results/vlm_responses.json"
OUTPUT = "/iopsstor/scratch/cscs/raghavthind/rh-bench/results/judged_responses.json"
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

PERCEP_FF_PROMPT = """Please act as an impartial and objective judge to evaluate the presence and severity of hallucination in the response provided by a Large Multimodal Model (LMM) to the user question. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation.

Evaluation Guidelines:
- Compare the LMM response with the standard human-generated answer.
- Identify any claims in the LMM response that are not supported by the image or question.
- The standard human-generated answer may not describe all objects or details comprehensively, so focus on whether the LMM introduces false claims rather than omitting details.
- Detailed reasoning or analysis by the LMM that is consistent with the image contents is acceptable and should not be considered hallucination.

Assign a rating from the following six levels:
Rating 5: No hallucination (response is fully consistent with image contents).
Rating 4: Minimal hallucination (minor, negligible false detail that does not significantly alter accuracy).
Rating 3: Mild hallucination (small false claim that adds unsupported but non-critical detail).
Rating 2: Moderate hallucination (clear false claim that affects the response's accuracy).
Rating 1: Severe hallucination (significant false claim that misrepresents key elements).
Rating 0: Extreme hallucination (response is largely or entirely detached from the image contents).

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
            {"role": "user", "content": user_prompt}
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


# Wait for judge server
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

print(f"Judging all {len(results)} entries...")
judged = 0

for i, rec in enumerate(results):
    qtype = rec["question_type"]
    subset = rec["subset"]
    model_ans = rec["clean_response"]
    gt = rec["gt_answer"]
    question = rec["question"]

    if subset == "reason":
        if qtype == "multi_choice":
            prompt = REASON_MC_PROMPT.format(question=question, model_answer=model_ans, ground_truth=gt)
        else:
            prompt = REASON_FF_PROMPT.format(question=question, model_answer=model_ans, ground_truth=gt)
        raw = query_judge(SYS_REASON, prompt)
        parsed = parse_json_response(raw)
        rec["judge_raw"] = raw
        rec["is_correct"] = bool(parsed.get("is_correct", False))
        rec["evaluation_reason"] = parsed.get("reason", "")

    else:  # halu / perception
        if qtype == "multi_choice":
            prompt = PERCEP_MC_PROMPT.format(question=question, ground_truth=gt, model_answer=model_ans)
            raw = query_judge(SYS_PERCEP, prompt)
            parsed = parse_json_response(raw)
            rec["judge_raw"] = raw
            rec["is_correct"] = bool(parsed.get("is_correct", False))
            rec["model_answer_extracted"] = parsed.get("model_answer_extracted", "")
            rec["evaluation_reason"] = parsed.get("reason", "")
            rec["is_positive"] = rec["is_correct"]
        else:
            prompt = PERCEP_FF_PROMPT.format(question=question, ground_truth=gt, model_answer=model_ans)
            raw = query_judge(SYS_PERCEP, prompt)
            parsed = parse_json_response(raw)
            rec["judge_raw"] = raw
            score = int(parsed.get("hallucination_score", 0))
            rec["hallucination_score"] = score
            rec["evaluation_reason"] = parsed.get("reason", "")
            rec["is_positive"] = score >= 3
            rec["is_correct"] = rec["is_positive"]

    judged += 1
    if judged % 50 == 0:
        print(f"  Judged {judged}/{len(results)}")
        json.dump(results, open(OUTPUT, "w"), indent=2)

json.dump(results, open(OUTPUT, "w"), indent=2)
print(f"\nDone. Saved → {OUTPUT}")
