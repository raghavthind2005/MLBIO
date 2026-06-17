#!/usr/bin/env python3
"""
BabyVision MLLM Evaluation Script

Evaluates multimodal large language models on the BabyVision benchmark
using an LLM judge to assess answer correctness.

Usage:
    python evaluate_model.py --model-api-key $KEY --model-base-url $URL ...
"""

import os
import json
import regex
import argparse
from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Pool

from utils import LLM_JUDGE_PROMPT, format_choices, image_to_base64, extract_boxed_answer

# Global clients (initialized in main)
client = None
client_judge = None
MODEL_NAME = None
JUDGE_MODEL_NAME = None


def process(data):
    """Process a single evaluation task."""
    NUM_RETRIES = 3

    for _ in range(NUM_RETRIES):
        try:
            question = data['question'] + "\nThink about the question and give your final answer in \\boxed{Answer} format."
            image_base64 = data['image']
            answer = data['answer']

            # Call the model
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_base64}},
                        {"type": "text", "text": question},
                    ]
                }],
                extra_body={"reasoning": {"enabled": True}}
            )

            model_output_result = completion.choices[0].message.content
            extracted_answer = extract_boxed_answer(model_output_result)

            # Call the judge model
            judge_model_completion = client_judge.chat.completions.create(
                model=JUDGE_MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": LLM_JUDGE_PROMPT.format(
                        question=question,
                        groundtruth=answer,
                        modeloutput=extracted_answer
                    )
                }]
            )

            judge_response_clean = str(judge_model_completion.choices[0].message.content).strip().lower()

            if 'true' in judge_response_clean:
                llm_judge_result = True
            elif 'false' in judge_response_clean:
                llm_judge_result = False
            else:
                llm_judge_result = False
                print(f"Warning: Unable to parse judge response: {judge_response_clean}")

            return {
                "Id": data['Id'],
                "Question": question,
                "ModelReasoning": getattr(completion.choices[0].message, 'reasoning_details', ""),
                "ModelResult": model_output_result,
                "GroundTruth": answer,
                "ExtractedAnswer": extracted_answer,
                "LLMJudgeResult": llm_judge_result,
                "Type": data['Type'],
                "Subtype": data['Subtype'],
            }
        except Exception as e:
            print(f"Error during processing ID {data['Id']}: {e}. Retrying...")

    # Return error result after all retries failed
    return {
        "Id": data['Id'],
        "Question": data.get('question', ''),
        "ModelReasoning": "",
        "ModelResult": "",
        "GroundTruth": data.get('answer', ''),
        "ExtractedAnswer": "",
        "LLMJudgeResult": False,
        "Error": str(e),
        "Type": data['Type'],
        "Subtype": data['Subtype'],
    }


def main():
    global client, client_judge, MODEL_NAME, JUDGE_MODEL_NAME

    parser = argparse.ArgumentParser(
        description="Evaluate MLLM on BabyVision benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate_model.py \\
        --model-api-key $MODEL_API_KEY \\
        --model-base-url https://openrouter.ai/api/v1 \\
        --model-name google/gemini-2.5-flash \\
        --judge-api-key $JUDGE_API_KEY \\
        --judge-base-url https://openrouter.ai/api/v1 \\
        --judge-model-name openai/gpt-4o \\
        --test-json-path ../data/babyvision_data/meta_data.jsonl \\
        --output-dir ./results
"""
    )

    parser.add_argument("--model-api-key", type=str, required=True, help="Model API key")
    parser.add_argument("--model-base-url", type=str, required=True, help="Model API base URL (OpenAI format)")
    parser.add_argument("--model-name", type=str, required=True, help="Model name to evaluate")
    parser.add_argument("--judge-api-key", type=str, required=True, help="Judge API key (Qwen-Max or GPT-5.2)")
    parser.add_argument("--judge-base-url", type=str, required=True, help="Judge model API base URL")
    parser.add_argument("--judge-model-name", type=str, required=True, help="Judge model name")
    parser.add_argument("--test-json-path", type=str, required=True, help="Path to test JSONL file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--num-processes", type=int, default=8, help="Number of parallel processes (default: 8)")
    parser.add_argument("--num-passes", type=int, default=3, help="Number of evaluation passes (default: 3)")

    args = parser.parse_args()

    # Set global model names
    MODEL_NAME = args.model_name
    JUDGE_MODEL_NAME = args.judge_model_name

    # Initialize API clients
    client = OpenAI(
        api_key=args.model_api_key,
        base_url=args.model_base_url,
        timeout=1800,
    )

    client_judge = OpenAI(
        api_key=args.judge_api_key,
        base_url=args.judge_base_url,
        timeout=1800,
    )

    # Load test data
    meta_data = []
    with open(args.test_json_path, "r") as f:
        for line in f:
            if line.strip():
                meta_data.append(json.loads(line.strip()))

    print(f"Loaded {len(meta_data)} tasks from {args.test_json_path}")

    # Prepare data for processing
    data = []
    test_json_dir = os.path.dirname(args.test_json_path)

    for item in meta_data:
        image_path = os.path.join(test_json_dir, item['image'])
        image_base64 = image_to_base64(image_path)

        if item['ansType'] == "blank":
            question = item['question']
            answer = item['blankAns']
        else:
            question = item['question'] + "\nChoices:\n" + format_choices(item['options'])
            answer = chr(65 + int(item['choiceAns']))

        data.append({
            "Id": item['taskId'],
            "question": question,
            "image": image_base64,
            "answer": answer,
            "Type": item['type'],
            "Subtype": item['subtype'],
        })

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Judge: {JUDGE_MODEL_NAME}")
    print(f"  Tasks: {len(data)}")
    print(f"  Processes: {args.num_processes}")
    print(f"  Passes: {args.num_passes}")
    print()

    # Run evaluation passes
    for pass_num in range(args.num_passes):
        print(f"Starting pass {pass_num + 1}/{args.num_passes}...")

        with Pool(processes=args.num_processes) as pool:
            results = list(tqdm(pool.imap(process, data), total=len(data), desc=f"Pass {pass_num + 1}"))

        output_path = os.path.join(args.output_dir, f"model_results_run_{pass_num + 1}.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        correct_count = sum(1 for r in results if r['LLMJudgeResult'])
        print(f"\n{'=' * 50}")
        print(f"Pass {pass_num + 1} Complete")
        print(f"{'=' * 50}")
        print(f"Total: {len(results)}")
        print(f"Correct: {correct_count}")
        print(f"Accuracy: {correct_count / len(results) * 100:.2f}%")
        print(f"Results saved to: {output_path}")
        print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
