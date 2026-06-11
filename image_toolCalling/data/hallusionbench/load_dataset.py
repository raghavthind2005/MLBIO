"""
HallusionBench dataset loader.

Dataset: Guan et al., "HallusionBench: An Advanced Diagnostic Suite for
Entangled Language Hallucination and Visual Illusion in Large VLMs", CVPR 2024.
Source: https://huggingface.co/datasets/rayguan/HallusionBench

Schema (per sample in HallusionBench.json):
  category      : "VD" (Visual Dependent) | "VS" (Visual Supplement)
  subcategory   : chart | table | map | ocr | illusion | math | figure | video
  set_id        : groups questions that share an image
  figure_id     : image variant id within a set (0 = original, 1+ = edited)
  question_id   : unique question id
  question      : yes/no question string
  gt_answer     : "0" (No) | "1" (Yes)
  gt_answer_details : explanation of the correct answer
  sample_note   : short tag describing the visual test type
  visual_input  : "0" = text-only | "1" = original image | "2" = edited image
  filename      : relative path to image under data/, or None if visual_input=0

Evaluation metrics (all LLM-free):
  qAcc  - per-question binary accuracy
  fAcc  - figure-level accuracy (all questions on one image correct)
  consistency - pair-level: original and edited image answered correctly
"""

import json
import os
from pathlib import Path
from typing import Optional


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
JSON_PATH = BENCHMARK_DIR / "HallusionBench.json"


def load(
    visual_input: Optional[str] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
) -> list[dict]:
    """
    Load HallusionBench samples, optionally filtered.

    Args:
        visual_input: filter by image type — "0" (text-only), "1" (original),
                      "2" (edited), or None for all.
        category:     "VD" | "VS" | None for all.
        subcategory:  e.g. "chart", "illusion" | None for all.

    Returns:
        List of sample dicts. Each dict has an added "image_path" key:
        an absolute Path to the image file, or None for text-only samples.
    """
    with open(JSON_PATH) as f:
        data = json.load(f)

    if visual_input is not None:
        data = [d for d in data if d["visual_input"] == visual_input]
    if category is not None:
        data = [d for d in data if d["category"] == category]
    if subcategory is not None:
        data = [d for d in data if d["subcategory"] == subcategory]

    for sample in data:
        if sample.get("filename"):
            sample["image_path"] = DATA_DIR / sample["filename"].lstrip("./")
        else:
            sample["image_path"] = None

    return data


def score(predictions: list[dict]) -> dict:
    """
    Compute qAcc and fAcc from model predictions.

    Args:
        predictions: list of dicts, each must have:
            - all original sample fields (set_id, figure_id, gt_answer, visual_input)
            - "model_prediction": "0" or "1" (normalized yes/no)

    Returns:
        dict with qAcc, fAcc, and per-category breakdowns.
    """
    from collections import defaultdict

    correct_q = 0
    total_q = len(predictions)

    # figure-level: key = (category, set_id, figure_id)
    figure_results = defaultdict(list)

    for p in predictions:
        pred = str(p["model_prediction"])
        gt = str(p["gt_answer"])
        is_correct = int(pred == gt)
        correct_q += is_correct

        fig_key = (p["category"], p["set_id"], p["figure_id"])
        figure_results[fig_key].append(is_correct)

    qacc = correct_q / total_q if total_q else 0.0
    facc = (
        sum(1 for v in figure_results.values() if all(v)) / len(figure_results)
        if figure_results else 0.0
    )

    # per-category breakdown
    from collections import Counter
    cat_correct = Counter()
    cat_total = Counter()
    subcat_correct = Counter()
    subcat_total = Counter()
    for p in predictions:
        pred = str(p["model_prediction"])
        gt = str(p["gt_answer"])
        is_correct = int(pred == gt)
        cat_correct[p["category"]] += is_correct
        cat_total[p["category"]] += 1
        subcat_correct[p["subcategory"]] += is_correct
        subcat_total[p["subcategory"]] += 1

    return {
        "qAcc": round(qacc, 4),
        "fAcc": round(facc, 4),
        "n_questions": total_q,
        "n_figures": len(figure_results),
        "by_category": {
            k: round(cat_correct[k] / cat_total[k], 4)
            for k in cat_total
        },
        "by_subcategory": {
            k: round(subcat_correct[k] / subcat_total[k], 4)
            for k in subcat_total
        },
    }


if __name__ == "__main__":
    samples = load()
    print(f"Total samples : {len(samples)}")

    from collections import Counter
    print(f"  text-only    : {sum(1 for s in samples if s['visual_input'] == '0')}")
    print(f"  original img : {sum(1 for s in samples if s['visual_input'] == '1')}")
    print(f"  edited img   : {sum(1 for s in samples if s['visual_input'] == '2')}")
    print(f"  VD           : {sum(1 for s in samples if s['category'] == 'VD')}")
    print(f"  VS           : {sum(1 for s in samples if s['category'] == 'VS')}")
    print()
    subcats = Counter(s["subcategory"] for s in samples)
    print("By subcategory:")
    for k, v in sorted(subcats.items()): print(f"  {k:<12}: {v}")
    print()

    # Sanity-check image paths
    missing = [s for s in samples if s["image_path"] and not s["image_path"].exists()]
    print(f"Missing images: {len(missing)}")
