"""
docci_data.py — load the DOCCI perception probe set (the TRAINING distribution).

This is the in-distribution probe: the exact data the model was RL-trained on
(perception_difficulty_curriculum.jsonl), in its native direct-answer MC format
("Respond using only the letter..."). The RL gain here is large and known
(accuracy 0.365 -> 0.746), so it's where we can actually localize the fix.

Each line:
  {"index","problem":"<image>...Options:\nA: ..\nB: ..\n<instruction>","answer":"B",
   "images":["DOCCI/images_downsampled_2x/xxx.jpg"], "pass_rate":..., ...}

Produces the SAME MCItem type as babyvision_data so mc_eval/module_graft/depth_probe
work unchanged.

CONTAMINATION NOTE: the model trained on these items (16 epochs). For LOCALIZATION
analyses (S2/S3/S4/S5 — "which weights/layers carry the perception competence") this is
fine: we're dissecting the learned mechanism, not claiming generalization. Base-model
numbers are uncontaminated. Trained-model numbers are train-accuracy (state it).

Usage (sanity needs the cluster paths):
  python docci_data.py --jsonl <…/perception_difficulty_curriculum.jsonl> \
                       --image-dir <…/images> --n 300
"""

import argparse
import json
import os
import random
import re

from babyvision_data import MCItem  # reuse the dataclass


_OPT_RE = re.compile(r"(?m)^\s*([A-F]):\s")


def load_docci_items(jsonl_path: str, image_dir: str,
                     n_sample: int = None, seed: int = 1) -> list:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if n_sample is not None and n_sample < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, n_sample)

    items = []
    for x in rows:
        problem = x["problem"].replace("<image>", "").strip()
        letters = sorted(set(_OPT_RE.findall(problem)))
        if not letters:
            continue
        gold = str(x["answer"]).strip()
        if gold not in letters:
            continue
        img = os.path.join(image_dir, x["images"][0])
        if not os.path.isfile(img):
            continue
        items.append(MCItem(
            task_id=x.get("index", ""),
            image_path=img,
            question=problem,                 # native trained prompt (instruction included)
            options=letters,
            n_options=len(letters),
            gold_index=ord(gold) - 65,
            gold_letter=gold,
            type=x.get("source", "DOCCI"),
            subtype=str(x.get("pass_rate", "")),   # stash pass_rate as subtype for difficulty analysis
        ))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    items = load_docci_items(args.jsonl, args.image_dir, n_sample=args.n, seed=args.seed)
    print(f"Loaded {len(items)} DOCCI MC items.")
    from collections import Counter
    print("option counts:", dict(sorted(Counter(i.n_options for i in items).items())))
    print("gold-letter balance:", dict(sorted(Counter(i.gold_letter for i in items).items())))
    s = items[0]
    print("\n=== sample ===")
    print(f"index: {s.task_id}  gold: {s.gold_letter}  n_opts: {s.n_options}")
    print(f"image: {s.image_path}  exists: {os.path.isfile(s.image_path)}")
    print(f"--- prompt ---\n{s.question}")


if __name__ == "__main__":
    main()
