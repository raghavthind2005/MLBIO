"""
mc_eval.py — the shared MC perception probe (one forward → option-letter readout).

For each multiple-choice babyVision item: feed image+question+lettered choices, run ONE
forward, read the logits at the answer position restricted to the present option-letter
tokens, argmax → predicted letter, compare to gold → deterministic accuracy.

This is:
  - the baseline/integration test (load base model, get its MC accuracy), and
  - the eval core reused by module_graft.py (run the same probe on grafted weights).

depth_probe.py extends this to every layer (logit-lens).

Requires the container + GPU.

Usage:
  # baseline accuracy of the base model:
  python mc_eval.py --base <model> --data-dir <babyvision_data>

  # accuracy of a trained checkpoint:
  python mc_eval.py --base <model> --ckpt <global_step_96/actor> --data-dir <babyvision_data>
"""

import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from babyvision_data import load_mc_items, option_letters
from ckpt_model import load_model


def build_letter_token_ids(tokenizer, max_letters: int = 6) -> dict:
    """Map each option letter to its first-token id. Printed in Step 0 for verification."""
    ids = {}
    for i in range(max_letters):
        L = chr(65 + i)
        enc = tokenizer.encode(L, add_special_tokens=False)
        ids[L] = enc[0] if enc else None
    return ids


@torch.no_grad()
def run_mc_probe(model, processor, items, device="cuda", verbose=True):
    """Returns (accuracy, results[list of dict]). One forward per item."""
    tokenizer = processor.tokenizer
    letter_ids = build_letter_token_ids(tokenizer)
    if verbose:
        print(f"[mc_eval] letter token ids: {letter_ids}")
        # decode-back sanity
        for L, tid in letter_ids.items():
            if tid is not None:
                print(f"    '{L}' -> id {tid} -> decodes '{tokenizer.decode([tid])}'")

    results = []
    n_correct = 0
    for k, it in enumerate(items):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": it.question},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = Image.open(it.image_path).convert("RGB")
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

        out = model(**inputs)
        last_logits = out.logits[0, -1, :]  # (vocab,)

        # restrict to the present option letters
        present = option_letters(it.n_options)
        cand_ids = [letter_ids[L] for L in present]
        cand_logits = torch.tensor([last_logits[tid].item() for tid in cand_ids])
        pred_idx = int(cand_logits.argmax().item())
        pred_letter = present[pred_idx]

        correct = (pred_letter == it.gold_letter)
        n_correct += int(correct)
        results.append(dict(
            task_id=it.task_id, type=it.type, subtype=it.subtype,
            n_options=it.n_options, gold=it.gold_letter,
            pred=pred_letter, correct=correct,
        ))
        if verbose and (k + 1) % 25 == 0:
            print(f"  … {k+1}/{len(items)}  running acc={n_correct/(k+1):.3f}")

    acc = n_correct / len(items) if items else float("nan")
    return acc, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", default=None, help="global_step_N/actor (optional)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="optional CSV of per-item results")
    args = ap.parse_args()

    items = load_mc_items(args.data_dir)
    print(f"[mc_eval] {len(items)} MC items")
    model, processor = load_model(args.base, ckpt_actor_dir=args.ckpt, device=args.device)

    acc, results = run_mc_probe(model, processor, items, device=args.device)
    print(f"\n[mc_eval] MC accuracy = {acc:.4f}  ({sum(r['correct'] for r in results)}/{len(results)})")

    # by-type breakdown
    from collections import defaultdict
    by_type = defaultdict(lambda: [0, 0])
    for r in results:
        by_type[r["type"]][0] += int(r["correct"])
        by_type[r["type"]][1] += 1
    print("by type:")
    for t, (c, n) in sorted(by_type.items()):
        print(f"  {t:32s}: {c}/{n} = {c/n:.3f}")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"wrote per-item results to {args.out}")


if __name__ == "__main__":
    main()
