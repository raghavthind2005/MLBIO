"""
probe_debug.py — diagnose WHY the MC probe is near-chance.

For a few MC items it prints:
  (1) the top-k UNRESTRICTED next-token predictions at the answer position
      -> does the model emit a bare letter, or a reasoning preamble?
  (2) the per-letter logits/probs (is the right letter even in contention?)
  (3) a short GREEDY generation + the \boxed{} answer it actually produces
      -> what does the model answer when allowed to respond normally?

This tells us if the failure is the PROBE DESIGN (reads noise, not the answer)
or the TASK (model genuinely can't do babyVision direct).

Usage:
  python probe_debug.py --base <model> --data-dir <babyvision_data> --n 6
  python probe_debug.py --base <model> --ckpt <global_step_96/actor> --data-dir <dir> --n 6
"""

import argparse
import os
import re
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from babyvision_data import load_mc_items, option_letters
from ckpt_model import load_model


def extract_boxed(text):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return m[-1].strip() if m else None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gen-tokens", type=int, default=512)
    args = ap.parse_args()

    items = load_mc_items(args.data_dir)[: args.n]
    model, processor = load_model(args.base, ckpt_actor_dir=args.ckpt, device=args.device)
    tok = processor.tokenizer
    letter_ids = {chr(65 + i): tok.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(6)}

    for it in items:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": it.question}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = Image.open(it.image_path).convert("RGB")
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(args.device)

        # (1)+(2): next-token distribution at the answer position
        out = model(**inputs)
        logits = out.logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        topk = torch.topk(probs, 10)
        present = option_letters(it.n_options)

        print("\n" + "=" * 72)
        print(f"taskId {it.task_id} | type {it.type} | gold = {it.gold_letter} ({it.n_options} opts)")
        print("  top-10 UNRESTRICTED next tokens:")
        for p, tid in zip(topk.values.tolist(), topk.indices.tolist()):
            print(f"     {p:6.3f}  id={tid:<6d} {repr(tok.decode([tid]))}")
        print("  per-letter probs:", {L: round(probs[letter_ids[L]].item(), 4) for L in present})

        # (3): greedy generation — what does it actually answer?
        gen = model.generate(**inputs, max_new_tokens=args.gen_tokens, do_sample=False)
        gen_text = tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        boxed = extract_boxed(gen_text)
        print(f"  GREEDY answer (\\boxed) = {boxed!r}   [gold {it.gold_letter}]")
        print(f"  generation ({len(gen_text)} chars): {gen_text[:300]!r}{' …' if len(gen_text) > 300 else ''}")


if __name__ == "__main__":
    main()
