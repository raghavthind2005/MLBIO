"""
depth_probe.py — S4/S5 logit-lens: is the answer decodable mid-stack, lost up top, re-surfaced by RL?

The MC probe (mc_eval) reads only the FINAL layer. Here we read EVERY layer via the logit lens:
at each decoder layer L, take the hidden state at the answer position, push it through the model's
OWN final norm + output head, restrict to the option-letter tokens, and measure decodability.

  decodability(L) = mean P(correct letter)   and   argmax-accuracy over present letters

Curve over L:
  S4 (base model)   : expect mid-layer peak, decay toward the top (perception lost going up the stack)
  S5 (trained model): expect the top-layer decay to FLATTEN (RL re-surfaces mid-layer perception)

No probe training needed — uses the model's own unembedding (standard logit-lens). Run for base and
for each checkpoint; overlay the curves.

Requires container + GPU. Run AFTER mc_eval validates the probe path (DOCCI).

Usage:
  python depth_probe.py --base <model> --dataset docci --jsonl <…> --image-dir <…> --n-sample 300 \
                        --out depth_base.csv
  python depth_probe.py --base <model> --ckpt <global_step_96/actor> --dataset docci ... --out depth_c1_96.csv
"""

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from babyvision_data import option_letters
from probe_loader import add_probe_args, load_probe
from ckpt_model import load_model
from PIL import Image


def locate_lens(model):
    """Return (final_norm, lm_head) for the logit lens. Defensive about attribute paths."""
    lm_head = model.get_output_embeddings()
    assert lm_head is not None, "could not find output embeddings (lm_head)"
    norm = None
    for path in ("model.language_model.norm", "model.model.norm", "language_model.norm", "model.norm"):
        obj = model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok:
            norm = obj
            print(f"[depth_probe] final norm found at: {path}")
            break
    assert norm is not None, "could not locate final norm module — print model to find it"
    return norm, lm_head


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="depth.csv")
    add_probe_args(ap)
    args = ap.parse_args()

    items = load_probe(args)
    print(f"[depth_probe] {len(items)} items ({args.dataset})")
    model, processor = load_model(args.base, ckpt_actor_dir=args.ckpt, device=args.device)
    norm, lm_head = locate_lens(model)
    tok = processor.tokenizer
    letter_ids = {chr(65 + i): tok.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(6)}

    n_layers = None
    # accumulators per layer
    sum_correct_prob = None
    sum_argmax_correct = None

    for k, it in enumerate(items):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": it.question}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = Image.open(it.image_path).convert("RGB")
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(args.device)

        out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states  # tuple len (n_layers+1): [0]=embeddings, [i]=after layer i

        if n_layers is None:
            n_layers = len(hs)  # includes embedding layer at index 0
            sum_correct_prob = [0.0] * n_layers
            sum_argmax_correct = [0.0] * n_layers
            print(f"[depth_probe] {n_layers} hidden-state slots (0=embed, last=final layer)")

        present = option_letters(it.n_options)
        cand_ids = [letter_ids[L] for L in present]
        gold_pos = present.index(it.gold_letter)

        for L in range(n_layers):
            h = hs[L][0, -1, :]                      # answer-position hidden state at layer L
            logits = lm_head(norm(h))                # logit lens: model's own readout
            cand = torch.tensor([logits[c].item() for c in cand_ids])
            probs = torch.softmax(cand, dim=-1)
            sum_correct_prob[L] += probs[gold_pos].item()
            sum_argmax_correct[L] += 1.0 if int(probs.argmax()) == gold_pos else 0.0

        if (k + 1) % 50 == 0:
            print(f"  … {k+1}/{len(items)}")

    n = len(items)
    rows = []
    for L in range(n_layers):
        rows.append(dict(layer=L,
                         mean_correct_prob=sum_correct_prob[L] / n,
                         argmax_acc=sum_argmax_correct[L] / n))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "mean_correct_prob", "argmax_acc"])
        w.writeheader()
        w.writerows(rows)
    print(f"[depth_probe] wrote {args.out}")

    # quick text view of the curve
    print("\nlayer :  P(correct)  argmax_acc")
    for r in rows:
        bar = "#" * int(r["argmax_acc"] * 40)
        print(f"  {r['layer']:2d}  :  {r['mean_correct_prob']:.3f}      {r['argmax_acc']:.3f}  {bar}")


if __name__ == "__main__":
    main()
