"""
activation_patch.py — Experiment 2: are the better representations RE-USABLE?

Test whether the perception fix is a PORTABLE representation, not only a weight change:
take the trained model's late-layer residual (answer position) and inject it into the BASE
model at inference; measure how much accuracy recovers (base 0.377 -> trained 0.657).

Two variants (same hooks):
  - per-item PATCH : overwrite base's residual at layer L (answer pos) with the trained
                     model's residual for that item; base finishes the forward.  (causal upper bound)
  - STEER vector   : v_L = mean_items(trained_resid_L - base_resid_L); add alpha*v_L to base's
                     residual at layer L.  (portable, deploy-time artifact)

Layer indexing: layer i = OUTPUT of decoder layer i (0-indexed, 0..35). Patching layer i sets
the residual that ENTERS layer i+1. (Depth-probe's hs[L] == output of decoder layer L-1; so a
divergence at depth-probe layer ~25 ≈ decoder layer ~24 here.)

Requires container + GPU. True causal test (real residuals + real output head; no logit-lens).

Usage:
  python activation_patch.py --base <model> --ckpt <full/step96/actor> \
      --dataset docci --jsonl <…> --image-dir <…> --n-sample 300 \
      --layers 12 16 20 24 28 32 35 --alphas 1 2 4 --out actpatch_c1.csv
"""

import argparse
import csv
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from probe_loader import add_probe_args, load_probe
from babyvision_data import option_letters
from ckpt_model import load_model
from mc_eval import build_letter_token_ids


# ── locate the decoder-layer ModuleList ──────────────────────────────────────
def get_layers(model):
    for path in ("model.language_model.layers", "model.model.layers", "language_model.layers"):
        obj = model
        ok = True
        for a in path.split("."):
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False
                break
        if ok:
            print(f"[actpatch] decoder layers at: {path} ({len(obj)} layers)")
            return obj
    raise RuntimeError("could not find decoder layers — print(model) to locate")


# ── hooks (handle layer output as tuple OR tensor) ───────────────────────────
def capture_hook(store, idx):
    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        store[idx] = hs[:, -1, :].detach().float().cpu().squeeze(0)  # (hidden,)
        return None
    return hook


def patch_hook(vec):  # vec on device, (hidden,)
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out[0][:, -1, :] = vec.to(out[0].dtype)
            return out
        out[:, -1, :] = vec.to(out.dtype)
        return out
    return hook


def steer_hook(vec, alpha):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out[0][:, -1, :] = out[0][:, -1, :] + alpha * vec.to(out[0].dtype)
            return out
        out[:, -1, :] = out[:, -1, :] + alpha * vec.to(out.dtype)
        return out
    return hook


# ── probe readout ────────────────────────────────────────────────────────────
def predict_correct(logits_last, item, letter_ids):
    present = option_letters(item.n_options)
    cand = torch.tensor([logits_last[letter_ids[L]].item() for L in present])
    return present[int(cand.argmax())] == item.gold_letter


@torch.no_grad()
def run_item(model, inputs_cpu, device):
    inp = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs_cpu.items()}
    return model(**inp).logits[0, -1, :].float()


def preprocess(items, processor):
    out = []
    for it in items:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": it.question}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = Image.open(it.image_path).convert("RGB")
        inp = processor(text=[text], images=[image], return_tensors="pt")
        out.append(dict(inp))  # keep on CPU
    return out


@torch.no_grad()
def capture_all(model, layers, inputs_list, items, letter_ids, device):
    """Run once; cache answer-pos residual at every layer + accuracy."""
    cache = [dict() for _ in items]
    n_correct = 0
    for idx, (inp, it) in enumerate(zip(inputs_list, items)):
        store = {}
        handles = [layers[L].register_forward_hook(capture_hook(store, L)) for L in range(len(layers))]
        logits = run_item(model, inp, device)
        for h in handles:
            h.remove()
        cache[idx] = store
        n_correct += int(predict_correct(logits, it, letter_ids))
        if (idx + 1) % 100 == 0:
            print(f"    captured {idx+1}/{len(items)}")
    return cache, n_correct / len(items)


@torch.no_grad()
def patch_eval(model, layers, inputs_list, items, cache, L, letter_ids, device):
    n_correct = 0
    for idx, (inp, it) in enumerate(zip(inputs_list, items)):
        vec = cache[idx][L].to(device)
        h = layers[L].register_forward_hook(patch_hook(vec))
        logits = run_item(model, inp, device)
        h.remove()
        n_correct += int(predict_correct(logits, it, letter_ids))
    return n_correct / len(items)


@torch.no_grad()
def steer_eval(model, layers, inputs_list, items, vL, alpha, L, letter_ids, device):
    n_correct = 0
    vL = vL.to(device)
    for idx, (inp, it) in enumerate(zip(inputs_list, items)):
        h = layers[L].register_forward_hook(steer_hook(vL, alpha))
        logits = run_item(model, inp, device)
        h.remove()
        n_correct += int(predict_correct(logits, it, letter_ids))
    return n_correct / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True, help="trained checkpoint (full/step96/actor)")
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 16, 20, 24, 28, 32, 35])
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="actpatch.csv")
    add_probe_args(ap)
    args = ap.parse_args()

    items = load_probe(args)
    print(f"[actpatch] {len(items)} items ({args.dataset}); patch layers={args.layers}")
    rows = []

    # ── Phase 1: trained — cache residuals + trained accuracy ─────────────────
    print("\n[actpatch] PHASE 1: trained model (capture)")
    model_t, processor = load_model(args.base, ckpt_actor_dir=args.ckpt, device=args.device)
    layers_t = get_layers(model_t)
    letter_ids = build_letter_token_ids(processor.tokenizer)
    inputs_list = preprocess(items, processor)
    trained_cache, acc_trained = capture_all(model_t, layers_t, inputs_list, items, letter_ids, args.device)
    print(f"[actpatch] trained accuracy = {acc_trained:.4f}")
    del model_t
    torch.cuda.empty_cache()

    # ── Phase 2: base — cache residuals + base accuracy; build steering vectors ─
    print("\n[actpatch] PHASE 2: base model (capture + baseline)")
    model_b, _ = load_model(args.base, device=args.device)
    layers_b = get_layers(model_b)
    base_cache, acc_base = capture_all(model_b, layers_b, inputs_list, items, letter_ids, args.device)
    print(f"[actpatch] base accuracy = {acc_base:.4f}")
    n_layers = len(layers_b)
    steer_vecs = {}
    for L in range(n_layers):
        diffs = torch.stack([trained_cache[i][L] - base_cache[i][L] for i in range(len(items))])
        steer_vecs[L] = diffs.mean(0)  # (hidden,)

    rows.append(dict(mode="base", layer=-1, alpha=0, accuracy=acc_base))
    rows.append(dict(mode="trained", layer=-1, alpha=0, accuracy=acc_trained))

    # ── Sanity: self-patch (base residual into base) must reproduce base acc ───
    Ls = args.layers[len(args.layers) // 2]
    acc_self = patch_eval(model_b, layers_b, inputs_list, items, base_cache, Ls, letter_ids, args.device)
    print(f"[actpatch] SANITY self-patch@L{Ls}: {acc_self:.4f} (must ≈ base {acc_base:.4f})")
    rows.append(dict(mode="sanity_self_patch", layer=Ls, alpha=0, accuracy=acc_self))

    # ── Phase 3: per-item PATCH sweep ─────────────────────────────────────────
    print("\n[actpatch] PHASE 3: per-item patch sweep")
    for L in args.layers:
        acc = patch_eval(model_b, layers_b, inputs_list, items, trained_cache, L, letter_ids, args.device)
        rec = (acc - acc_base) / (acc_trained - acc_base) if acc_trained != acc_base else float("nan")
        print(f"  patch@L{L:2d}: acc={acc:.4f}  recovers {rec*100:5.1f}%")
        rows.append(dict(mode="patch", layer=L, alpha=0, accuracy=acc))

    # ── Phase 4: STEER sweep ──────────────────────────────────────────────────
    print("\n[actpatch] PHASE 4: steering-vector sweep")
    for L in args.layers:
        for a in args.alphas:
            acc = steer_eval(model_b, layers_b, inputs_list, items, steer_vecs[L], a, L, letter_ids, args.device)
            rec = (acc - acc_base) / (acc_trained - acc_base) if acc_trained != acc_base else float("nan")
            print(f"  steer@L{L:2d} a={a}: acc={acc:.4f}  recovers {rec*100:5.1f}%")
            rows.append(dict(mode="steer", layer=L, alpha=a, accuracy=acc))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "layer", "alpha", "accuracy"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[actpatch] wrote {args.out}")
    print(f"[actpatch] base={acc_base:.4f}  trained={acc_trained:.4f}  "
          f"best_patch={max((r['accuracy'] for r in rows if r['mode']=='patch'), default=0):.4f}  "
          f"best_steer={max((r['accuracy'] for r in rows if r['mode']=='steer'), default=0):.4f}")


if __name__ == "__main__":
    main()
