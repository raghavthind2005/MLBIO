#!/usr/bin/env python3
"""
E2 Phase A — build + VALIDATE the scene-state probe.  GO/NO-GO GATE.

Question: is the CLEVR scene state *linearly decodable* from the model's TEXT/reasoning-position
hidden states? (If it's only decodable at the frozen image tokens, drift can't be tracked and E2
pivots — E2_DESIGN.md §12.)

Method (transformers, teacher-forced on the saved chains):
  - For the 179 CORRECT items: reconstruct prompt+image, append saved output_token_ids, one forward
    pass with output_hidden_states, grab hidden vectors at {prompt_end + reasoning-position fracs} x
    {layer grid}.
  - Labels = the 15 CLEVR attribute-value MARGINALS (presence of each color/shape/size/material),
    from the scene graph.
  - Train linear presence-probes; validate HELD-OUT BY SCENE + a shuffled-label SELECTIVITY control.
  - Report accuracy(layer x position) + selectivity + a GO/NO-GO verdict.

Guarded under __main__. Modes: smoke (20 items, few layers) | full.
"""
import os, sys, io, json, time, re
import numpy as np

MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
RECIN  = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out/v2_full_records.jsonl"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED  = "\n\nPut your final answer in \\boxed{}."

LAYERS   = [12, 18, 24, 30, 36] if MODE != "smoke" else [18, 24, 30]
POS_FRAC = [0.25, 0.5, 0.75, 0.95]          # fraction through the <think> region; plus "prompt_end"
N_CORR   = 179 if MODE != "smoke" else 20

COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
SHAPES = ["cube", "sphere", "cylinder"]
SIZES  = ["large", "small"]
MATS   = ["rubber", "metal"]
ATTRS  = [("color", c) for c in COLORS] + [("shape", s) for s in SHAPES] + \
         [("size", z) for z in SIZES] + [("material", m) for m in MATS]     # 15 marginals

def log(*a): print(*a, flush=True)
def presence_label(scene):
    objs = scene["objects"]
    return np.array([int(any(o[a] == v for o in objs)) for a, v in ATTRS], dtype=np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    log("="*70, f"\nE2 PHASE A  MODE={MODE}  layers={LAYERS}  pos={['prompt_end']+POS_FRAC}")
    import torch, transformers
    import torch.nn.functional as F
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    log(f"torch {torch.__version__}  transformers {transformers.__version__}  gpu {torch.cuda.get_device_name(0)}")

    rows = [json.loads(l) for l in open(RECIN)]
    # correct items only (A=1) — re-derive correctness cheaply from saved parsed flag is unreliable (synonyms);
    # use the saved 'parsed.correct' OR fall back. We trust v2 'parsed' here only to *select a pool*; labels are GT.
    corr = [r for r in rows if r["parsed"]["correct"]][:N_CORR]
    log(f"correct-pool items for probe: {len(corr)}")

    proc = AutoProcessor.from_pretrained(MODEL)
    tok = proc.tokenizer
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    log(f"model loaded in {time.time()-t0:.0f}s")
    close_ids = tok("</think>", add_special_tokens=False).input_ids
    log(f"</think> token ids = {close_ids}")

    def find_close(resp_ids):
        n = len(close_ids)
        for i in range(len(resp_ids) - n + 1):
            if resp_ids[i:i+n] == close_ids: return i
        return len(resp_ids)

    # ---- extract hidden vectors ----
    POSN = ["prompt_end"] + POS_FRAC
    X = {(L, p): [] for L in LAYERS for p in POSN}       # feature bank
    Y, SCENE = [], []
    t0 = time.time()
    for k, r in enumerate(corr):
        img = Image.open(f"{IMGDIR}/{r['image_filename']}").convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["question"] + BOXED}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = proc(text=[text], images=[img], return_tensors="pt")
        prompt_ids = enc["input_ids"]                     # [1,P]
        P = prompt_ids.shape[1]
        resp = list(r["output_token_ids"])
        full = torch.cat([prompt_ids, torch.tensor([resp])], dim=1).to("cuda")
        extra = {kk: enc[kk].to("cuda") for kk in ("pixel_values", "image_grid_thw") if kk in enc}
        with torch.no_grad():
            out = model(input_ids=full, attention_mask=torch.ones_like(full), output_hidden_states=True, **extra)
        hs = out.hidden_states                            # tuple len n_layers+1, each [1, seq, H]
        r_end = P + find_close(resp)                       # end of <think> region
        think_len = max(1, r_end - P)
        pos_idx = {"prompt_end": P - 1}
        for f in POS_FRAC: pos_idx[f] = min(full.shape[1]-1, P + int(f * think_len))
        for L in LAYERS:
            for p in POSN:
                X[(L, p)].append(hs[L][0, pos_idx[p]].float().cpu().numpy())
        Y.append(presence_label(r["scene"])); SCENE.append(r["image_index"])
        del out, hs, full
        if (k+1) % 20 == 0: log(f"  extracted {k+1}/{len(corr)} ({time.time()-t0:.0f}s)")
    Y = np.stack(Y); SCENE = np.array(SCENE)
    log(f"extraction done ({time.time()-t0:.0f}s); features per cell: {Y.shape[0]}")

    # ---- held-out-by-SCENE split ----
    uscenes = np.array(sorted(set(SCENE.tolist())))
    rng = np.random.RandomState(0); rng.shuffle(uscenes)
    cut = int(0.7 * len(uscenes)); train_sc = set(uscenes[:cut].tolist())
    tr = np.array([s in train_sc for s in SCENE]); te = ~tr
    log(f"scene split: {tr.sum()} train / {te.sum()} test features; {len(train_sc)}/{len(uscenes)} scenes train")

    def probe_acc(Xall, ycol, shuffle=False):
        import torch
        Xt = torch.tensor(Xall, dtype=torch.float32)
        mu, sd = Xt[tr].mean(0), Xt[tr].std(0) + 1e-6
        Xn = (Xt - mu) / sd
        y = torch.tensor(ycol, dtype=torch.float32)
        ytr = y[tr].clone()
        if shuffle:
            perm = torch.randperm(ytr.shape[0]); ytr = ytr[perm]
        if ytr.sum() == 0 or ytr.sum() == ytr.shape[0]:        # degenerate marginal in train
            return None
        w = torch.zeros(Xn.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        Xtr = Xn[tr]
        for _ in range(300):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, ytr) + 1e-3 * w.pow(2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = (Xn[te] @ w + b > 0).float()
        return (pred == y[te]).float().mean().item()

    # ---- validate every (layer, position) ----
    log("="*70, "\nPROBE VALIDATION  (held-out-by-scene presence accuracy | selectivity = real - shuffled)")
    log(f"{'layer':>6} {'pos':>10} {'acc':>6} {'ctrl':>6} {'select':>7}  (mean over the 15 marginals)")
    best = (None, -1)
    grid = {}
    for L in LAYERS:
        for p in POSN:
            Xall = np.stack(X[(L, p)])
            accs, ctrls = [], []
            for j in range(len(ATTRS)):
                a = probe_acc(Xall, Y[:, j], shuffle=False)
                c = probe_acc(Xall, Y[:, j], shuffle=True)
                if a is not None: accs.append(a)
                if c is not None: ctrls.append(c)
            acc, ctrl = float(np.mean(accs)), float(np.mean(ctrls))
            grid[(L, p)] = (acc, ctrl);
            log(f"{L:>6} {str(p):>10} {acc:>6.3f} {ctrl:>6.3f} {acc-ctrl:>7.3f}")
            if acc > best[1]: best = ((L, p), acc)

    (bL, bp), bacc = best
    bctrl = grid[(bL, bp)][1]
    log("="*70, "\nGO / NO-GO")
    log(f"best cell: layer {bL}, pos {bp}  ->  held-out acc {bacc:.3f}, selectivity {bacc-bctrl:.3f}")
    verdict = "GO — scene state IS linearly decodable from text positions" if (bacc >= 0.90 and (bacc-bctrl) >= 0.15) \
        else ("MARGINAL — reparametrize/try more layers before trusting" if bacc >= 0.80 else
              "NO-GO — scene not decodable from text positions; pivot (attention-flow)")
    log(f"VERDICT: {verdict}")
    json.dump(dict(mode=MODE, best_layer=bL, best_pos=str(bp), best_acc=bacc, best_selectivity=bacc-bctrl,
                   grid={f"{L}|{p}": grid[(L, p)] for (L, p) in grid}, verdict=verdict),
              open(f"{OUT}/phaseA_{MODE}_summary.json", "w"), indent=2, default=str)
    log(f"saved -> {OUT}/phaseA_{MODE}_summary.json")


if __name__ == "__main__":
    main()
