#!/usr/bin/env python3
"""
E2 Phase A (v2) — build + VALIDATE the scene-state probe.  GO/NO-GO GATE.

Fixes over v1 (which gave a FALSE no-go): the v1 probe was 2560-dim on ~14 examples
(overparametrized -> overfits -> chance regardless of signal). Now:
  - EXTRACT hidden states once (GPU), SAVE to .npz  -> re-tune probes offline, no re-extract.
  - PCA-reduce (fit on train) before the linear probe -> not overparametrized.
  - IMAGE-TOKEN sanity baseline: probe the scene from the image tokens themselves (which
    provably encode it). If even THAT fails, the probe is broken (not a scene verdict);
    if it works but text positions don't, THAT is a real no-go.
  - Balanced accuracy (chance=0.5) + shuffled-label selectivity, held-out BY SCENE.

Modes (argv[1]): smoke (40 items) | full (179) | analyze (reuse saved full .npz, probe only).
"""
import os, sys, io, json, time
import numpy as np

MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
RECIN  = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out/v2_full_records.jsonl"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED  = "\n\nPut your final answer in \\boxed{}."
IMAGE_PAD_ID = 151655
NPZ    = f"{OUT}/phaseA_features_{'full' if MODE!='smoke' else 'smoke'}.npz"

LAYERS   = [6, 12, 18, 24, 30, 36]
POS_FRAC = [0.25, 0.5, 0.75, 0.95]                     # + "image_mean" + "prompt_end"
POSN     = ["image_mean", "prompt_end"] + POS_FRAC
N_ITEMS  = 40 if MODE == "smoke" else 179
PCA_K    = 32
L2       = 1.0

COLORS = ["gray","red","blue","green","brown","purple","cyan","yellow"]
SHAPES = ["cube","sphere","cylinder"]; SIZES = ["large","small"]; MATS = ["rubber","metal"]
ATTRS  = [("color",c) for c in COLORS]+[("shape",s) for s in SHAPES]+[("size",z) for z in SIZES]+[("material",m) for m in MATS]

def log(*a): print(*a, flush=True)
def presence(scene): return np.array([int(any(o[a]==v for o in scene["objects"])) for a,v in ATTRS], dtype=np.float32)

# ---------------------------------------------------------------- extraction (GPU)
def extract():
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    rows = [json.loads(l) for l in open(RECIN)]
    corr = [r for r in rows if r["parsed"]["correct"]][:N_ITEMS]
    log(f"correct items: {len(corr)}")
    proc = AutoProcessor.from_pretrained(MODEL); tok = proc.tokenizer
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    log(f"model loaded {time.time()-t0:.0f}s")
    close_ids = tok("</think>", add_special_tokens=False).input_ids
    def find_close(r):
        n=len(close_ids)
        for i in range(len(r)-n+1):
            if r[i:i+n]==close_ids: return i
        return len(r)
    feats = {(L,p): [] for L in LAYERS for p in POSN}
    Y, SC = [], []; t0=time.time()
    for k,r in enumerate(corr):
        img=Image.open(f"{IMGDIR}/{r['image_filename']}").convert("RGB")
        msgs=[{"role":"user","content":[{"type":"image"},{"type":"text","text":r["question"]+BOXED}]}]
        text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        enc=proc(text=[text],images=[img],return_tensors="pt"); P=enc["input_ids"].shape[1]
        resp=list(r["output_token_ids"])
        full=torch.cat([enc["input_ids"],torch.tensor([resp])],dim=1).to("cuda")
        extra={kk:enc[kk].to("cuda") for kk in ("pixel_values","image_grid_thw") if kk in enc}
        with torch.no_grad():
            out=model(input_ids=full,attention_mask=torch.ones_like(full),output_hidden_states=True,**extra)
        hs=out.hidden_states
        img_pos=(full[0]==IMAGE_PAD_ID).nonzero(as_tuple=True)[0]
        r_end=P+find_close(resp); tl=max(1,r_end-P)
        pidx={"prompt_end":P-1}
        for f in POS_FRAC: pidx[f]=min(full.shape[1]-1,P+int(f*tl))
        for L in LAYERS:
            for p in POSN:
                if p=="image_mean": v=hs[L][0,img_pos].float().mean(0)
                else:               v=hs[L][0,pidx[p]].float()
                feats[(L,p)].append(v.cpu().numpy())
        Y.append(presence(r["scene"])); SC.append(r["image_index"])
        del out,hs,full
        if (k+1)%20==0: log(f"  {k+1}/{len(corr)} ({time.time()-t0:.0f}s)")
    save={f"{L}|{p}":np.stack(feats[(L,p)]) for L in LAYERS for p in POSN}
    save["Y"]=np.stack(Y); save["SC"]=np.array(SC)
    np.savez_compressed(NPZ, **save); log(f"saved features -> {NPZ}")
    return save

# ---------------------------------------------------------------- probe analysis (cheap)
def bal_acc(y,pred):
    accs=[(pred[y==c]==c).mean() for c in (0,1) if (y==c).sum()>0]
    return float(np.mean(accs)) if len(accs)==2 else None      # need both classes present in test

def probe(Xtr,ytr,Xte,l2=L2):
    import torch,torch.nn.functional as F
    Xt=torch.tensor(Xtr); yt=torch.tensor(ytr); Xe=torch.tensor(Xte)
    w=torch.zeros(Xtr.shape[1],requires_grad=True); b=torch.zeros(1,requires_grad=True)
    opt=torch.optim.Adam([w,b],lr=0.05)
    for _ in range(500):
        loss=F.binary_cross_entropy_with_logits(Xt@w+b,yt)+l2*w.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return (Xe@w+b>0).float().numpy()

def analyze(data):
    Y=data["Y"]; SC=data["SC"]; n=Y.shape[0]
    us=np.array(sorted(set(SC.tolist()))); rng=np.random.RandomState(0); rng.shuffle(us)
    tr_sc=set(us[:int(0.7*len(us))].tolist())
    tr=np.array([s in tr_sc for s in SC]); te=~tr
    log(f"items={n}  scene-split {tr.sum()} train / {te.sum()} test  ({len(tr_sc)}/{len(us)} scenes)")
    log("="*70,"\nPROBE (PCA-%d + logistic, balanced held-out acc | chance=0.5 | selectivity=real-shuffled)"%PCA_K)
    log(f"{'layer':>6} {'pos':>11} {'bal_acc':>8} {'shuf':>6} {'select':>7} {'nmarg':>5}")
    grid={}
    for L in LAYERS:
        for p in POSN:
            X=data[f"{L}|{p}"].astype(np.float32)
            mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Xn=(X-mu)/sd
            # PCA fit on train
            U,S,Vt=np.linalg.svd(Xn[tr],full_matrices=False); comp=Vt[:min(PCA_K,tr.sum()-1)]
            Xp=Xn@comp.T
            accs,shufs=[],[]
            for j in range(len(ATTRS)):
                y=Y[:,j]
                if y[tr].sum() in (0,tr.sum()): continue     # degenerate in train
                a=bal_acc(y[te], probe(Xp[tr],y[tr],Xp[te]))
                yp=y[tr].copy(); np.random.RandomState(j).shuffle(yp)
                s=bal_acc(y[te], probe(Xp[tr],yp,Xp[te]))
                if a is not None: accs.append(a)
                if s is not None: shufs.append(s)
            acc=float(np.mean(accs)) if accs else float("nan")
            shuf=float(np.mean(shufs)) if shufs else float("nan")
            grid[(L,p)]=(acc,shuf,len(accs))
            log(f"{L:>6} {str(p):>11} {acc:>8.3f} {shuf:>6.3f} {acc-shuf:>7.3f} {len(accs):>5}")
    # ---- verdict ----
    img_best=max((grid[(L,"image_mean")][0] for L in LAYERS if not np.isnan(grid[(L,"image_mean")][0])), default=float("nan"))
    txt_cells={(L,p):grid[(L,p)] for L in LAYERS for p in POSN if p not in ("image_mean","prompt_end")}
    (tL,tp),tbest=max(txt_cells.items(), key=lambda kv: (kv[1][0] if not np.isnan(kv[1][0]) else -1))
    log("="*70,"\nGO / NO-GO")
    log(f"image-token baseline (best over layers): bal_acc {img_best:.3f}   <- must be HIGH or the probe itself is broken")
    log(f"best TEXT position: layer {tL} pos {tp}  bal_acc {txt_cells[(tL,tp)][0]:.3f}")
    if np.isnan(img_best) or img_best < 0.65:
        verdict="PROBE BROKEN — even image tokens don't decode the scene; fix probe (k/L2/target), NOT a scene verdict"
    elif txt_cells[(tL,tp)][0] >= 0.70:
        verdict="GO — scene IS decodable from text positions (proceed to Phase B)"
    elif txt_cells[(tL,tp)][0] >= 0.60:
        verdict="MARGINAL — reparametrize (target=counts, more layers/positions) before deciding"
    else:
        verdict="NO-GO — scene decodes at image tokens but NOT text positions -> pivot to attention-flow"
    log(f"VERDICT: {verdict}")
    json.dump(dict(mode=MODE,pca_k=PCA_K,image_baseline=img_best,best_text=[tL,str(tp),txt_cells[(tL,tp)][0]],
                   grid={f"{L}|{p}":grid[(L,p)] for (L,p) in grid},verdict=verdict),
              open(f"{OUT}/phaseA_{MODE}_summary.json","w"),indent=2,default=str)
    log(f"saved -> {OUT}/phaseA_{MODE}_summary.json")


def main():
    os.makedirs(OUT,exist_ok=True)
    log("="*70,f"\nE2 PHASE A v2  MODE={MODE}  layers={LAYERS}  pca_k={PCA_K}")
    import torch,transformers
    log(f"torch {torch.__version__}  transformers {transformers.__version__}  gpu {torch.cuda.get_device_name(0)}")
    if MODE=="analyze":
        data=dict(np.load(NPZ.replace('smoke','full')))
        log(f"loaded {NPZ.replace('smoke','full')}")
    else:
        data=extract()
    analyze(data)


if __name__=="__main__":
    main()
