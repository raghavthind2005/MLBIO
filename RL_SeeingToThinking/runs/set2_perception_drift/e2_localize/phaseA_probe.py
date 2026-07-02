#!/usr/bin/env python3
"""
E2 Phase A (v3) — build + VALIDATE the scene-state probe.  GO/NO-GO GATE.

v3 rules out the "under-powered probe" caveat (amendment A): PCA-k SWEEP {16,32,64,128}
with fast closed-form RIDGE probes, so a text-position NO-GO can't be blamed on dropping a
weak low-variance signal. Image-token baseline (must be HIGH) certifies the probe works.

Extract hidden states once (GPU) -> save .npz -> probe sweep (cheap, numpy).
Modes (argv[1]): smoke (40) | full (179) | analyze (reuse saved full .npz, probe only).
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
POS_FRAC = [0.25, 0.5, 0.75, 0.95]
POSN     = ["image_mean", "prompt_end"] + POS_FRAC
N_ITEMS  = 40 if MODE == "smoke" else 179
KS       = [16, 32, 64, 128]
RIDGE_LAM = 1.0

COLORS = ["gray","red","blue","green","brown","purple","cyan","yellow"]
SHAPES = ["cube","sphere","cylinder"]; SIZES = ["large","small"]; MATS = ["rubber","metal"]
ATTRS  = [("color",c) for c in COLORS]+[("shape",s) for s in SHAPES]+[("size",z) for z in SIZES]+[("material",m) for m in MATS]

def log(*a): print(*a, flush=True)
def presence(scene): return np.array([int(any(o[a]==v for o in scene["objects"])) for a,v in ATTRS], dtype=np.float32)
def bal_acc(y,pred):
    accs=[(pred[y==c]==c).mean() for c in (0,1) if (y==c).sum()>0]
    return float(np.mean(accs)) if len(accs)==2 else None
def ridge_bal_acc(Xtr,ytr,Xte,yte,lam=RIDGE_LAM):
    Xm=Xtr.mean(0); ym=ytr.mean(); Xc=Xtr-Xm; D=Xc.shape[1]
    w=np.linalg.solve(Xc.T@Xc+lam*np.eye(D), Xc.T@(ytr-ym))
    return bal_acc(yte, (((Xte-Xm)@w+ym)>0.5).astype(np.float64))

# ---------------------------------------------------------------- extraction (GPU)
def extract():
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    rows=[json.loads(l) for l in open(RECIN)]
    corr=[r for r in rows if r["parsed"]["correct"]][:N_ITEMS]; log(f"correct items: {len(corr)}")
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    t0=time.time(); model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda").eval()
    log(f"model loaded {time.time()-t0:.0f}s")
    close_ids=tok("</think>",add_special_tokens=False).input_ids
    def find_close(r):
        n=len(close_ids)
        for i in range(len(r)-n+1):
            if r[i:i+n]==close_ids: return i
        return len(r)
    feats={(L,p):[] for L in LAYERS for p in POSN}; Y,SC=[],[]; t0=time.time()
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
        hs=out.hidden_states; img_pos=(full[0]==IMAGE_PAD_ID).nonzero(as_tuple=True)[0]
        r_end=P+find_close(resp); tl=max(1,r_end-P); pidx={"prompt_end":P-1}
        for f in POS_FRAC: pidx[f]=min(full.shape[1]-1,P+int(f*tl))
        for L in LAYERS:
            for p in POSN:
                v=hs[L][0,img_pos].float().mean(0) if p=="image_mean" else hs[L][0,pidx[p]].float()
                feats[(L,p)].append(v.cpu().numpy())
        Y.append(presence(r["scene"])); SC.append(r["image_index"]); del out,hs,full
        if (k+1)%20==0: log(f"  {k+1}/{len(corr)} ({time.time()-t0:.0f}s)")
    save={f"{L}|{p}":np.stack(feats[(L,p)]) for L in LAYERS for p in POSN}
    save["Y"]=np.stack(Y); save["SC"]=np.array(SC); np.savez_compressed(NPZ,**save); log(f"saved -> {NPZ}")
    return save

# ---------------------------------------------------------------- probe sweep (cheap)
def analyze(data):
    Y=data["Y"]; SC=data["SC"]; n=Y.shape[0]
    us=np.array(sorted(set(SC.tolist()))); np.random.RandomState(0).shuffle(us)
    tr_sc=set(us[:int(0.7*len(us))].tolist())
    tr=np.array([s in tr_sc for s in SC]); te=~tr
    log(f"items={n}  scene-split {tr.sum()} train / {te.sum()} test  ({len(tr_sc)}/{len(us)} scenes)")
    log("="*70,f"\nPROBE SWEEP (PCA-k in {KS} + ridge; balanced held-out acc, chance=0.5; best-over-k reported)")
    log(f"{'layer':>6} {'pos':>11} {'bestk':>5} {'bal_acc':>8} {'shuf':>6} {'select':>7} {'nmarg':>5}")
    grid={}; perk_text={k:[] for k in KS}
    for L in LAYERS:
        for p in POSN:
            X=data[f"{L}|{p}"].astype(np.float64)
            mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Xn=(X-mu)/sd
            _,_,Vt=np.linalg.svd(Xn[tr],full_matrices=False)
            best=(-1.0,None,float("nan"),0)
            for k in KS:
                if k>tr.sum()-1: continue
                Xp=Xn@Vt[:k].T; accs,shufs=[],[]
                for j in range(len(ATTRS)):
                    y=Y[:,j]
                    if y[tr].sum() in (0,tr.sum()): continue
                    a=ridge_bal_acc(Xp[tr],y[tr],Xp[te],y[te])
                    yp=y[tr].copy(); np.random.RandomState(j).shuffle(yp)
                    s=ridge_bal_acc(Xp[tr],yp,Xp[te],y[te])
                    if a is not None: accs.append(a)
                    if s is not None: shufs.append(s)
                acc=float(np.mean(accs)) if accs else float("nan")
                shuf=float(np.mean(shufs)) if shufs else float("nan")
                if not np.isnan(acc):
                    if acc>best[0]: best=(acc,k,shuf,len(accs))
                    if p not in ("image_mean","prompt_end"): perk_text[k].append(acc)
            grid[(L,p)]=best
            log(f"{L:>6} {str(p):>11} {str(best[1]):>5} {best[0]:>8.3f} {best[2]:>6.3f} {best[0]-best[2]:>7.3f} {best[3]:>5}")
    log("per-k mean TEXT bal_acc:", {k: round(float(np.mean(v)),3) for k,v in perk_text.items() if v})
    img_best=max((grid[(L,"image_mean")][0] for L in LAYERS), default=-1)
    txt={(L,p):grid[(L,p)] for L in LAYERS for p in POSN if p not in ("image_mean","prompt_end")}
    (tL,tp),tb=max(txt.items(), key=lambda kv: kv[1][0])
    log("="*70,"\nGO / NO-GO")
    log(f"image-token baseline (best over layers/k): bal_acc {img_best:.3f}   <- must be HIGH or probe is broken")
    log(f"best TEXT position (best over layers/k):    layer {tL} pos {tp} k{tb[1]}  bal_acc {tb[0]:.3f}")
    if img_best < 0.65:
        verdict="PROBE BROKEN — image tokens don't decode; fix probe, NOT a scene verdict"
    elif tb[0] >= 0.70:
        verdict="GO — scene decodable from text positions (proceed to Phase B)"
    elif tb[0] >= 0.60:
        verdict="MARGINAL — weak text signal; decide with care"
    else:
        verdict="NO-GO — scene decodes at image tokens but NOT text positions (even w/ k-sweep) -> pivot to attention-flow"
    log(f"VERDICT: {verdict}")
    json.dump(dict(mode=MODE,ks=KS,image_baseline=img_best,best_text=[tL,str(tp),tb[1],tb[0]],
                   per_k_text={k:(float(np.mean(v)) if v else None) for k,v in perk_text.items()},
                   grid={f"{L}|{p}":grid[(L,p)] for (L,p) in grid},verdict=verdict),
              open(f"{OUT}/phaseA_{MODE}_summary.json","w"),indent=2,default=str)
    log(f"saved -> {OUT}/phaseA_{MODE}_summary.json")


def main():
    os.makedirs(OUT,exist_ok=True)
    log("="*70,f"\nE2 PHASE A v3  MODE={MODE}  layers={LAYERS}  KS={KS}")
    import torch,transformers
    log(f"torch {torch.__version__}  transformers {transformers.__version__}  gpu {torch.cuda.get_device_name(0)}")
    data=dict(np.load(NPZ.replace('smoke','full'))) if MODE=="analyze" else extract()
    analyze(data)


if __name__=="__main__":
    main()
