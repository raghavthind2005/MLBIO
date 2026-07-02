#!/usr/bin/env python3
"""
E2b Phase 0 — Patchscopes readout GO/NO-GO.

Question: can we recover the model's *committed answer value* from an intermediate
reasoning-state, BEFORE it is emitted? (prerequisite for tracking belief(t) and its flip)

Method (Patchscopes, Ghandeharioun et al. 2401.06102; training-free, no probe, no judge,
reads a FROZEN state -> no re-access): take a source hidden state h at (layer L, source
position) from the model's real chain; PATCH it into the residual of a tiny readout prompt
"Answer: \\boxed{" at layer L, last position; read the next-token distribution restricted to
the CLEVR answer vocabulary -> the value that state "commits to".

Validation (all vs the MODEL's OWN answer -> tests the READOUT, not correctness):
  - CALIBRATION: patch the pre-answer state  -> must recover the emitted answer (sanity).
  - EARLY (the real test): patch the </think> state (pre-emission) -> recover the answer?
  - IMAGE (sanity): patch mean image-token state -> decodes a scene-present attribute?
GO if some layer gives EARLY recovery >> chance with CALIBRATION high.

Modes (argv[1]): smoke (12) | full (40). Guarded under __main__.
"""
import os, sys, io, re, json, time
import numpy as np

MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
RECIN  = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out/v2_full_records.jsonl"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED  = "\n\nPut your final answer in \\boxed{}."
IMAGE_PAD_ID, THINK_CLOSE = 151655, 151668
LAYERS = [12, 18, 24, 30, 34] if MODE != "smoke" else [18, 24, 30]
N_ITEMS = 40 if MODE != "smoke" else 12
READOUT = "Answer: \\boxed{"

COLORS=["gray","red","blue","green","brown","purple","cyan","yellow"]; SHAPES=["cube","sphere","cylinder"]
SIZES=["large","small"]; MATS=["rubber","metal"]
SYN={"grey":"gray","matte":"rubber","metallic":"metal","shiny":"metal","big":"large","tiny":"small","block":"cube","ball":"sphere"}
VALUES=[str(i) for i in range(11)]+["yes","no"]+COLORS+SHAPES+SIZES+MATS

def log(*a): print(*a, flush=True)
def canon(s):
    s=re.sub(r"[^a-z0-9]+"," ",str(s).lower()).strip()
    return SYN.get(s,s)
def extract_boxed(t):
    i=t.rfind("\\boxed{")
    if i<0: return None
    j,d,b=i+7,1,[]
    while j<len(t) and d:
        c=t[j]
        if c=="{":d+=1
        elif c=="}":d-=1
        if d:b.append(c)
        j+=1
    return "".join(b)


def main():
    os.makedirs(OUT,exist_ok=True)
    log("="*70,f"\nE2b PHASE 0  Patchscopes  MODE={MODE}  layers={LAYERS}  readout={READOUT!r}")
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    t0=time.time(); model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda").eval()
    log(f"model loaded {time.time()-t0:.0f}s")

    # locate decoder layers
    layers=None
    for path in ("model.language_model.layers","language_model.model.layers","model.model.language_model.layers"):
        obj=model
        try:
            for p in path.split("."): obj=getattr(obj,p)
            layers=obj; log(f"decoder layers at model.{path}  (n={len(layers)})"); break
        except AttributeError: continue
    assert layers is not None, "could not find decoder layers"

    # answer vocabulary -> first-token id -> canonical value
    id2val={}
    for v in VALUES+list(SYN.keys()):
        for variant in (v," "+v):
            ids=tok(variant,add_special_tokens=False).input_ids
            if ids: id2val.setdefault(ids[0], canon(v))
    ans_ids=torch.tensor(sorted(id2val.keys()),device="cuda")
    log(f"answer vocab: {len(id2val)} token-ids -> {len(set(id2val.values()))} values")

    readout_ids=tok(READOUT,return_tensors="pt").input_ids.to("cuda")
    patch_pos=readout_ids.shape[1]-1
    lm_head=model.get_output_embeddings()

    def patchscope(source_h, layer):
        """patch source_h into readout residual at (layer, patch_pos); return decoded value."""
        cell=layers[layer]
        def pre_hook(mod,args,kwargs):
            hs=args[0]; hs[0,patch_pos]=source_h.to(hs.dtype); return (args,kwargs)
        h=cell.register_forward_pre_hook(pre_hook,with_kwargs=True)
        try:
            with torch.no_grad(): out=model(input_ids=readout_ids)
        finally: h.remove()
        logits=out.logits[0,-1]
        pick=ans_ids[logits[ans_ids].argmax()].item()
        return id2val[pick]

    rows=[json.loads(l) for l in open(RECIN)][:N_ITEMS]
    log(f"items: {len(rows)}")
    # accuracy accumulators: layer -> [calib_correct, early_correct, n]
    acc={L:[0,0,0] for L in LAYERS}; img_hits=0; img_n=0
    for k,r in enumerate(rows):
        model_ans=canon(extract_boxed(r["full_text"]))
        if not model_ans: continue
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
        # positions
        think_pos=P+(resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1)
        ans_pos=full.shape[1]-2                                  # state predicting the last emitted token (~answer)
        img_pos=(full[0]==IMAGE_PAD_ID).nonzero(as_tuple=True)[0]
        scene_vals={canon(o[a]) for o in r["scene"]["objects"] for a in ("color","shape","size","material")}
        for L in LAYERS:
            calib=patchscope(hs[L][0,ans_pos],L)
            early=patchscope(hs[L][0,think_pos],L)
            acc[L][0]+=int(calib==model_ans); acc[L][1]+=int(early==model_ans); acc[L][2]+=1
        imv=patchscope(hs[LAYERS[len(LAYERS)//2]][0,img_pos].mean(0),LAYERS[len(LAYERS)//2])
        img_hits+=int(imv in scene_vals); img_n+=1
        if MODE=="smoke":
            L0=LAYERS[len(LAYERS)//2]
            log(f"[{k}] gt_model_ans={model_ans!r} | L{L0} calib={patchscope(hs[L0][0,ans_pos],L0)!r} "
                f"early={patchscope(hs[L0][0,think_pos],L0)!r} img={imv!r}")

    log("="*70,"\nPATCHSCOPES VALIDATION  (recovery of the model's OWN answer)")
    log(f"{'layer':>6} {'calibration':>12} {'early(</think>)':>16}")
    best=(None,-1)
    for L in LAYERS:
        c,e,n=acc[L]
        log(f"{L:>6} {c/n:>12.3f} {e/n:>16.3f}")
        if e/n>best[1]: best=(L,e/n)
    log(f"image-token -> scene-attribute hit rate: {img_hits/max(1,img_n):.3f}")
    bl,be=best; bc=acc[bl][0]/acc[bl][2]
    log("="*70,"\nGO / NO-GO")
    verdict=("GO — committed answer recoverable pre-emission (proceed to Phase 1 belief-trajectory)"
             if (be>=0.60 and bc>=0.80) else
             ("MARGINAL — try alt readout / tuned lens" if be>=0.45 else
              "NO-GO — belief not readable via this readout; try other readouts, else rely on Method A (intervention)"))
    log(f"best layer {bl}: early={be:.3f} calib={bc:.3f}  ->  {verdict}")
    json.dump(dict(mode=MODE,readout=READOUT,best_layer=bl,early=be,calib=bc,
                   img_hit=img_hits/max(1,img_n),acc={L:acc[L] for L in LAYERS},verdict=verdict),
              open(f"{OUT}/phaseB0_{MODE}_summary.json","w"),indent=2,default=str)
    log(f"saved -> {OUT}/phaseB0_{MODE}_summary.json")


if __name__=="__main__":
    main()
