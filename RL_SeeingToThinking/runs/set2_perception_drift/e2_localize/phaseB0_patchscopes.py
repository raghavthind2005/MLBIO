#!/usr/bin/env python3
"""
E2b Phase 0 (v2) — belief-readout GO/NO-GO.  Fix broken-instrument v1 (calibration=0).

Can we recover the model's *committed answer value* from an intermediate state, before it's
emitted? Prereq for tracking belief(t)/flip.

Three readouts, escalating:
  LL@ans   : logit-lens at the true answer-predicting position -> MUST recover the answer
             (validates vocab + position; if this fails, everything else is meaningless).
  PS@ans   : Patchscopes (patch state into "\\boxed{" readout) at the answer position (calibration).
  PS@think : Patchscopes at the </think> position (the real EARLY test: answer pre-emission?).
  (Ghandeharioun et al. 2401.06102; logit-lens/tuned-lens 2303.08112; Future Lens 2311.04897.)

Modes: smoke (12) | full (40). Guarded under __main__.
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
    s=re.sub(r"\\(?:text|mathrm|mathbf|mathsf)\s*\{([^{}]*)\}",r"\1",str(s))
    s=re.sub(r"[^a-z0-9]+"," ",s.lower()).strip()
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
def match(dv, ans): return bool(dv) and (dv==ans or dv in ans.split() or ans in dv.split())


def main():
    os.makedirs(OUT,exist_ok=True)
    log("="*70,f"\nE2b PHASE 0 v2  MODE={MODE}  layers={LAYERS}  readout={READOUT!r}")
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    t0=time.time(); model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda").eval()
    log(f"model loaded {time.time()-t0:.0f}s")

    lm=None
    for path in ("model.language_model","language_model.model","model.model.language_model"):
        obj=model
        try:
            for p in path.split("."): obj=getattr(obj,p)
            if hasattr(obj,"layers") and hasattr(obj,"norm"): lm=obj; log(f"text model at model.{path} (n_layers={len(obj.layers)})"); break
        except AttributeError: continue
    assert lm is not None, "text backbone not found"
    layers, norm, lm_head = lm.layers, lm.norm, model.get_output_embeddings()

    id2val={}
    for v in VALUES+list(SYN.keys()):
        for variant in (v," "+v):
            ids=tok(variant,add_special_tokens=False).input_ids
            if ids: id2val.setdefault(ids[0], canon(v))
    ans_ids=torch.tensor(sorted(id2val.keys()),device="cuda")
    log(f"answer vocab: {len(id2val)} ids -> {len(set(id2val.values()))} values")

    readout_ids=tok(READOUT,return_tensors="pt").input_ids.to("cuda")
    patch_pos=readout_ids.shape[1]-1

    def decode_ans(logits):
        return id2val[ans_ids[logits[ans_ids].argmax()].item()]
    def logit_lens(h):
        with torch.no_grad(): return decode_ans(lm_head(norm(h.unsqueeze(0).to(lm_head.weight.dtype)))[0])
    def patchscope(source_h, layer):
        def pre(mod,args,kwargs):
            args[0][0,patch_pos]=source_h.to(args[0].dtype); return (args,kwargs)
        h=layers[layer].register_forward_pre_hook(pre,with_kwargs=True)
        try:
            with torch.no_grad(): out=model(input_ids=readout_ids)
        finally: h.remove()
        return decode_ans(out.logits[0,-1])

    rows=[json.loads(l) for l in open(RECIN)][:N_ITEMS]
    log(f"items: {len(rows)}")
    acc={L:[0,0,0,0] for L in LAYERS}    # ll@ans, ps@ans, ps@think, n
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
        think_pos=P+(resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1)
        # true answer-token position: last resp token that decodes to the answer; state before it predicts it
        ans_tok=None
        for i in range(len(resp)-1,-1,-1):
            if match(canon(tok.decode([resp[i]])), model_ans): ans_tok=P+i; break
        ans_predict=max(0,(ans_tok-1) if ans_tok else full.shape[1]-1)
        for L in LAYERS:
            ll =match(logit_lens(hs[L][0,ans_predict]), model_ans)
            psa=match(patchscope(hs[L][0,ans_predict], L), model_ans)
            pst=match(patchscope(hs[L][0,think_pos], L), model_ans)
            acc[L][0]+=ll; acc[L][1]+=psa; acc[L][2]+=pst; acc[L][3]+=1
        if MODE=="smoke":
            L0=LAYERS[len(LAYERS)//2]
            log(f"[{k}] ans={model_ans!r} L{L0}: ll@ans={logit_lens(hs[L0][0,ans_predict])!r} "
                f"ps@ans={patchscope(hs[L0][0,ans_predict], L0)!r} ps@think={patchscope(hs[L0][0,think_pos], L0)!r}")

    log("="*70,"\nREADOUT VALIDATION (recover the model's OWN answer)")
    log(f"{'layer':>6} {'LL@ans(sanity)':>15} {'PS@ans(calib)':>14} {'PS@think(early)':>16}")
    best=(None,-1)
    for L in LAYERS:
        a,b,c,n=acc[L]
        log(f"{L:>6} {a/n:>15.3f} {b/n:>14.3f} {c/n:>16.3f}")
        if c/n>best[1]: best=(L,c/n)
    ll_best=max(acc[L][0]/acc[L][3] for L in LAYERS)
    bl,be=best; bc=acc[bl][1]/acc[bl][3]
    log("="*70,"\nGO / NO-GO")
    if ll_best < 0.7:
        verdict=f"INSTRUMENT BROKEN — logit-lens can't even recover the answer at its own position (LL@ans={ll_best:.2f}); fix vocab/position before any belief claim"
    elif be >= 0.55:
        verdict=f"GO — committed answer recoverable pre-emission (best layer {bl}, PS@think={be:.2f})"
    elif be >= 0.4:
        verdict=f"MARGINAL — try alt readout / tuned lens (PS@think={be:.2f})"
    else:
        verdict=f"NO-GO for Method B via this readout (LL sanity ok={ll_best:.2f} but PS@think={be:.2f}); lean on Method A (intervention)"
    log(f"LL sanity(best)={ll_best:.3f}  best-layer {bl}: PS@think={be:.3f} PS@ans={bc:.3f}  -> {verdict}")
    json.dump(dict(mode=MODE,readout=READOUT,ll_sanity=ll_best,best_layer=bl,ps_think=be,ps_calib=bc,
                   acc={L:acc[L] for L in LAYERS},verdict=verdict),
              open(f"{OUT}/phaseB0_{MODE}_summary.json","w"),indent=2,default=str)
    log(f"saved -> {OUT}/phaseB0_{MODE}_summary.json")


if __name__=="__main__":
    main()
