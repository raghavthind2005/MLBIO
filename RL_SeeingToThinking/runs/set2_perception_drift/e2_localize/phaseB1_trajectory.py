#!/usr/bin/env python3
"""
E2b Phase 1 — the DEFINITIVE belief-flip test (soft, position-swept).

Instrument validated in Phase 0 (Patchscopes PS@ans=1.0 @ L30). Here we sweep MANY positions
across each RIPE chain and read a SOFT belief with P(correct)-P(wrong) (the difference cancels
the readout's prior). If P(correct) leads early and P(wrong) overtakes late -> that IS the flip.

  belief(t) via Patchscopes @ {24,30}: patch state into "\\boxed{" readout -> softmax over the
  CLEVR answer vocab -> P(gt) and P(model's wrong answer). Anchor: at the answer position the
  belief must lean to the model's emitted (wrong) answer (trajectory endpoint sanity).

RIPE = multi-sample majority (A=0 & D_maj=1); matched CORRECT controls (belief must NOT flip).
Saves full per-(item x position x layer) trajectories. Modes: smoke (3+3) | full (17+17).
"""
import os, sys, io, re, json, time
import numpy as np

MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
V2     = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out/v2_full_records.jsonl"
MULTI  = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out/e1_gate_multi_full.jsonl"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED  = "\n\nPut your final answer in \\boxed{}."
THINK_CLOSE = 151668
LAYERS = [24, 30]
READOUT = "Answer: \\boxed{"
FRACS  = [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]

COLORS=["gray","red","blue","green","brown","purple","cyan","yellow"]; SHAPES=["cube","sphere","cylinder"]
SIZES=["large","small"]; MATS=["rubber","metal"]
SYN={"grey":"gray","matte":"rubber","metallic":"metal","shiny":"metal","big":"large","tiny":"small","block":"cube","ball":"sphere"}
VALUES=[str(i) for i in range(11)]+["yes","no"]+COLORS+SHAPES+SIZES+MATS

def log(*a): print(*a, flush=True)
def canon(s):
    s=re.sub(r"\\(?:text|mathrm|mathbf|mathsf)\s*\{([^{}]*)\}",r"\1",str(s))
    return SYN.get(re.sub(r"[^a-z0-9]+"," ",s.lower()).strip(), re.sub(r"[^a-z0-9]+"," ",s.lower()).strip())
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
    log("="*70,f"\nE2b PHASE 1 belief trajectory  MODE={MODE}  layers={LAYERS}")
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
            if hasattr(obj,"layers") and hasattr(obj,"norm"): lm=obj; break
        except AttributeError: continue
    layers,norm,lm_head=lm.layers,lm.norm,model.get_output_embeddings()

    id2val={}
    for v in VALUES+list(SYN.keys()):
        for variant in (v," "+v):
            ids=tok(variant,add_special_tokens=False).input_ids
            if ids: id2val.setdefault(ids[0], canon(v))
    ans_ids=torch.tensor(sorted(id2val.keys()),device="cuda"); id_list=sorted(id2val.keys())
    readout_ids=tok(READOUT,return_tensors="pt").input_ids.to("cuda"); patch_pos=readout_ids.shape[1]-1

    def val_probs(full_logits):
        p=torch.softmax(full_logits[ans_ids].float(),0).cpu().numpy(); d={}
        for i,tid in enumerate(id_list): d[id2val[tid]]=d.get(id2val[tid],0.0)+float(p[i])
        return d
    def patchscope_probs(h,layer):
        def pre(m,a,k): a[0][0,patch_pos]=h.to(a[0].dtype); return (a,k)
        hh=layers[layer].register_forward_pre_hook(pre,with_kwargs=True)
        try:
            with torch.no_grad(): out=model(input_ids=readout_ids)
        finally: hh.remove()
        return val_probs(out.logits[0,-1])
    def ll_probs(h):
        with torch.no_grad(): return val_probs(lm_head(norm(h.unsqueeze(0).to(lm_head.weight.dtype)))[0])

    # ---- select items ----
    v2={json.loads(l)["qi"]:json.loads(l) for l in open(V2)}
    multi={json.loads(l)["qi"]:json.loads(l) for l in open(MULTI)}
    ripe=[qi for qi,m in multi.items() if m["A"]==0 and m["D_maj"]==1]
    correct=[qi for qi,m in multi.items() if m["A"]==1]
    if MODE=="smoke": ripe,correct=ripe[:3],correct[:3]
    else: correct=correct[:len(ripe)]
    items=[(qi,True) for qi in ripe]+[(qi,False) for qi in correct]
    log(f"RIPE={len(ripe)} correct-controls={len(correct)}")

    recs=[]
    fout=open(f"{OUT}/phaseB1_{MODE}_traj.jsonl","w")
    for qi,is_ripe in items:
        r=v2[qi]; gt=canon(r["gt_norm"]); model_ans=canon(extract_boxed(r["full_text"]))
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
        think=P+(resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1); tl=max(1,think-P)
        ans_tok=None
        for i in range(len(resp)-1,-1,-1):
            dv=canon(tok.decode([resp[i]]))
            if dv and (dv==model_ans or dv in model_ans.split()): ans_tok=P+i; break
        ans_predict=max(0,(ans_tok-1) if ans_tok else full.shape[1]-1)
        POS=[("f%.2f"%f, P+int(f*tl)) for f in FRACS]+[("think",think),("ans",ans_predict)]
        traj={}
        for L in LAYERS:
            for lab,idx in POS:
                pp=patchscope_probs(hs[L][0,idx],L)
                traj[f"{L}|{lab}"]=dict(pgt=round(pp.get(gt,0.0),4), pwrong=round(pp.get(model_ans,0.0),4))
            traj[f"{L}|ans_ll"]=dict(pgt=round(ll_probs(hs[L][0,ans_predict]).get(gt,0.0),4),
                                     pwrong=round(ll_probs(hs[L][0,ans_predict]).get(model_ans,0.0),4))
        rec=dict(qi=qi,is_ripe=is_ripe,gt=gt,model_ans=model_ans,depth=r["depth"],traj=traj)
        recs.append(rec); fout.write(json.dumps(rec)+"\n")
        # per-item margin at L30
        m=lambda lab: traj[f"30|{lab}"]["pgt"]-traj[f"30|{lab}"]["pwrong"]
        early=np.mean([m("f%.2f"%f) for f in FRACS[:3]]); late=m("think"); anc=m("ans")
        log(f"[{'RIPE' if is_ripe else 'corr'} qi{qi}] gt={gt!r} wrong={model_ans!r} | L30 margin early={early:+.3f} think={late:+.3f} ans_anchor={anc:+.3f}")
    fout.close()

    # ---- flip analysis (L30 patchscope) ----
    log("="*70,"\nFLIP ANALYSIS (L30, margin = P(gt) - P(wrong))")
    def marg(g,lab): return g["traj"]["30|"+lab]["pgt"]-g["traj"]["30|"+lab]["pwrong"]
    def early_marg(g): return float(np.mean([marg(g,"f%.2f"%f) for f in FRACS[:3]]))
    def summ(group,name):
        if not group: return
        E=float(np.mean([early_marg(g) for g in group]))
        Lm=float(np.mean([marg(g,"think") for g in group]))
        flips=sum(1 for g in group if early_marg(g)>0.01 and marg(g,"think")<-0.01)
        anc=float(np.mean([g["traj"]["30|ans"]["pwrong"]-g["traj"]["30|ans"]["pgt"] for g in group]))
        log(f"  {name}: n={len(group)}  early_margin={E:+.3f}  think_margin={Lm:+.3f}  "
            f"FLIP(correct->wrong)={flips}/{len(group)}  ans-anchor(pwrong-pgt)={anc:+.3f}")
    summ([g for g in recs if g["is_ripe"]],"RIPE   ")
    summ([g for g in recs if not g["is_ripe"]],"correct")
    log("\nread: RIPE early_margin>0 & think_margin<0 with FLIPs => belief flips during reasoning (Method B ALIVE).")
    log("      RIPE margins ~0 throughout => no maintained belief (Method B closed); ans-anchor>0 validates endpoint.")
    log(f"saved trajectories -> {OUT}/phaseB1_{MODE}_traj.jsonl")
    json.dump(dict(mode=MODE,n_ripe=len(ripe),n_correct=len(correct)),open(f"{OUT}/phaseB1_{MODE}_summary.json","w"),indent=2)


if __name__=="__main__":
    main()
