#!/usr/bin/env python3
"""
E3 / Method A — SIGNALS pass (HF, fast: single forwards, NO generation). Companion to phaseA2_vsweep.py.

Per RIPE item x position p in {0.25,0.50,0.75,think}: one forward of the original chain to p, capture the
model's state at p, keyed by (qi,pos) to merge with the vLLM outcomes in A3. Signals (MLP/residual PRIMARY,
Set-1 60/30):
  per-LAYER: MLP-write norm, attn-write norm, residual norm, cross-layer residual delta,
             logit-lens belief margin P(gt)-P(wrong) & entropy, cosine(hidden_p, image-token subspace)
  scalar:    next-token entropy; item meta (depth, chain_len, nobj).
(attention-to-IMAGE mass -> phaseA2_attn.py, eager/length-guarded, later.)
Modes: smoke (2) | full. Fast (~280 forwards ~15 min). Guarded under __main__.
"""
import os, sys, re, json, time, math
import torch
MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
DSETS  = os.environ.get("DSETS", "full,hard").split(",")
N_RIPE = int(os.environ.get("N_RIPE", "0"))
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED  = "\n\nPut your final answer in \\boxed{}."
THINK_CLOSE = 151668
POSFRACS = [0.25, 0.50, 0.75]
COLORS=["gray","red","blue","green","brown","purple","cyan","yellow"]; SHAPES=["cube","sphere","cylinder"]
SIZES=["large","small"]; MATS=["rubber","metal"]
SYN={"grey":"gray","matte":"rubber","metallic":"metal","shiny":"metal","big":"large","tiny":"small","block":"cube","ball":"sphere"}
VALUES=[str(i) for i in range(11)]+["yes","no"]+COLORS+SHAPES+SIZES+MATS
def log(*a): print(*a, flush=True)
def canon(s):
    if not s: return ""
    s=re.sub(r"\\(?:text|mathrm|mathbf|textbf|mathsf|textit)\s*\{([^{}]*)\}",r"\1",str(s)).replace("\\"," ")
    p=re.sub(r"[^a-z0-9]+"," ",s.lower()).strip()
    if re.search(r"\bmatte\b",p) or "non metal" in p or "not metal" in p: return "rubber"
    if re.search(r"\b(metallic|shiny)\b",p): return "metal"
    p=re.sub(r"\bbig\b","large",p); p=re.sub(r"\b(tiny|little)\b","small",p)
    return SYN.get(p,p)
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
    log("="*70,f"\nE3 SIGNALS PASS (HF)  MODE={MODE}  pos={POSFRACS}+think")
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    IMG_PAD=tok.convert_tokens_to_ids("<|image_pad|>")
    t0=time.time(); model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda").eval()
    log(f"model loaded {time.time()-t0:.0f}s")
    lm=None
    for path in ("model.language_model","language_model.model","model.model.language_model"):
        obj=model
        try:
            for p in path.split("."): obj=getattr(obj,p)
            if hasattr(obj,"layers") and hasattr(obj,"norm"): lm=obj; break
        except AttributeError: continue
    layers,final_norm,lm_head=lm.layers,lm.norm,model.get_output_embeddings(); NL=len(layers); log(f"LM layers={NL}")

    id2val={}
    for v in VALUES+list(SYN.keys()):
        for variant in (v," "+v):
            ids=tok(variant,add_special_tokens=False).input_ids
            if ids: id2val.setdefault(ids[0], canon(v))
    ans_ids=sorted(id2val.keys()); ans_ids_t=torch.tensor(ans_ids,device="cuda")
    def belief(lv,gt,wrong):
        p=torch.softmax(lv[ans_ids_t].float(),0).cpu(); d={}
        for i,tid in enumerate(ans_ids): d[id2val[tid]]=d.get(id2val[tid],0.0)+float(p[i])
        tot=sum(d.values()); ent=-sum((x/tot)*math.log(x/tot) for x in d.values() if x>0) if tot>0 else 0.0
        return round(d.get(gt,0.0)-d.get(wrong,0.0),4), round(ent,4)

    CAP={}; FLAG=[False]
    def mk(idx,kind):
        def hook(m,inp,out):
            if not FLAG[0]: return
            t=out[0] if isinstance(out,tuple) else out
            CAP.setdefault(idx,{})[kind]=float(t[0,-1].float().norm().item())
        return hook
    for i,ly in enumerate(layers):
        ly.mlp.register_forward_hook(mk(i,"mlp")); ly.self_attn.register_forward_hook(mk(i,"attn"))

    V2,GATE={},{}
    for d in DSETS:
        for l in open(f"{OUT}/v2_{d}_records.jsonl"):
            r=json.loads(l); V2.setdefault(r["qi"],r)
        for l in open(f"{OUT}/e1_gate_multi_{d}.jsonl"):
            g=json.loads(l); GATE.setdefault(g["qi"],g)
    ripe=sorted([qi for qi,g in GATE.items() if g["A"]==0 and g["D_maj"]==1 and qi in V2])
    if N_RIPE: ripe=ripe[:N_RIPE]
    if MODE=="smoke": ripe=ripe[:2]
    log(f"RIPE items: {len(ripe)}")

    def img_of(qi): return Image.open(f"{IMGDIR}/{V2[qi]['image_filename']}").convert("RGB")
    def user_tpl(qi):
        return proc.apply_chat_template([{"role":"user","content":[{"type":"image"},
                {"type":"text","text":V2[qi]['question']+BOXED}]}], tokenize=False, add_generation_prompt=True)
    def positions(qi):
        resp=list(V2[qi]["output_token_ids"]); ti=resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1
        return resp,[(f"f{f:.2f}",max(1,int(f*ti))) for f in POSFRACS]+[("think",min(len(resp),ti+1))]
    def img_span(ids0):
        ids=ids0.tolist(); i=0
        while i<len(ids):
            if ids[i]==IMG_PAD:
                j=i
                while j<len(ids) and ids[j]==IMG_PAD: j+=1
                return (i,j)
            i+=1
        return None

    @torch.no_grad()
    def signal_forward(qi, prefix_text, gt, wrong):
        enc=proc(text=[user_tpl(qi)+prefix_text],images=[img_of(qi)],return_tensors="pt")
        enc={k:v.to("cuda") for k,v in enc.items()}; span=img_span(enc["input_ids"][0])
        CAP.clear(); FLAG[0]=True; out=model(**enc,output_hidden_states=True); FLAG[0]=False
        hs=out.hidden_states; sig={k:[] for k in ("mlp","attn","resid","resid_delta","bmargin","bent","cos_img")}
        for L in range(NL):
            h=hs[L+1][0,-1]
            sig["resid"].append(round(float(h.float().norm()),3))
            sig["resid_delta"].append(round(float((hs[L+1][0,-1]-hs[L][0,-1]).float().norm()),3))
            sig["mlp"].append(round(CAP.get(L,{}).get("mlp",float("nan")),3))
            sig["attn"].append(round(CAP.get(L,{}).get("attn",float("nan")),3))
            ll=lm_head(final_norm(h.unsqueeze(0).to(lm_head.weight.dtype)))[0]; bm,be=belief(ll,gt,wrong)
            sig["bmargin"].append(bm); sig["bent"].append(be)
            if span:
                ir=hs[L+1][0,span[0]:span[1]].mean(0)
                sig["cos_img"].append(round(float(torch.cosine_similarity(h.float(),ir.float(),dim=0)),4))
            else: sig["cos_img"].append(float("nan"))
        fp=torch.softmax(out.logits[0,-1].float(),-1)
        sig["next_entropy"]=round(float(-(fp*torch.log(fp+1e-9)).sum()),4); sig["img_span_len"]=(span[1]-span[0]) if span else 0
        return sig

    t0=time.time(); n=0
    with open(f"{OUT}/phaseA2_sig_{MODE}.jsonl","w") as f:
        for qi in ripe:
            r=V2[qi]; gt=canon(r["gt_norm"]); wrong=canon(extract_boxed(r["full_text"])); resp,pos=positions(qi)
            for plabel,ptok in pos:
                sig=signal_forward(qi,tok.decode(resp[:ptok]),gt,wrong); n+=1
                f.write(json.dumps(dict(qi=qi,pos=plabel,gt=gt,wrong=wrong,depth=r["depth"],
                        chain_len=len(resp),nobj=len(r["scene"]["objects"]),sig=sig))+"\n")
                if MODE=="smoke":
                    log(f"  qi{qi} {plabel:6}: mlp[0/{NL//2}/{NL-1}]={sig['mlp'][0]:.1f}/{sig['mlp'][NL//2]:.1f}/{sig['mlp'][NL-1]:.1f} "
                        f"attn={sig['attn'][NL//2]:.1f} bmargin(mid)={sig['bmargin'][NL//2]:+.3f} next_ent={sig['next_entropy']:.2f} cos_img(mid)={sig['cos_img'][NL//2]:+.3f}")
    log(f"saved {n} item-positions -> {OUT}/phaseA2_sig_{MODE}.jsonl  ({time.time()-t0:.0f}s)")


if __name__=="__main__":
    main()
