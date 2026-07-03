#!/usr/bin/env python3
"""
E3 / Method A — A2 POSITION SWEEP (the causal localizer + the fix, in one).

For each RIPE item (perception intact, answer wrong) we re-present a visual PAYLOAD inline at a
ladder of reasoning positions p, then FREE-GENERATE a new final answer (greedy). The
Delta-corr(p) = corr(V1,p) - corr(V0,p) curve localizes where re-accessing the image fixes RIPE;
corr(V1)-corr(V_scr) isolates visual CONTENT from the mere act of inserting a block.

Instrument validated in A0 (`phaseA0_mechanics.py`): inline mid-assistant image injection conditions
the model (inj_inline 5/5), M-RoPE correct-by-construction, scrambled = valid placebo.

Payloads @ position p:  V0 null (no 2nd image) | V1 original image | V_scr patch-shuffled placebo
Positions: fracs {.10,.25,.50,.75,.90} of pre-</think> length + `think` (after </think>) + `ans` (before final answer token)
Controls: specificity (V1 on CORRECT items -> breakage) + conflict (inject a DIFFERENT scene on correct items -> must shift, confirming 2-image attention)
Substrate: pooled RIPE (A=0 & D_maj=1) + correct (A=1) from DSETs {full, hard}.

Greedy (temperature=0) so every difference is the image channel, not sampling. vLLM continuous batching.
Modes (argv[1]): smoke (2 RIPE, all pos/cond, prints answers to re-verify injection) | full.
Guarded under __main__ (vLLM spawn). Saves every (item x position x condition) record.
"""
import os, sys, io, re, json, time, base64, random
from collections import defaultdict

MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
DSETS  = os.environ.get("DSETS", "full,hard").split(",")
N_RIPE = int(os.environ.get("N_RIPE", "0"))            # 0 = all
N_CTRL = int(os.environ.get("N_CTRL", "20"))           # specificity + conflict pool
MAXTOK = int(os.environ.get("MAX_TOK", "8192"))
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED  = "\n\nPut your final answer in \\boxed{}."
THINK_CLOSE = 151668
FRACS  = [0.10, 0.25, 0.50, 0.75, 0.90]

SYN={"grey":"gray","matte":"rubber","metallic":"metal","shiny":"metal","big":"large","tiny":"small","block":"cube","ball":"sphere"}
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
def correct(text, gt):
    p=canon(extract_boxed(text)); g=canon(gt)
    return int(bool(g) and (p==g or g in p.split() or p in g.split()))
def scramble(img, n=8):
    from PIL import Image
    w,h=img.size; tw,th=w//n,h//n
    tiles=[img.crop((c*tw,r*th,(c+1)*tw,(r+1)*th)) for r in range(n) for c in range(n)]
    rnd=random.Random(0); rnd.shuffle(tiles)
    out=Image.new("RGB",(tw*n,th*n))
    for i,t in enumerate(tiles):
        r,c=divmod(i,n); out.paste(t,(c*tw,r*th))
    return out


def main():
    log("="*70, f"\nE3 A2 POSITION SWEEP  MODE={MODE}  DSETS={DSETS}  fracs={FRACS}+think+ans")
    from PIL import Image
    from transformers import AutoProcessor
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    probe=proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":"x"}]}],
                                   tokenize=False, add_generation_prompt=False)
    VIS=re.search(r"<\|vision_start\|>.*?<\|vision_end\|>", probe, re.DOTALL).group(0)

    # ---- pool RIPE + correct items across DSETs ----
    V2, GATE = {}, {}
    for d in DSETS:
        for l in open(f"{OUT}/v2_{d}_records.jsonl"):
            r=json.loads(l); V2.setdefault(r["qi"], r)
        for l in open(f"{OUT}/e1_gate_multi_{d}.jsonl"):
            g=json.loads(l); GATE.setdefault(g["qi"], g)
    ripe=[qi for qi,g in GATE.items() if g["A"]==0 and g["D_maj"]==1 and qi in V2]
    corr_items=[qi for qi,g in GATE.items() if g["A"]==1 and qi in V2]
    ripe.sort(); corr_items.sort()
    if N_RIPE: ripe=ripe[:N_RIPE]
    if MODE=="smoke": ripe=ripe[:2]; corr_items=corr_items[:2]
    log(f"pooled: RIPE={len(ripe)}  correct={len(corr_items)}")

    def img_of(qi): return Image.open(f"{IMGDIR}/{V2[qi]['image_filename']}").convert("RGB")
    def positions(qi):
        resp=list(V2[qi]["output_token_ids"])
        ti=resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1
        model_ans=canon(extract_boxed(V2[qi]["full_text"]))
        ai=len(resp)-1
        for i in range(len(resp)-1,-1,-1):
            dv=canon(tok.decode([resp[i]]))
            if dv and (dv==model_ans or dv in model_ans.split()): ai=i; break
        pos=[(f"f{f:.2f}", max(1,int(f*ti))) for f in FRACS]
        pos+=[("think", min(len(resp), ti+1)), ("ans", max(1, ai))]
        return resp, pos

    def user_tpl(qi):
        q=V2[qi]["question"]
        return proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":q+BOXED}]}],
                                        tokenize=False, add_generation_prompt=True)

    # ---- build the full job list: (tag, qi, pos_label, cond, prompt, images) ----
    jobs=[]
    def add(tag, qi, plabel, ptok, resp, cond, second_qi=None):
        prefix=tok.decode(resp[:ptok])
        inject = cond!="V0"
        prompt=user_tpl(qi)+prefix+(VIS if inject else "")
        imgs=[img_of(qi)]
        if inject:
            if cond=="V1":   imgs.append(img_of(qi))
            elif cond=="V_scr": imgs.append(scramble(img_of(qi)))
            elif cond=="conflict": imgs.append(img_of(second_qi))
        jobs.append(dict(tag=tag, qi=qi, pos=plabel, cond=cond, prompt=prompt,
                         images=imgs, gt=canon(V2[qi]["gt_norm"]), second=second_qi))

    for qi in ripe:
        resp,pos=positions(qi)
        for plabel,ptok in pos:
            for cond in ("V0","V1","V_scr"):
                add("ripe", qi, plabel, ptok, resp, cond)
    # specificity: V0/V1 on correct items at think+ans (does injection BREAK correct?)
    for qi in corr_items[:N_CTRL]:
        resp,pos=positions(qi)
        for plabel,ptok in [p for p in pos if p[0] in ("think","ans")]:
            for cond in ("V0","V1"):
                add("ctrl", qi, plabel, ptok, resp, cond)
    # conflict: inject a DIFFERENT scene at think on correct items -> answer must move (2-image attention)
    pool=corr_items[:N_CTRL]
    for k,qi in enumerate(pool[:min(10,len(pool))]):
        resp,pos=positions(qi); ti=[p for p in pos if p[0]=="think"][0][1]
        other=pool[(k+1)%len(pool)]
        add("conflict", qi, "think", ti, resp, "conflict", second_qi=other)
    log(f"total generations: {len(jobs)}")

    # ---- vLLM greedy ----
    import torch, vllm
    from vllm import LLM, SamplingParams
    log(f"torch {torch.__version__}  vllm {vllm.__version__}")
    t0=time.time()
    llm=LLM(model=MODEL, dtype="bfloat16", max_model_len=40960, gpu_memory_utilization=0.90,
            limit_mm_per_prompt={"image":2}, trust_remote_code=False)
    log(f"engine up {time.time()-t0:.0f}s")
    sp=SamplingParams(temperature=0.0, max_tokens=MAXTOK)
    reqs=[{"prompt":j["prompt"], "multi_modal_data":{"image":j["images"]}} for j in jobs]
    t0=time.time(); outs=llm.generate(reqs, sp); log(f"generated {len(outs)} in {time.time()-t0:.0f}s")

    # ---- score + save ----
    recs=[]
    with open(f"{OUT}/phaseA2_sweep_{MODE}.jsonl","w") as f:
        for j,o in zip(jobs, outs):
            txt=o.outputs[0].text; ans=canon(extract_boxed(txt)); ok=correct(txt, j["gt"])
            rec=dict(tag=j["tag"], qi=j["qi"], pos=j["pos"], cond=j["cond"], gt=j["gt"],
                     ans=ans, ok=ok, second=j["second"], out_len=len(o.outputs[0].token_ids), full_text=txt)
            recs.append(rec); f.write(json.dumps(rec)+"\n")

    # ---- Delta-corr curve ----
    def rate(tag,pos,cond):
        s=[r for r in recs if r["tag"]==tag and r["pos"]==pos and r["cond"]==cond]
        return (sum(r["ok"] for r in s)/len(s), len(s)) if s else (0.0,0)
    ORDER=[f"f{f:.2f}" for f in FRACS]+["think","ans"]
    log("="*70, "\nRIPE correction by position  (V1 = re-inject original, V0 = null, V_scr = placebo)")
    log(f"  {'pos':7} {'V0':>10} {'V1':>10} {'V_scr':>10} {'dCorr(V1-V0)':>13} {'content(V1-Vscr)':>17}")
    for p in ORDER:
        v0,n=rate("ripe",p,"V0"); v1,_=rate("ripe",p,"V1"); vs,_=rate("ripe",p,"V_scr")
        log(f"  {p:7} {v0:>10.3f} {v1:>10.3f} {vs:>10.3f} {v1-v0:>+13.3f} {v1-vs:>+17.3f}   (n={n})")
    # specificity + conflict
    for p in ("think","ans"):
        b0,_=rate("ctrl",p,"V0"); b1,n=rate("ctrl",p,"V1")
        log(f"  [specificity {p}] correct-item acc  V0={b0:.3f}  V1={b1:.3f}  (breakage if V1<V0, n={n})")
    cf=[r for r in recs if r["tag"]=="conflict"]
    if cf:
        moved=sum(1 for r in cf if not r["ok"])   # correct item, conflicting image -> answer should move OFF gt
        log(f"  [conflict] {moved}/{len(cf)} correct items MOVED off gt when a different scene was injected (2-image attention confirmed if high)")
    log(f"\nread: a positive dCorr(V1-V0) at some position = re-accessing the image CAUSALLY corrects RIPE there;")
    log(f"      peak position = the localizer; V1-V_scr>0 confirms it's visual CONTENT not insertion.")
    log(f"saved -> {OUT}/phaseA2_sweep_{MODE}.jsonl")


if __name__=="__main__":
    main()
