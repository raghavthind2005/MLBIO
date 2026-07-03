#!/usr/bin/env python3
"""
E3 / Method A — OUTCOMES sweep in vLLM (fast). vLLM proven faithful for the two-image config
(gate phaseA2_gate.py: 14/16; the 2 misses were MAX_NEW truncation, not injection — short cases agreed exactly).

Per RIPE item x position p in {0.25,0.50,0.75,think}: re-inject a payload inline at p, free-generate, score.
  V0 null | V1 original | V_scr scrambled placebo | V_viz visual oracle (zoom-crop) | V_text scene-graph oracle.
High max_tokens so mid-think re-reasoning COMPLETES (truncation was the only HF/vLLM disagreement).
Signals (MLP/residual/belief per layer) come from the companion HF pass phaseA2_sig.py, merged by (qi,pos) in A3.

Modes (argv[1]): smoke (2 RIPE) | full. Guarded under __main__ (vLLM spawn).
"""
import os, sys, re, json, time, random
MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
DSETS  = os.environ.get("DSETS", "full,hard").split(",")
N_RIPE = int(os.environ.get("N_RIPE", "0"))
MAXTOK = int(os.environ.get("MAX_TOK", "16384"))
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED  = "\n\nPut your final answer in \\boxed{}."
THINK_CLOSE = 151668
POSFRACS = [0.25, 0.50, 0.75]

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
def scramble(img,n=8):
    from PIL import Image
    w,h=img.size; tw,th=w//n,h//n
    tiles=[img.crop((c*tw,r*th,(c+1)*tw,(r+1)*th)) for r in range(n) for c in range(n)]
    rnd=random.Random(0); rnd.shuffle(tiles); out=Image.new("RGB",(tw*n,th*n))
    for i,t in enumerate(tiles): r,c=divmod(i,n); out.paste(t,(c*tw,r*th))
    return out
def viz_oracle(img,scene,margin=60,up=2):
    W,H=img.size; xs=[o["pixel_coords"][0] for o in scene["objects"]]; ys=[o["pixel_coords"][1] for o in scene["objects"]]
    x0=max(0,int(min(xs)-margin)); x1=min(W,int(max(xs)+margin)); y0=max(0,int(min(ys)-margin)); y1=min(H,int(max(ys)+margin))
    if x1<=x0 or y1<=y0: return img.resize((W*up,H*up))
    c=img.crop((x0,y0,x1,y1)); return c.resize((c.width*up,c.height*up))
def text_oracle(scene):
    return "[Reference — the image contains exactly these objects: "+"; ".join(
        f"{o['size']} {o['color']} {o['material']} {o['shape']}" for o in scene["objects"])+".]"


def main():
    log("="*70,f"\nE3 vLLM OUTCOMES SWEEP  MODE={MODE}  pos={POSFRACS}+think  MAXTOK={MAXTOK}")
    from transformers import AutoProcessor
    from PIL import Image
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    VIS=re.search(r"<\|vision_start\|>.*?<\|vision_end\|>",
                  proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":"x"}]}],
                  tokenize=False, add_generation_prompt=False), re.DOTALL).group(0)

    V2,GATE={},{}
    for d in DSETS:
        for l in open(f"{OUT}/v2_{d}_records.jsonl"):
            r=json.loads(l); V2.setdefault(r["qi"],r)
        for l in open(f"{OUT}/e1_gate_multi_{d}.jsonl"):
            g=json.loads(l); GATE.setdefault(g["qi"],g)
    ripe=sorted([qi for qi,g in GATE.items() if g["A"]==0 and g["D_maj"]==1 and qi in V2])
    if N_RIPE: ripe=ripe[:N_RIPE]
    if MODE=="smoke": ripe=ripe[:3]
    log(f"RIPE items: {len(ripe)}")

    def img_of(qi): return Image.open(f"{IMGDIR}/{V2[qi]['image_filename']}").convert("RGB")
    def user_tpl(qi):
        return proc.apply_chat_template([{"role":"user","content":[{"type":"image"},
                {"type":"text","text":V2[qi]['question']+BOXED}]}], tokenize=False, add_generation_prompt=True)
    def positions(qi):
        resp=list(V2[qi]["output_token_ids"]); ti=resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1
        return resp,[(f"f{f:.2f}",max(1,int(f*ti))) for f in POSFRACS]+[("think",min(len(resp),ti+1))]

    CONDS=["V0","V1","V_scr","V_viz","V_text"]
    jobs=[]
    for qi in ripe:
        r=V2[qi]; scene=r["scene"]; img=img_of(qi); gt=canon(r["gt_norm"]); resp,pos=positions(qi)
        for plabel,ptok in pos:
            base=user_tpl(qi)+tok.decode(resp[:ptok])
            specs={"V0":(base,[img]), "V1":(base+VIS,[img,img]), "V_scr":(base+VIS,[img,scramble(img)]),
                   "V_viz":(base+VIS,[img,viz_oracle(img,scene)]),
                   "V_text":(base+"\n\n"+text_oracle(scene)+"\n",[img])}
            for c in CONDS:
                prompt,imgs=specs[c]
                jobs.append(dict(qi=qi,pos=plabel,cond=c,prompt=prompt,images=imgs,gt=gt,depth=r["depth"],chain_len=len(resp)))
    log(f"jobs={len(jobs)}")

    import torch, vllm
    from vllm import LLM, SamplingParams
    log(f"torch {torch.__version__} vllm {vllm.__version__}")
    t0=time.time()
    llm=LLM(model=MODEL,dtype="bfloat16",max_model_len=40960,gpu_memory_utilization=0.90,
            limit_mm_per_prompt={"image":2},trust_remote_code=False)
    log(f"engine up {time.time()-t0:.0f}s")
    sp=SamplingParams(temperature=0.0,max_tokens=MAXTOK)
    reqs=[{"prompt":j["prompt"],"multi_modal_data":{"image":j["images"]}} for j in jobs]
    t0=time.time(); outs=llm.generate(reqs,sp); log(f"generated {len(outs)} in {time.time()-t0:.0f}s")

    recs={}
    with open(f"{OUT}/phaseA2_vsweep_{MODE}.jsonl","w") as f:
        for j,o in zip(jobs,outs):
            txt=o.outputs[0].text; key=(j["qi"],j["pos"])
            recs.setdefault(key,dict(qi=j["qi"],pos=j["pos"],gt=j["gt"],depth=j["depth"],chain_len=j["chain_len"],out={}))
            recs[key]["out"][j["cond"]]=dict(ans=canon(extract_boxed(txt)), ok=correct(txt,j["gt"]),
                                             tok=len(o.outputs[0].token_ids))
        for r in recs.values(): f.write(json.dumps(r)+"\n")
    allrec=list(recs.values())

    def rate(pos,c):
        s=[r for r in allrec if r["pos"]==pos and c in r["out"]]
        return (sum(x["out"][c]["ok"] for x in s)/len(s), len(s)) if s else (0.0,0)
    log("="*70,"\nRIPE correction by position (net of V0):")
    log(f"  {'pos':7} {'V0':>6} {'V1':>6} {'V_scr':>6} {'V_viz':>6} {'V_text':>6} | dViz  dText")
    for p in [f"f{f:.2f}" for f in POSFRACS]+["think"]:
        v0,n=rate(p,"V0"); rr={c:rate(p,c)[0] for c in CONDS}
        log(f"  {p:7} {rr['V0']:>6.2f} {rr['V1']:>6.2f} {rr['V_scr']:>6.2f} {rr['V_viz']:>6.2f} {rr['V_text']:>6.2f} | "
            f"{rr['V_viz']-v0:>+5.2f} {rr['V_text']-v0:>+5.2f}  (n={n})")
    # truncation watch
    trunc=sum(1 for r in allrec for c in r["out"] if r["out"][c]["tok"]>=MAXTOK-2)
    log(f"\ntruncation (hit MAXTOK): {trunc} cells  (if high, raise MAX_TOK)")
    log("read: V_text≫V0 => perfect info fixes RIPE (not H4). V_viz>V1 => better VISION helps (H3). V1≈V0 => same image redundant.")
    log(f"saved -> {OUT}/phaseA2_vsweep_{MODE}.jsonl")


if __name__=="__main__":
    main()
