#!/usr/bin/env python3
"""
E3 / Method A — HF-vs-vLLM EQUIVALENCE GATE for the TWO-IMAGE mid-think injection (the sweep's config).

Batched HF is unfaithful (mrope+left-pad). Sequential HF is correct but slow. vLLM is the only fast option,
but its faithfulness for our injection is unproven. This gate settles it: build identical two-image injection
cases, generate with HF (reference, sequential greedy) and vLLM (greedy), compare the boxed answers.

  HF==vLLM on (almost) all cases  => vLLM is faithful for our config -> run the full sweep in vLLM (~15 min).
  disagreement                    => vLLM is out -> sequential HF with trimmed scope.

HF and vLLM run in ONE process (HF first, freed, then vLLM at modest gpu_mem) so inputs are byte-identical.
Guarded under __main__ (vLLM spawn). ~16 cases.
"""
import os, sys, re, json, time, random, gc
import torch

DSETS  = os.environ.get("DSETS", "full,hard").split(",")
MAXNEW = int(os.environ.get("MAX_NEW", "1024"))
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED  = "\n\nPut your final answer in \\boxed{}."
THINK_CLOSE = 151668

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
    log("="*70,"\nE3 HF-vs-vLLM EQUIVALENCE GATE (two-image mid-think injection)")
    from transformers import AutoProcessor, AutoModelForImageTextToText
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
    ripe=sorted([qi for qi,g in GATE.items() if g["A"]==0 and g["D_maj"]==1 and qi in V2])[:2]

    def img_of(qi): return Image.open(f"{IMGDIR}/{V2[qi]['image_filename']}").convert("RGB")
    def user_tpl(qi):
        return proc.apply_chat_template([{"role":"user","content":[{"type":"image"},
                {"type":"text","text":V2[qi]['question']+BOXED}]}], tokenize=False, add_generation_prompt=True)
    def prefixes(qi):
        resp=list(V2[qi]["output_token_ids"]); ti=resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1
        return {"f0.50":tok.decode(resp[:max(1,int(0.5*ti))]), "think":tok.decode(resp[:min(len(resp),ti+1)])}

    # build identical cases (prompt string + image list) — used verbatim by BOTH backends
    cases=[]
    for qi in ripe:
        scene=V2[qi]["scene"]; img=img_of(qi); px=prefixes(qi)
        for pl,prefix in px.items():
            base=user_tpl(qi)+prefix
            cases.append((qi,pl,"V0",     base,               [img]))
            cases.append((qi,pl,"V1",     base+VIS,           [img,img]))
            cases.append((qi,pl,"V_viz",  base+VIS,           [img,viz_oracle(img,scene)]))
            cases.append((qi,pl,"V_text", base+"\n\n"+text_oracle(scene)+"\n", [img]))
    log(f"cases={len(cases)}")

    # ---- HF reference (sequential greedy) ----
    t0=time.time(); model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda").eval()
    log(f"HF loaded {time.time()-t0:.0f}s")
    hf=[]
    for qi,pl,c,prompt,imgs in cases:
        enc=proc(text=[prompt],images=imgs,return_tensors="pt"); enc={k:v.to("cuda") for k,v in enc.items()}
        with torch.no_grad(): o=model.generate(**enc,max_new_tokens=MAXNEW,do_sample=False)
        hf.append(canon(extract_boxed(tok.decode(o[0][enc["input_ids"].shape[1]:],skip_special_tokens=True))))
    log(f"HF done {time.time()-t0:.0f}s")
    del model; gc.collect(); torch.cuda.empty_cache()

    # ---- vLLM (same inputs) ----
    import vllm
    from vllm import LLM, SamplingParams
    llm=LLM(model=MODEL,dtype="bfloat16",max_model_len=20480,gpu_memory_utilization=0.55,
            limit_mm_per_prompt={"image":2},trust_remote_code=False)
    sp=SamplingParams(temperature=0.0,max_tokens=MAXNEW)
    reqs=[{"prompt":prompt,"multi_modal_data":{"image":imgs}} for (_,_,_,prompt,imgs) in cases]
    outs=llm.generate(reqs,sp)
    vl=[canon(extract_boxed(o.outputs[0].text)) for o in outs]

    # ---- compare ----
    log("="*70,"\nHF vs vLLM (two-image mid-think injection):")
    agree=0
    for (qi,pl,c,_,_),h,v in zip(cases,hf,vl):
        ok=(h==v); agree+=ok
        log(f"  qi{qi} {pl:6} {c:7} HF={h!r:>8} vLLM={v!r:>8}  {'ok' if ok else 'MISMATCH'}")
    log(f"\nAGREEMENT: {agree}/{len(cases)}")
    log("  >= ~90% => vLLM faithful for our config -> run full sweep in vLLM.  low => vLLM out -> sequential HF.")


if __name__=="__main__":
    main()
