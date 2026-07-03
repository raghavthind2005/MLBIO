#!/usr/bin/env python3
"""
E3 / Method A — A2 INSTRUMENT VERIFICATION (in vLLM). Run BEFORE trusting any sweep number.

The sweep smoke showed Delta-corr(V1-V0)=0 and conflict=no-move. That is EITHER a real finding
(re-injected image attended but redundant) OR a bug (the inline 2nd image is not conditioning the
model in vLLM: M-RoPE / placeholder expansion). A0 proved inline injection works in HF; this proves
whether it works IN vLLM, using the SAME construction the sweep uses.

Decisive checks on EASY items the model normally answers correctly (greedy):
  normal        image in user turn                                  -> reference upper bound
  sole_real     NO user image; real image injected INLINE at p      -> MUST ~= normal  (vLLM inline injection conditions the model)
  sole_scr      NO user image; SCRAMBLED image injected inline       -> MUST collapse   (placebo valid; not text-only guessing)
  noimg         no image anywhere                                    -> blind floor
  two_real      user image + SAME image injected inline at p         -> ~= normal (redundant; 2-image path doesn't break)
  two_conflict  user image + DIFFERENT scene injected inline at p    -> acc DROP => 2nd copy IS attended in 2-image config

VERDICT:
  sole_real ~= normal >> sole_scr ~= noimg   => vLLM inline injection is a VALID vision channel (sweep is trustworthy).
  sole_real ~= noimg                          => BUG: inline image not conditioning in vLLM -> STOP, fix before sweep.
  two_conflict << normal                      => 2nd copy attended in 2-image config -> a V1-null in the sweep = "looked, didn't help".
  two_conflict ~= normal                      => 2nd copy IGNORED when a 1st copy exists -> sweep must REPLACE not ADD the image.

Guarded under __main__ (vLLM spawn). N easy items (default 10). Injection position p = 0.5 (mid), no prefix
(cleanest mechanism test = A0's inj_inline, reproduced in vLLM).
"""
import os, sys, re, json, time, random

N      = int(os.environ.get("N_ITEMS", "10"))
DSET   = os.environ.get("DSET", "full")
MAXDEP = int(os.environ.get("MAX_DEPTH", "8"))
MAXTOK = int(os.environ.get("MAX_TOK", "4096"))
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
V2     = f"{OUT}/v2_{DSET}_records.jsonl"
BOXED  = "\n\nPut your final answer in \\boxed{}."

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
    log("="*70, f"\nE3 A2 INSTRUMENT VERIFICATION (vLLM)  N={N}  DSET={DSET}")
    from PIL import Image
    from transformers import AutoProcessor
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    probe=proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":"x"}]}],
                                   tokenize=False, add_generation_prompt=False)
    VIS=re.search(r"<\|vision_start\|>.*?<\|vision_end\|>", probe, re.DOTALL).group(0)

    rows=[json.loads(l) for l in open(V2)]
    easy=[r for r in rows if r["depth"]<=MAXDEP and correct(r["full_text"], r["gt_norm"])][:N]
    log(f"selected {len(easy)} easy correct items")
    def img_of(r): return Image.open(f"{IMGDIR}/{r['image_filename']}").convert("RGB")

    # NO reasoning prefix: the injected image is the ONLY source of visual info, so sole_real correctness
    # can ONLY come from attending the injected image (a prefix generated-with-image would leak the scene
    # and make the test uninformative). This is A0's inj_inline reproduced in vLLM. Image sits at the start
    # of the assistant turn = genuinely inline (M-RoPE assigns it a mid-sequence position after the user turn).
    def tpl(with_img, q):
        content=([{"type":"image"}] if with_img else [])+[{"type":"text","text":q+BOXED}]
        return proc.apply_chat_template([{"role":"user","content":content}], tokenize=False, add_generation_prompt=True)

    jobs=[]
    for k,r in enumerate(easy):
        q=r["question"]; img=img_of(r); scr=scramble(img)
        other=img_of(easy[(k+1)%len(easy)])
        # (cond, prompt, images)  — injection at assistant start, no prefix leak
        jobs.append((r,"normal",       tpl(True,q),         [img]))
        jobs.append((r,"sole_real",    tpl(False,q)+VIS,    [img]))
        jobs.append((r,"sole_scr",     tpl(False,q)+VIS,    [scr]))
        jobs.append((r,"noimg",        tpl(False,q),        None))
        jobs.append((r,"two_real",     tpl(True,q)+VIS,     [img,img]))
        jobs.append((r,"two_conflict", tpl(True,q)+VIS,     [img,other]))
    log(f"total generations: {len(jobs)}")

    import torch, vllm
    from vllm import LLM, SamplingParams
    log(f"torch {torch.__version__}  vllm {vllm.__version__}")
    t0=time.time()
    llm=LLM(model=MODEL, dtype="bfloat16", max_model_len=20480, gpu_memory_utilization=0.90,
            limit_mm_per_prompt={"image":2}, trust_remote_code=False)
    log(f"engine up {time.time()-t0:.0f}s")
    sp=SamplingParams(temperature=0.0, max_tokens=MAXTOK)
    reqs=[({"prompt":p,"multi_modal_data":{"image":im}} if im else {"prompt":p}) for (_,_,p,im) in jobs]
    t0=time.time(); outs=llm.generate(reqs, sp); log(f"generated {len(outs)} in {time.time()-t0:.0f}s")

    from collections import defaultdict
    tally=defaultdict(lambda:[0,0]); ex=defaultdict(list)
    with open(f"{OUT}/phaseA2_verify.jsonl","w") as f:
        for (r,cond,_,_),o in zip(jobs,outs):
            txt=o.outputs[0].text; ans=canon(extract_boxed(txt)); ok=correct(txt,r["gt_norm"])
            tally[cond][0]+=ok; tally[cond][1]+=1
            if len(ex[cond])<3: ex[cond].append((canon(r["gt_norm"]),ans,ok))
            f.write(json.dumps(dict(qi=r["qi"],cond=cond,gt=canon(r["gt_norm"]),ans=ans,ok=ok))+"\n")

    log("="*70,"\nVERIFICATION SUMMARY (accuracy by condition)")
    for c in ["normal","sole_real","sole_scr","noimg","two_real","two_conflict"]:
        k,n=tally[c]; sample=" | ".join(f"gt={g!r} ans={a!r}{'✓' if o else '✗'}" for g,a,o in ex[c])
        log(f"  {c:13} {k}/{n} = {k/max(1,n):.2f}   e.g. {sample}")
    def acc(c): return tally[c][0]/max(1,tally[c][1])
    log("\nVERDICT:")
    log(f"  vLLM inline injection VALID if  sole_real({acc('sole_real'):.2f}) ~ normal({acc('normal'):.2f}) >> sole_scr({acc('sole_scr'):.2f}) ~ noimg({acc('noimg'):.2f})")
    log(f"  2nd-copy ATTENDED in 2-img cfg if  two_conflict({acc('two_conflict'):.2f}) << normal({acc('normal'):.2f})")
    log(f"    -> if sole_real ~ noimg: BUG (image not conditioning in vLLM) -> STOP, do not run sweep")
    log(f"    -> if two_conflict ~ normal: 2nd copy ignored when 1st exists -> sweep must REPLACE not ADD the image")
    log(f"saved -> {OUT}/phaseA2_verify.jsonl")


if __name__=="__main__":
    main()
