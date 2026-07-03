#!/usr/bin/env python3
"""
E3 / Method A — A0 MECHANICS PILOT (smoke). Validate the re-injection instrument BEFORE any Delta-corr.

The whole causal claim rests on one assumption: when we re-present the original image mid-sequence,
the model actually ATTENDS to it (and M-RoPE is wired right). An off-the-shelf model saw images in
USER turns during training, so a mid-ASSISTANT inline image may be out-of-distribution and ignored.
This pilot answers, on a few EASY items the model normally gets right:

  Does image-supplied-ONLY-via-injection reproduce normal accuracy? (-> injection is a valid vision channel)
  Which PLACEMENT conditions the model: a new USER turn ("truncate-and-re-ask") vs INLINE in the assistant?
  Is the SCRAMBLED image a proper placebo (collapses to blind)? (-> V_scr is a valid control)

Conditions (greedy decode, so differences are the image channel, not sampling):
  normal        image in the user turn (reference upper bound)
  inj_user_real image ONLY via a follow-up USER turn (real image)
  inj_user_scr  same, but PATCH-SHUFFLED image (placebo)
  inj_inline    image spliced INLINE at the start of the assistant turn (real image)
  noimg         no image anywhere (blind floor)

READ: pass if  acc(normal) ~ acc(inj_user_real) >> acc(inj_user_scr) ~ acc(noimg).
      If inj_inline also ~ normal -> inline placement works (mid-reasoning injection viable).
      If inj_user_real ~ noimg -> the model IGNORES re-injected images off-the-shelf -> M1 needs rethink
      (report honestly; do not proceed to A1 on a dead instrument).

Pure HF transformers (custom interleaving is easier here than vLLM). Smoke only (N items small).
"""
import os, sys, re, io, json, time, random
from PIL import Image

DSET   = os.environ.get("DSET", "full")
N      = int(os.environ.get("N_ITEMS", "5"))
MAXDEPTH = int(os.environ.get("MAX_DEPTH", "6"))        # pick EASY items (normal answer correct)
MAXNEW = int(os.environ.get("MAX_NEW", "2048"))
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
    """patch-shuffle: destroy scene layout, keep token count + low-level color/texture stats."""
    w,h=img.size; tw,th=w//n,h//n
    tiles=[img.crop((c*tw,r*th,(c+1)*tw,(r+1)*th)) for r in range(n) for c in range(n)]
    rnd=random.Random(0); rnd.shuffle(tiles)
    out=Image.new("RGB",(tw*n,th*n))
    for i,t in enumerate(tiles):
        r,c=divmod(i,n); out.paste(t,(c*tw,r*th))
    return out


def main():
    log("="*70, f"\nE3 A0 MECHANICS PILOT  DSET={DSET}  N={N}  max_new={MAXNEW}")
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    t0=time.time(); model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda").eval()
    log(f"model loaded {time.time()-t0:.0f}s")

    # native un-expanded vision placeholder (processor expands <|image_pad|> to grid-many later)
    probe=proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":"x"}]}],
                                   tokenize=False, add_generation_prompt=False)
    m=re.search(r"<\|vision_start\|>.*?<\|vision_end\|>", probe, re.DOTALL)
    VIS=m.group(0); log(f"vision placeholder = {VIS!r}")

    # ---- pick EASY items the model normally gets right ----
    rows=[json.loads(l) for l in open(V2)]
    easy=[r for r in rows if r["depth"]<=MAXDEPTH and correct(r["full_text"], r["gt_norm"])][:N]
    if len(easy)<N: easy=[r for r in rows if correct(r["full_text"], r["gt_norm"])][:N]
    log(f"selected {len(easy)} easy correct items (depth<={MAXDEPTH})")

    def gen(text, images):
        enc=proc(text=[text], images=images, return_tensors="pt") if images else proc(text=[text], return_tensors="pt")
        enc={k:v.to("cuda") for k,v in enc.items()}
        with torch.no_grad():
            out=model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False, temperature=None, top_p=None, top_k=None)
        new=out[0][enc["input_ids"].shape[1]:]
        return tok.decode(new, skip_special_tokens=True)

    def build(cond, q, img, scr):
        if cond=="normal":
            msgs=[{"role":"user","content":[{"type":"image"},{"type":"text","text":q+BOXED}]}]
            return proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True), [img]
        if cond in ("inj_user_real","inj_user_scr"):
            use = scr if cond=="inj_user_scr" else img
            msgs=[{"role":"user","content":[{"type":"text","text":q+BOXED}]},
                  {"role":"assistant","content":[{"type":"text","text":"I need to look at the image."}]},
                  {"role":"user","content":[{"type":"image"},{"type":"text","text":"Here is the image. Give the final answer."+BOXED}]}]
            return proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True), [use]
        if cond=="inj_inline":
            base=proc.apply_chat_template([{"role":"user","content":[{"type":"text","text":q+BOXED}]}],
                                          tokenize=False, add_generation_prompt=True)
            return base+VIS+"\n", [img]
        if cond=="noimg":
            msgs=[{"role":"user","content":[{"type":"text","text":q+BOXED}]}]
            return proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True), None
        raise ValueError(cond)

    CONDS=["normal","inj_user_real","inj_user_scr","inj_inline","noimg"]
    tally={c:[0,0] for c in CONDS}   # [correct, n]
    fout=open(f"{OUT}/phaseA0_mechanics.jsonl","w")
    for r in easy:
        q=r["question"]; gt=r["gt_norm"]
        img=Image.open(f"{IMGDIR}/{r['image_filename']}").convert("RGB"); scr=scramble(img)
        log("-"*70, f"\nqi={r['qi']} d={r['depth']} gt={canon(gt)!r}  Q={q[:70]}...")
        rec={"qi":r["qi"],"gt":canon(gt),"depth":r["depth"]}
        for c in CONDS:
            text,imgs=build(c,q,img,scr)
            resp=gen(text,imgs); ok=correct(resp,gt); box=canon(extract_boxed(resp))
            tally[c][0]+=ok; tally[c][1]+=1; rec[c]={"ok":ok,"ans":box}
            log(f"  {c:14} -> ans={box!r:>10}  {'OK' if ok else 'x'}")
        fout.write(json.dumps(rec)+"\n")
    fout.close()

    log("="*70, "\nA0 SUMMARY (accuracy by condition)")
    for c in CONDS:
        k,n=tally[c]; log(f"  {c:14} {k}/{n} = {k/max(1,n):.2f}")
    log("\nPASS if normal ~ inj_user_real >> inj_user_scr ~ noimg  (injection is a real, content-bearing vision channel)")
    log("     inj_inline ~ normal => inline mid-assistant injection works; else use user-turn 'truncate-and-re-ask' placement")
    log("     inj_user_real ~ noimg => model IGNORES re-injected images off-the-shelf => STOP, M1 needs rethink")
    log(f"saved -> {OUT}/phaseA0_mechanics.jsonl")


if __name__=="__main__":
    main()
