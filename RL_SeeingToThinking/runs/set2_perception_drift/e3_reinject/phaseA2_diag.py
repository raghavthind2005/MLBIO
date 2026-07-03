#!/usr/bin/env python3
"""
E3 / Method A — A2 INJECTION-PROCEDURE DIAGNOSTIC (HF, output_attentions). Prove the technique is correct.

Motivation (honest): A0 (HF) inj_inline = 5/5, but verify (vLLM) sole_real = 0.20 EMPTY — same construction,
opposite result. That means we do NOT yet know the injected image is being placed/positioned correctly.
`two_conflict` disrupting (0.40) could be a POSITIONALLY-CORRUPTED image (garbled M-RoPE) that disturbs but
never informs. This diagnostic reads it directly in HF:

  PART 1 (ATTENTION — the direct proof): build the two-image injection (user image + 2nd image spliced at p),
    forward with output_attentions, and measure attention mass from the pre-generation token onto
    image-block-1 (user) vs image-block-2 (injected). If block-2 gets ~0 attention => injection is NOT attended
    (broken). If comparable to block-1 => cleanly injected and looked at.

  PART 2 (CAUSAL leverage by position): at positions {0.25, 0.50, think}, inject NONE / SAME / DIFFERENT / SCRAMBLED
    and generate. If a DIFFERENT image changes the answer at a position => the injection has real leverage there.

  PART 3 (engine cross-check): reproduce A0 sole inj_inline IN HF (should be high) to confirm the HF path still
    works and isolate the vLLM discrepancy.

  PART 4 (framing): bare `<image>` vs a textual frame ("Let me re-examine the image: <image>") — does an
    in-distribution frame change whether it's attended/used.

If PART 1 shows block-2 is attended AND PART 2 shows a different image moves answers, the technique is CORRECT
and "same-image re-injection doesn't help" is a real result. If block-2 attention ~0, the procedure is broken
and we fix it (or run the sweep in HF) before believing anything.
"""
import os, sys, re, json, time, random
import torch

N      = int(os.environ.get("N_ITEMS", "5"))
DSET   = os.environ.get("DSET", "full")
MAXDEP = int(os.environ.get("MAX_DEPTH", "8"))
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
V2     = f"{OUT}/v2_{DSET}_records.jsonl"
BOXED  = "\n\nPut your final answer in \\boxed{}."
THINK_CLOSE=151668

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
    log("="*70, f"\nE3 A2 INJECTION DIAGNOSTIC (HF)  N={N}")
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    IMG_PAD=tok.convert_tokens_to_ids("<|image_pad|>")
    probe=proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":"x"}]}],
                                   tokenize=False, add_generation_prompt=False)
    VIS=re.search(r"<\|vision_start\|>.*?<\|vision_end\|>", probe, re.DOTALL).group(0)
    t0=time.time()
    model=AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                                      attn_implementation="eager").eval()
    log(f"model loaded {time.time()-t0:.0f}s (eager attn for output_attentions)  IMG_PAD={IMG_PAD}")

    rows=[json.loads(l) for l in open(V2)]
    easy=[r for r in rows if r["depth"]<=MAXDEP and correct(r["full_text"], r["gt_norm"])][:N]
    log(f"selected {len(easy)} easy correct items")
    def img_of(r): return Image.open(f"{IMGDIR}/{r['image_filename']}").convert("RGB")
    def prefix_at(r, frac):
        resp=list(r["output_token_ids"]); ti=resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1
        p = (ti+1) if frac=="think" else max(1,int(frac*ti))
        return tok.decode(resp[:p])
    def user_tpl(with_img, q):
        content=([{"type":"image"}] if with_img else [])+[{"type":"text","text":q+BOXED}]
        return proc.apply_chat_template([{"role":"user","content":content}], tokenize=False, add_generation_prompt=True)

    def encode(text, images):
        enc=proc(text=[text], images=images if images else None, return_tensors="pt")
        return {k:v.to("cuda") for k,v in enc.items()}
    def gen(text, images, maxnew=1024):
        enc=encode(text,images)
        with torch.no_grad():
            out=model.generate(**enc, max_new_tokens=maxnew, do_sample=False)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    # ---------- PART 3: HF sole inj_inline sanity (compare to vLLM 0.20) ----------
    log("-"*70, "\nPART 3 — HF sole inj_inline (image only via injection, no user image; A0 got 5/5):")
    k=0
    for r in easy:
        img=img_of(r); text=user_tpl(False,r["question"])+VIS+"\n"
        ans=canon(extract_boxed(gen(text,[img]))); ok=correct("\\boxed{%s}"%ans, r["gt_norm"]); k+=ok
        log(f"  qi={r['qi']} gt={canon(r['gt_norm'])!r} sole_inline_ans={ans!r} {'OK' if ok else 'x'}")
    log(f"  HF sole_inline acc = {k}/{len(easy)}  (if high => HF path fine, vLLM sole_real=0.20 is an ENGINE issue)")

    # ---------- PART 1: ATTENTION on the injected block (the direct proof) ----------
    log("-"*70, "\nPART 1 — attention mass onto image-block-1 (user) vs image-block-2 (INJECTED), at p=0.50:")
    shortest=sorted(easy, key=lambda r: len(r["output_token_ids"]))[:2]
    for r in shortest:
        img=img_of(r); q=r["question"]; prefix=prefix_at(r,0.50)
        text=user_tpl(True,q)+prefix+VIS          # user image + injected same image (bare)
        enc=encode(text,[img,img])
        ids=enc["input_ids"][0].tolist()
        spans=[]; i=0
        while i<len(ids):
            if ids[i]==IMG_PAD:
                j=i
                while j<len(ids) and ids[j]==IMG_PAD: j+=1
                spans.append((i,j)); i=j
            else: i+=1
        if len(spans)<2:
            log(f"  qi={r['qi']} FOUND {len(spans)} image blocks (expected 2) -> INJECTION MALFORMED"); continue
        with torch.no_grad():
            out=model(**enc, output_attentions=True)
        atts=out.attentions  # tuple[L] (b, heads, seq, seq)
        L=len(atts); qpos=enc["input_ids"].shape[1]-1
        rows_rep=[]
        for Li in (L//4, L//2, 3*L//4, L-1):
            a=atts[Li][0].float().mean(0)          # heads-avg -> seq,seq
            m1=a[qpos, spans[0][0]:spans[0][1]].sum().item()
            m2=a[qpos, spans[1][0]:spans[1][1]].sum().item()
            rows_rep.append((Li,m1,m2))
        log(f"  qi={r['qi']}  block1={spans[0]} block2={spans[1]}  (block sizes {spans[0][1]-spans[0][0]}, {spans[1][1]-spans[1][0]})")
        for Li,m1,m2 in rows_rep:
            log(f"     layer {Li:>2}: attn->img1={m1:.4f}  attn->img2(INJECTED)={m2:.4f}  ratio2/1={ (m2/m1 if m1>0 else float('nan')):.2f}")
    log("  read: attn->img2 ~ 0 => injected block NOT attended (procedure broken). Comparable to img1 => attended.")

    # ---------- PART 2: causal leverage by position (none/same/diff/scr) ----------
    log("-"*70, "\nPART 2 — does injecting a DIFFERENT image change the answer? (leverage by position)")
    for frac in (0.25, 0.50, "think"):
        log(f"  position p={frac}:")
        for k,r in enumerate(easy):
            img=img_of(r); q=r["question"]; prefix=prefix_at(r,frac); other=img_of(easy[(k+1)%len(easy)]); scr=scramble(img)
            maxnew=2048 if frac!="think" else 1024
            base=user_tpl(True,q)+prefix
            a_none=canon(extract_boxed(gen(base,[img],maxnew)))
            a_same=canon(extract_boxed(gen(base+VIS,[img,img],maxnew)))
            a_diff=canon(extract_boxed(gen(base+VIS,[img,other],maxnew)))
            a_scr =canon(extract_boxed(gen(base+VIS,[img,scr],maxnew)))
            gt=canon(r["gt_norm"])
            log(f"    qi={r['qi']} gt={gt!r:>8} | none={a_none!r:>8} same={a_same!r:>8} diff={a_diff!r:>8} scr={a_scr!r:>8}"
                f"  {'[diff MOVED]' if a_diff!=a_none else ''}")
    log("  read: diff != none at a position => injected image has causal leverage there (technique works).")
    log("="*70, "\nDIAGNOSTIC DONE")


if __name__=="__main__":
    main()
