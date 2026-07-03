#!/usr/bin/env python3
"""
E3 / Method A — INSTRUMENTED HF SWEEP (main pass). Validated injection technique (A2 diag: injected block
is attended; vLLM was void -> HF only). Greedy, deterministic Delta-corr.

Per RIPE item x position p:
  SIGNAL FORWARD (original chain to p, SDPA): capture per-LAYER  MLP-write norm (PRIMARY, Set-1 60/30),
    attn-write norm, residual norm & cross-layer delta, logit-lens belief margin P(gt)-P(wrong) & entropy,
    cosine(hidden_p, image-token subspace); + scalar next-token entropy; + item-level meta. Saved for A3.
  GENERATION (5 conditions): V0 null | V1 original image | V_scr scrambled placebo |
    V_viz visual oracle (zoom-to-objects crop, upscaled) | V_text oracle (GT scene graph as text).
  Delta-corr(cond,p) = corr(cond) - corr(V0). corr(V_viz/V_text) tells us if a BETTER representation fixes RIPE.

NB attention-to-IMAGE mass is captured in the companion `phaseA2_attn.py` (eager, length-guarded) to avoid OOM.
Modes (argv[1]): smoke (2 RIPE, prints signals + validates oracles) | full. Guarded under __main__.
"""
import os, sys, re, json, time, random, math
import torch

MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
DSETS  = os.environ.get("DSETS", "full,hard").split(",")
N_RIPE = int(os.environ.get("N_RIPE", "0"))
MAXNEW = int(os.environ.get("MAX_NEW", "3072"))
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED  = "\n\nPut your final answer in \\boxed{}."
THINK_CLOSE = 151668
POSFRACS = [0.50, 0.75]                                   # + think + ans (leverage was late in the diagnostic)

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
def viz_oracle(img, scene, margin=60, up=2):
    """zoom to the bounding box of ALL objects (pixel_coords) + upscale. Robust, no program introspection."""
    W,H=img.size
    xs=[o["pixel_coords"][0] for o in scene["objects"]]; ys=[o["pixel_coords"][1] for o in scene["objects"]]
    x0=max(0,int(min(xs)-margin)); x1=min(W,int(max(xs)+margin))
    y0=max(0,int(min(ys)-margin)); y1=min(H,int(max(ys)+margin))
    if x1<=x0 or y1<=y0: return img.resize((W*up,H*up)), (0,0,W,H)   # degenerate -> whole image upscaled
    crop=img.crop((x0,y0,x1,y1))
    return crop.resize((crop.width*up, crop.height*up)), (x0,y0,x1,y1)
def text_oracle(scene):
    parts=[f"{o['size']} {o['color']} {o['material']} {o['shape']}" for o in scene["objects"]]
    return "[Reference — the image contains exactly these objects: " + "; ".join(parts) + ".]"


def main():
    log("="*70, f"\nE3 INSTRUMENTED HF SWEEP  MODE={MODE}  DSETS={DSETS}  pos={POSFRACS}+think+ans")
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    IMG_PAD=tok.convert_tokens_to_ids("<|image_pad|>")
    probe=proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":"x"}]}],
                                   tokenize=False, add_generation_prompt=False)
    VIS=re.search(r"<\|vision_start\|>.*?<\|vision_end\|>", probe, re.DOTALL).group(0)
    t0=time.time()
    model=AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    log(f"model loaded {time.time()-t0:.0f}s (sdpa)  IMG_PAD={IMG_PAD}")

    # locate LM stack (same traversal as phaseB1)
    lm=None
    for path in ("model.language_model","language_model.model","model.model.language_model"):
        obj=model
        try:
            for p in path.split("."): obj=getattr(obj,p)
            if hasattr(obj,"layers") and hasattr(obj,"norm"): lm=obj; break
        except AttributeError: continue
    layers,final_norm,lm_head=lm.layers,lm.norm,model.get_output_embeddings()
    NL=len(layers); log(f"LM layers={NL}")

    # answer-vocab belief readout (logit-lens over CLEVR answer tokens)
    id2val={}
    for v in VALUES+list(SYN.keys()):
        for variant in (v," "+v):
            ids=tok(variant,add_special_tokens=False).input_ids
            if ids: id2val.setdefault(ids[0], canon(v))
    ans_ids=sorted(id2val.keys())
    ans_ids_t=torch.tensor(ans_ids,device="cuda")
    def belief(logits_vec, gt, wrong):
        p=torch.softmax(logits_vec[ans_ids_t].float(),0).cpu()
        d={}
        for i,tid in enumerate(ans_ids): d[id2val[tid]]=d.get(id2val[tid],0.0)+float(p[i])
        tot=sum(d.values())
        ent=-sum((x/tot)*math.log(x/tot) for x in d.values() if x>0) if tot>0 else 0.0
        return round(d.get(gt,0.0)-d.get(wrong,0.0),4), round(ent,4)

    # ---- per-layer write-norm hooks (PRIMARY signal) ----
    CAP={}; FLAG=[False]
    def mk(idx,kind):
        def hook(m,inp,out):
            if not FLAG[0]: return
            t=out[0] if isinstance(out,tuple) else out
            CAP.setdefault(idx,{})[kind]=float(t[0,-1].float().norm().item())
        return hook
    for i,ly in enumerate(layers):
        ly.mlp.register_forward_hook(mk(i,"mlp")); ly.self_attn.register_forward_hook(mk(i,"attn"))

    # ---- pool RIPE ----
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
        resp=list(V2[qi]["output_token_ids"])
        ti=resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1
        model_ans=canon(extract_boxed(V2[qi]["full_text"])); ai=len(resp)-1
        for i in range(len(resp)-1,-1,-1):
            dv=canon(tok.decode([resp[i]]))
            if dv and (dv==model_ans or dv in model_ans.split()): ai=i; break
        pos=[(f"f{f:.2f}",max(1,int(f*ti))) for f in POSFRACS]+[("think",min(len(resp),ti+1)),("ans",max(1,ai))]
        return resp,pos,model_ans

    def encode(text, images):
        enc=proc(text=[text], images=images, return_tensors="pt")
        return {k:v.to("cuda") for k,v in enc.items()}
    def img_span(input_ids):
        ids=input_ids[0].tolist(); i=0
        while i<len(ids):
            if ids[i]==IMG_PAD:
                j=i
                while j<len(ids) and ids[j]==IMG_PAD: j+=1
                return (i,j)
            i+=1
        return None

    def signal_forward(qi, prefix_text, gt, wrong):
        enc=encode(user_tpl(qi)+prefix_text, [img_of(qi)])
        span=img_span(enc["input_ids"])
        CAP.clear(); FLAG[0]=True
        with torch.no_grad(): out=model(**enc, output_hidden_states=True)
        FLAG[0]=False
        hs=out.hidden_states                                  # (NL+1) each [1,seq,dim]
        sig={k:[] for k in ("mlp","attn","resid","resid_delta","bmargin","bent","cos_img")}
        for L in range(NL):
            h=hs[L+1][0,-1]
            sig["resid"].append(round(float(h.float().norm()),3))
            sig["resid_delta"].append(round(float((hs[L+1][0,-1]-hs[L][0,-1]).float().norm()),3))
            sig["mlp"].append(round(CAP.get(L,{}).get("mlp",float("nan")),3))
            sig["attn"].append(round(CAP.get(L,{}).get("attn",float("nan")),3))
            ll=lm_head(final_norm(h.unsqueeze(0).to(lm_head.weight.dtype)))[0]
            bm,be=belief(ll,gt,wrong); sig["bmargin"].append(bm); sig["bent"].append(be)
            if span:
                ir=hs[L+1][0,span[0]:span[1]].mean(0)
                sig["cos_img"].append(round(float(torch.cosine_similarity(h.float(),ir.float(),dim=0)),4))
            else: sig["cos_img"].append(float("nan"))
        fp=torch.softmax(out.logits[0,-1].float(),-1)
        sig["next_entropy"]=round(float(-(fp*torch.log(fp+1e-9)).sum()),4)
        sig["img_span_len"]= (span[1]-span[0]) if span else 0
        return sig

    def gen(text, images):
        enc=encode(text,images)
        with torch.no_grad():
            o=model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False)
        return tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    def build(qi, prefix_text, cond, img, scene):
        base=user_tpl(qi)+prefix_text
        if cond=="V0":    return base, [img]
        if cond=="V1":    return base+VIS, [img,img]
        if cond=="V_scr": return base+VIS, [img, scramble(img)]
        if cond=="V_viz": return base+VIS, [img, viz_oracle(img,scene)[0]]
        if cond=="V_text":return base+"\n\n"+text_oracle(scene)+"\n", [img]

    CONDS=["V0","V1","V_scr","V_viz","V_text"]
    recs=[]; fout=open(f"{OUT}/phaseA2_hf_{MODE}.jsonl","w")
    for qi in ripe:
        r=V2[qi]; scene=r["scene"]; img=img_of(qi); gt=canon(r["gt_norm"])
        resp,pos,model_ans=positions(qi); wrong=model_ans
        if MODE=="smoke":
            crop,bbox=viz_oracle(img,scene)
            log("-"*70, f"\nqi={qi} d={r['depth']} gt={gt!r} wrong={wrong!r}  #obj={len(scene['objects'])}")
            log(f"  viz_oracle bbox={bbox} img={img.size} crop={crop.size}   (bbox must lie within img & cover objects)")
            log(f"  text_oracle = {text_oracle(scene)[:120]}...")
        for plabel,ptok in pos:
            prefix_text=tok.decode(resp[:ptok])
            sig=signal_forward(qi, prefix_text, gt, wrong)
            outcomes={}
            for c in CONDS:
                text,imgs=build(qi,prefix_text,c,img,scene)
                resp_c=gen(text,imgs); outcomes[c]=dict(ans=canon(extract_boxed(resp_c)), ok=correct(resp_c,gt))
            rec=dict(qi=qi, pos=plabel, ptok=ptok, gt=gt, wrong=wrong, depth=r["depth"],
                     chain_len=len(resp), nobj=len(scene["objects"]), sig=sig, out=outcomes)
            recs.append(rec); fout.write(json.dumps(rec)+"\n")
            if MODE=="smoke":
                log(f"  [{plabel:6}] " + "  ".join(f"{c}={outcomes[c]['ans']!r}{'✓' if outcomes[c]['ok'] else '✗'}" for c in CONDS))
                log(f"     sig L0/{NL//2}/{NL-1}: mlp={sig['mlp'][0]:.1f}/{sig['mlp'][NL//2]:.1f}/{sig['mlp'][NL-1]:.1f} "
                    f"attn={sig['attn'][0]:.1f}/{sig['attn'][NL//2]:.1f}/{sig['attn'][NL-1]:.1f} "
                    f"bmargin(mid)={sig['bmargin'][NL//2]:+.3f} next_ent={sig['next_entropy']:.2f} cos_img(mid)={sig['cos_img'][NL//2]:+.3f}")
    fout.close()

    # ---- Delta-corr summary ----
    def rate(pos,c):
        s=[r for r in recs if r["pos"]==pos];
        return (sum(x["out"][c]["ok"] for x in s)/len(s), len(s)) if s else (0.0,0)
    log("="*70, "\nRIPE correction by position (net of V0):")
    log(f"  {'pos':7} {'V0':>6} {'V1':>6} {'V_scr':>6} {'V_viz':>6} {'V_text':>6}  | dViz  dText  (dText=text_oracle-V0)")
    for p in [f"f{f:.2f}" for f in POSFRACS]+["think","ans"]:
        v0,n=rate(p,"V0")
        r={c:rate(p,c)[0] for c in CONDS}
        log(f"  {p:7} {r['V0']:>6.2f} {r['V1']:>6.2f} {r['V_scr']:>6.2f} {r['V_viz']:>6.2f} {r['V_text']:>6.2f}  | "
            f"{r['V_viz']-v0:>+5.2f} {r['V_text']-v0:>+5.2f}  (n={n})")
    log("\nread: V_text≫V0 => perfect info fixes RIPE (not H4). V_viz>V1 => better VISION helps (H3). V1≈V0 => same image redundant.")
    log(f"saved -> {OUT}/phaseA2_hf_{MODE}.jsonl  (per-layer signals in 'sig' for A3 trigger-signature fit)")


if __name__=="__main__":
    main()
