#!/usr/bin/env python3
"""
Set 3 / Phase 2 — parity DIAGNOSTIC: is the p2_gate token-divergence an INJECTION fault or generic HF-vs-vLLM
long-context numeric divergence? Control = V0 (NO injection at all), at increasing context length:
  noprefix (just the question) → V0@f0.25 → V0@f0.50 → V0@f0.75 (longer prefix each time).
If divergence GROWS with prefix length under NO injection, the p2_gate faults are generic numerics (kernels+bf16),
not injection faults → the token-trajectory metric is invalid for HF-vs-vLLM and must be replaced.
Prints HF-vs-vLLM common-prefix + HF top-2 logprob gap at divergence. Guarded under __main__.
"""
import os, re, json, time, gc
import torch
MODEL="/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR="/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT="/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED="\n\nPut your final answer in \\boxed{}."; THINK_CLOSE=151668; MAXNEW=128
def log(*a): print(*a, flush=True)

def main():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    orig={json.loads(l)["qi"]:json.loads(l) for l in open(f"{OUT}/set3_orig_records.jsonl")}
    pools=[json.loads(l) for l in open(f"{OUT}/set3_pools.jsonl")]
    poolS=[r["qi"] for r in pools if r["pool"]=="S"][:3]
    def img_of(qi): return Image.open(f"{IMGDIR}/{orig[qi]['image_filename']}").convert("RGB")
    def user_tpl(qi):
        return proc.apply_chat_template([{"role":"user","content":[{"type":"image"},
                {"type":"text","text":orig[qi]['question']+BOXED}]}], tokenize=False, add_generation_prompt=True)
    def prefixes(qi):
        resp=list(orig[qi]["output_token_ids"]); ti=resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1
        return {"noprefix":"","f0.25":tok.decode(resp[:max(1,int(0.25*ti))]),
                "f0.50":tok.decode(resp[:max(1,int(0.50*ti))]),"f0.75":tok.decode(resp[:max(1,int(0.75*ti))])}
    # V0-only cases (no injection), increasing context
    cases=[]
    for qi in poolS:
        px=prefixes(qi)
        for lab in ("noprefix","f0.25","f0.50","f0.75"):
            cases.append((qi,lab,user_tpl(qi)+px[lab],[img_of(qi)], len(tok(px[lab]).input_ids)))
    log(f"V0 control cases={len(cases)} (no injection; increasing prefix length)")

    t0=time.time(); model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda").eval()
    log(f"HF loaded {time.time()-t0:.0f}s")
    hf=[]; hf_gaps=[]
    for qi,lab,prompt,imgs,plen in cases:
        enc=proc(text=[prompt],images=imgs,return_tensors="pt"); enc={k:v.to("cuda") for k,v in enc.items()}
        with torch.no_grad():
            out=model.generate(**enc,max_new_tokens=MAXNEW,do_sample=False,output_scores=True,return_dict_in_generate=True)
        hf.append(out.sequences[0][enc["input_ids"].shape[1]:].tolist())
        g=[]
        for s in out.scores:
            lg=torch.log_softmax(s[0].float(),-1); t2=torch.topk(lg,2).values; g.append(float(t2[0]-t2[1]))
        hf_gaps.append(g)
    del model; gc.collect(); torch.cuda.empty_cache()

    from vllm import LLM, SamplingParams
    llm=LLM(model=MODEL,dtype="bfloat16",max_model_len=40960,gpu_memory_utilization=0.55,
            limit_mm_per_prompt={"image":1},trust_remote_code=False)
    outs=llm.generate([{"prompt":p,"multi_modal_data":{"image":im}} for (_,_,p,im,_) in cases],
                      SamplingParams(temperature=0.0,max_tokens=MAXNEW))
    vl=[list(o.outputs[0].token_ids) for o in outs]
    def cpfx(a,b):
        n=0
        for x,y in zip(a,b):
            if x==y:n+=1
            else:break
        return n
    log("="*70, "\nV0 (NO injection) HF-vs-vLLM trajectory by context length:")
    for (qi,lab,_,_,plen),h,v,g in zip(cases,hf,vl,hf_gaps):
        cp=cpfx(h,v); gap=g[cp] if cp<len(g) else 0.0
        log(f"  qi{qi} {lab:8} prefix_tok={plen:>5}  common_prefix={cp:>3}/{min(len(h),len(v))}  gap@div={gap:.3f}")
    log("\nread: if common_prefix SHRINKS as prefix_tok grows (under NO injection) => generic HF-vs-vLLM numeric")
    log("      divergence over long context; the p2_gate token-trajectory faults are an ARTIFACT, not injection faults.")

if __name__=="__main__":
    main()
