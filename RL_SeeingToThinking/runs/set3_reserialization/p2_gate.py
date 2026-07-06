#!/usr/bin/env python3
"""
Set 3 / Phase 2 — HF-vs-vLLM PARITY GATE for the NEW Set-3 injection configs (blocking, directive §6, ≥14/16).
Set-2 validated only 2-IMAGE injection; Set 3 adds TEXT injection (V_self/V_text). This re-validates that
vLLM reproduces HF for text AND image payloads before the sweep is trusted.

METHOD (fast + strict): greedy is deterministic, so if the injection produces the SAME model state in HF and
vLLM, they emit the SAME token trajectory. We compare the FIRST N greedy tokens after injection (token-id
exact prefix), NOT the final boxed answer — this needs no long mid-think re-think (which made the answer-based
gate too slow), and an identical prefix is stronger evidence of faithfulness than a matching final answer.
Early divergence => vLLM mishandles the injected payload. HF and vLLM in ONE process on byte-identical inputs.
Guarded under __main__ (vLLM spawn).
"""
import os, sys, io, json, time, base64, gc, re
import torch
from common import canon_ans, extract_boxed
from p2_common import vself_text, render_scene_text, scramble

N_ITEMS = int(os.environ.get("N_ITEMS", "4"))
MAXNEW  = int(os.environ.get("MAX_NEW", "128"))
LOCK    = int(os.environ.get("LOCK", "64"))            # require identical trajectory for >= this many tokens
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED  = "\n\nPut your final answer in \\boxed{}."
THINK_CLOSE = 151668
def log(*a): print(*a, flush=True)

def main():
    log("="*70, "\nSet3 P2 PARITY GATE (HF vs vLLM, Set-3 text+image injection)")
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    VIS=re.search(r"<\|vision_start\|>.*?<\|vision_end\|>",
                  proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":"x"}]}],
                  tokenize=False, add_generation_prompt=False), re.DOTALL).group(0)

    orig={json.loads(l)["qi"]:json.loads(l) for l in open(f"{OUT}/set3_orig_records.jsonl")}
    vself={json.loads(l)["qi"]:json.loads(l) for l in open(f"{OUT}/set3_vself.jsonl")}
    pools=[json.loads(l) for l in open(f"{OUT}/set3_pools.jsonl")]
    poolS=[r["qi"] for r in pools if r["pool"]=="S"][:N_ITEMS]

    def img_of(qi): return Image.open(f"{IMGDIR}/{orig[qi]['image_filename']}").convert("RGB")
    def user_tpl(qi):
        return proc.apply_chat_template([{"role":"user","content":[{"type":"image"},
                {"type":"text","text":orig[qi]['question']+BOXED}]}], tokenize=False, add_generation_prompt=True)
    def prefix50(qi):
        resp=list(orig[qi]["output_token_ids"]); ti=resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1
        return tok.decode(resp[:max(1,int(0.50*ti))])

    # build identical cases (prompt string + image list) used verbatim by BOTH backends
    cases=[]
    for qi in poolS:
        img=img_of(qi); scene=orig[qi]["scene"]; base=user_tpl(qi)+prefix50(qi)
        cases.append((qi,"V1",     base+VIS,                                   [img,img]))            # image inject
        cases.append((qi,"V_scr",  base+VIS,                                   [img,scramble(img)])) # image inject (placebo)
        cases.append((qi,"V_self", base+"\n\n"+vself_text(vself[qi]["v_self_payload"])+"\n", [img])) # text inject
        cases.append((qi,"V_text", base+"\n\n"+render_scene_text(scene)+"\n",  [img]))               # text inject
    log(f"cases={len(cases)}  (image-inject: V1,V_scr; text-inject: V_self,V_text × {len(poolS)} Pool-S @ f0.50)")

    # ---- HF reference (sequential greedy) ----
    t0=time.time(); model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda").eval()
    log(f"HF loaded {time.time()-t0:.0f}s")
    hf=[]
    for qi,c,prompt,imgs in cases:
        enc=proc(text=[prompt],images=imgs,return_tensors="pt"); enc={k:v.to("cuda") for k,v in enc.items()}
        with torch.no_grad(): o=model.generate(**enc,max_new_tokens=MAXNEW,do_sample=False)
        hf.append(o[0][enc["input_ids"].shape[1]:].tolist())          # generated token ids
    log(f"HF done {time.time()-t0:.0f}s"); del model; gc.collect(); torch.cuda.empty_cache()

    # ---- vLLM (same inputs) ----
    import vllm
    from vllm import LLM, SamplingParams
    llm=LLM(model=MODEL,dtype="bfloat16",max_model_len=40960,gpu_memory_utilization=0.55,
            limit_mm_per_prompt={"image":2},trust_remote_code=False)
    sp=SamplingParams(temperature=0.0,max_tokens=MAXNEW)
    outs=llm.generate([{"prompt":p,"multi_modal_data":{"image":im}} for (_,_,p,im) in cases], sp)
    vl=[list(o.outputs[0].token_ids) for o in outs]

    def cprefix(a,b):
        n=0
        for x,y in zip(a,b):
            if x==y: n+=1
            else: break
        return n
    log("="*70, f"\nHF vs vLLM — first-{MAXNEW}-token greedy trajectory (identical prefix => engine-faithful injection):")
    faithful=0
    for (qi,c,_,_),h,v in zip(cases,hf,vl):
        cp=cprefix(h,v); m=min(len(h),len(v)); ok=cp>=min(LOCK,m)
        faithful+=ok
        log(f"  qi{qi} {c:7} common_prefix={cp}/{m}  {'ok' if ok else 'DIVERGES-EARLY'}")
    log(f"\nPARITY: {faithful}/{len(cases)} cases with identical HF/vLLM trajectory for >={LOCK} tokens  (require >=14/16)")
    log("  identical greedy trajectory => vLLM reproduces HF's post-injection state; early divergence => unfaithful injection.")

if __name__=="__main__":
    main()
