#!/usr/bin/env python3
"""
Set 3 / Phase 2 — HF-vs-vLLM PARITY GATE (blocking, directive §6; supervisor-approved trajectory metric + riders).

Certifies vLLM reproduces HF's POST-INJECTION state token-exactly. Greedy is deterministic, so identical model
state => identical token trajectory. We compare the first-MAXNEW greedy tokens (token-id exact prefix), which
needs no long mid-think re-think.

RIDERS (supervisor, binding):
 R1. At the first-divergence token, log HF's top-2 logprob gap. gap<0.10 = benign near-tie drift (counts against
     the budget only); gap>=0.10 = STATE FAULT => gate FAILS regardless of count.
 R2. Cases span payload-modality × position: >=8 TEXT (V_self/V_text) and >=8 IMAGE (V1/V_scr) over f0.25/f0.50/f0.75.
     (Here 18 = 3 items × 3 positions × {1 text, 1 image} = 9 text + 9 image.)
 R3. Chain of trust logged in the experiment record (Set-2 A0/A2 validated HF injection behaviorally; this gate
     certifies vLLM == HF post-injection). Rationale for dropping final-answer agreement recorded as pre-outcome.

PASS = zero faults AND full-match (>= LOCK tokens) on >= (n-2) of the cases. Guarded under __main__ (vLLM spawn).
"""
import os, sys, io, json, time, base64, gc, re
import torch
from common import extract_boxed
from p2_common import vself_text, render_scene_text, scramble

N_ITEMS = int(os.environ.get("N_ITEMS", "3"))
MAXNEW  = int(os.environ.get("MAX_NEW", "128"))
LOCK    = int(os.environ.get("LOCK", "64"))
GAP_FAULT = 0.10
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED  = "\n\nPut your final answer in \\boxed{}."
THINK_CLOSE = 151668
def log(*a): print(*a, flush=True)

def main():
    log("="*70, "\nSet3 P2 PARITY GATE (HF vs vLLM; trajectory metric + riders R1/R2/R3)")
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
    def positions_all(qi):
        resp=list(orig[qi]["output_token_ids"]); ti=resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1
        return resp, [(f"f{f:.2f}", max(1,int(f*ti))) for f in (0.25,0.50,0.75)]

    # R2: balanced cases — per (item,position) one TEXT + one IMAGE payload, cycling payload subtype
    text_cycle=["V_self","V_text"]; img_cycle=["V1","V_scr"]; cases=[]; k=0
    for qi in poolS:
        img=img_of(qi); scene=orig[qi]["scene"]; resp,posl=positions_all(qi)
        for plabel,ptok in posl:
            base=user_tpl(qi)+tok.decode(resp[:ptok])
            tc=text_cycle[k%2]; ic=img_cycle[k%2]; k+=1
            tprompt = base+"\n\n"+(vself_text(vself[qi]["v_self_payload"]) if tc=="V_self" else render_scene_text(scene))+"\n"
            cases.append((qi,plabel,tc,tprompt,[img]))
            cases.append((qi,plabel,ic,base+VIS,[img, (img if ic=="V1" else scramble(img))]))
    nt=sum(1 for c in cases if c[2] in("V_self","V_text")); ni=len(cases)-nt
    log(f"cases={len(cases)}  TEXT={nt}  IMAGE={ni}  over f0.25/f0.50/f0.75  (R2: need >=8 each)")

    # ---- HF reference: token ids + per-step top-2 logprob gap ----
    t0=time.time(); model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda").eval()
    log(f"HF loaded {time.time()-t0:.0f}s")
    hf=[]; hf_gaps=[]
    for qi,pl,c,prompt,imgs in cases:
        enc=proc(text=[prompt],images=imgs,return_tensors="pt"); enc={k:v.to("cuda") for k,v in enc.items()}
        with torch.no_grad():
            out=model.generate(**enc,max_new_tokens=MAXNEW,do_sample=False,output_scores=True,return_dict_in_generate=True)
        hf.append(out.sequences[0][enc["input_ids"].shape[1]:].tolist())
        gaps=[]
        for s in out.scores:
            lg=torch.log_softmax(s[0].float(),-1); t2=torch.topk(lg,2).values; gaps.append(float(t2[0]-t2[1]))
        hf_gaps.append(gaps)
    log(f"HF done {time.time()-t0:.0f}s"); del model; gc.collect(); torch.cuda.empty_cache()

    # ---- vLLM (same inputs) ----
    import vllm
    from vllm import LLM, SamplingParams
    llm=LLM(model=MODEL,dtype="bfloat16",max_model_len=40960,gpu_memory_utilization=0.55,
            limit_mm_per_prompt={"image":2},trust_remote_code=False)
    outs=llm.generate([{"prompt":p,"multi_modal_data":{"image":im}} for (_,_,_,p,im) in cases],
                      SamplingParams(temperature=0.0,max_tokens=MAXNEW))
    vl=[list(o.outputs[0].token_ids) for o in outs]

    def cprefix(a,b):
        n=0
        for x,y in zip(a,b):
            if x==y: n+=1
            else: break
        return n
    log("="*70, f"\nHF vs vLLM first-{MAXNEW}-token trajectory:")
    passed=0; benign=[]; faults=[]
    for (qi,pl,c,_,_),h,v,gaps in zip(cases,hf,vl,hf_gaps):
        cp=cprefix(h,v); m=min(len(h),len(v))
        if cp>=min(LOCK,m):
            passed+=1; log(f"  qi{qi} {pl} {c:7} common_prefix={cp}/{m}  ok")
        else:
            gap=gaps[cp] if cp<len(gaps) else 0.0
            if gap>=GAP_FAULT: faults.append((qi,c,pl,cp,round(gap,3))); tag=f"FAULT (HF top2-gap={gap:.3f}>=0.10)"
            else: benign.append((qi,c,pl,cp,round(gap,3))); tag=f"benign (HF top2-gap={gap:.3f}<0.10)"
            log(f"  qi{qi} {pl} {c:7} common_prefix={cp}/{m}  DIVERGES@{cp}  {tag}")
    gate_pass = (len(faults)==0) and (passed >= len(cases)-2)
    log("="*70, f"\nfull-match {passed}/{len(cases)}   benign-divergence {len(benign)}   FAULTS {len(faults)}")
    if faults: log(f"  FAULTS (state faults — DEBUG before any sweep): {faults}")
    log(f"GATE {'PASS' if gate_pass else 'FAIL'}  (R1: 0 faults; full-match >= {len(cases)-2}/{len(cases)})")

if __name__=="__main__":
    main()
