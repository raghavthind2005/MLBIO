#!/usr/bin/env python3
"""
Set 3 / Phase 2 — parity gate v3, LAYER B: causal SENSITIVITY (conflict test), vLLM, on robustly-correct items.
Run AFTER Layer A passes. n>=30. Pre-committed decision tree (record Part 4b).

At f0.50 on robustly-correct items:  V0 | V_text (correct scene) | V_text_wrong (diff scene, matched count)
                                     | V1 (correct image) | conflict_img (diff scene image).
Text arm "clearly live" = wrong-payload drop >= 0.20 AND right-payload drop <= 0.10 (paired vs V0; McNemar+bootstrap).
Decision (given Layer A PASS): drop>=0.20 -> gate PASS. drop<0.20 -> PASS ON DELIVERY + log
  "model resists conflicting mid-think text on correct items" (WEAK headwind for H1 only; NOT the Pool-S
  serialization-gap mechanism; do NOT tune the instrument). Guarded under __main__.
"""
import os, sys, io, json, time, base64
from collections import defaultdict
from common import answer_correct
from p2_common import render_scene_text, scramble
import stats

MAXTOK=int(os.environ.get("MAX_TOK","16384"))
MODEL="/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR="/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT="/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED="\n\nPut your final answer in \\boxed{}."; THINK_CLOSE=151668
def log(*a): print(*a, flush=True)

def main():
    log("="*70, "\nSet3 P2 PARITY GATE v3 — LAYER B (causal sensitivity / conflict)")
    from transformers import AutoProcessor
    from PIL import Image
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    import re
    VIS=re.search(r"<\|vision_start\|>.*?<\|vision_end\|>",
                  proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":"x"}]}],
                  tokenize=False, add_generation_prompt=False), re.DOTALL).group(0)
    orig={json.loads(l)["qi"]:json.loads(l) for l in open(f"{OUT}/set3_orig_records.jsonl")}
    gp=[json.loads(l)["qi"] for l in open(f"{OUT}/set3_gatepool.jsonl") if json.loads(l)["robust_correct"]==1]
    log(f"robustly-correct items: {len(gp)}")
    # matched-object-count pickers (from all orig)
    by_count=defaultdict(list)
    for qi,r in orig.items(): by_count[len(r["scene"]["objects"])].append(qi)
    for k in by_count: by_count[k].sort()
    def other(qi):
        n=len(orig[qi]["scene"]["objects"]); lst=by_count[n]
        return None if len(lst)<2 else lst[(lst.index(qi)+1)%len(lst)]
    def img_of(qi): return Image.open(f"{IMGDIR}/{orig[qi]['image_filename']}").convert("RGB")
    def user_tpl(qi): return proc.apply_chat_template([{"role":"user","content":[{"type":"image"},
                {"type":"text","text":orig[qi]['question']+BOXED}]}], tokenize=False, add_generation_prompt=True)
    def base50(qi):
        resp=list(orig[qi]["output_token_ids"]); ti=resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1
        return user_tpl(qi)+tok.decode(resp[:max(1,int(0.50*ti))])

    jobs=[]; used=[]
    for qi in gp:
        oi=other(qi)
        if oi is None: continue
        used.append(qi); img=img_of(qi); base=base50(qi); scene=orig[qi]["scene"]; oimg=img_of(oi)
        jobs.append((qi,"V0", base, [img]))
        jobs.append((qi,"V_text", base+"\n\n"+render_scene_text(scene)+"\n", [img]))
        jobs.append((qi,"V_text_wrong", base+"\n\n"+render_scene_text(orig[oi]["scene"])+"\n", [img]))
        jobs.append((qi,"V1", base+VIS, [img,img]))
        jobs.append((qi,"conflict_img", base+VIS, [img,oimg]))
    log(f"items used={len(used)}  jobs={len(jobs)}")

    import torch, vllm
    from vllm import LLM, SamplingParams
    t0=time.time()
    llm=LLM(model=MODEL,dtype="bfloat16",max_model_len=40960,gpu_memory_utilization=0.90,
            limit_mm_per_prompt={"image":2},trust_remote_code=False); log(f"engine up {time.time()-t0:.0f}s")
    outs=llm.generate([{"prompt":p,"multi_modal_data":{"image":im}} for (_,_,p,im) in jobs],
                      SamplingParams(temperature=0.0,max_tokens=MAXTOK))
    ok=defaultdict(dict)
    for (qi,c,_,_),o in zip(jobs,outs): ok[qi][c]=answer_correct(o.outputs[0].text, orig[qi]["gt_norm"])
    def vec(c): return [ok[qi][c] for qi in used]
    v0,vt,vtw,v1,vc = vec("V0"),vec("V_text"),vec("V_text_wrong"),vec("V1"),vec("conflict_img")
    def acc(v): return sum(v)/len(v)
    log("="*70, f"\naccuracy (n={len(used)}):  V0={acc(v0):.2f} V_text={acc(vt):.2f} V_text_wrong={acc(vtw):.2f} "
        f"V1={acc(v1):.2f} conflict_img={acc(vc):.2f}")
    # paired vs V0 (delta = acc(cond)-acc(V0); negative = drop)
    rt=stats.paired_compare(v0, vt, "V_text");  rtw=stats.paired_compare(v0, vtw, "V_text_wrong")
    ri=stats.paired_compare(v0, vc, "conflict_img")
    right_drop = acc(v0)-acc(vt); wrong_drop = acc(v0)-acc(vtw); img_drop = acc(v1)-acc(vc)
    log(f"TEXT arm:  right-drop(V0-V_text)={right_drop:+.2f}  wrong-drop(V0-V_text_wrong)={wrong_drop:+.2f}")
    log(f"  V_text vs V0:       Δ={rt['delta']:+.3f} CI={rt['ci']} McNemar p={rt['mcnemar_p']} (flips: cond_wins={rt['cond_wins']} v0_wins={rt['base_wins']})")
    log(f"  V_text_wrong vs V0: Δ={rtw['delta']:+.3f} CI={rtw['ci']} McNemar p={rtw['mcnemar_p']} (flips: cond_wins={rtw['cond_wins']} v0_wins={rtw['base_wins']})")
    log(f"IMAGE arm (corroborating): V1={acc(v1):.2f} conflict_img={acc(vc):.2f} drop={img_drop:+.2f} "
        f"(Δ vs V0: {ri['delta']:+.3f} CI={ri['ci']})")
    clearly_live = (wrong_drop>=0.20 and right_drop<=0.10)
    log("="*70, f"\nTEXT ARM clearly-live (wrong-drop>=0.20 AND right-drop<=0.10): {clearly_live}")
    if clearly_live:
        log("DECISION (given Layer A PASS): gate PASS.")
    else:
        log("DECISION (given Layer A PASS): PASS ON DELIVERY. LOG pre-outcome observation: "
            "'model resists conflicting mid-think text on correct items' — WEAK headwind for H1 only "
            "(not the Pool-S serialization-gap mechanism; instrument NOT to be tuned).")

if __name__=="__main__":
    main()
