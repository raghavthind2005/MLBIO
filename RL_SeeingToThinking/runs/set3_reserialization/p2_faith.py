#!/usr/bin/env python3
"""
Set 3 / Phase 2 — WITHIN-vLLM injection-faithfulness gate (no HF → no HF-vs-vLLM numeric artifact).
Directly certifies what the sweep needs: does vLLM READ and USE the injected payload?

Causal design (Set-2 A2 conflict logic, in vLLM only): take items the model answers CORRECTLY, inject at f0.50:
  V0            no injection                         — control (should stay correct)
  V_text        CORRECT scene-graph text             — correct info (should stay correct)
  V_text_wrong  DIFFERENT scene-graph text           — if text is attended, answer MOVES off correct
  V1            CORRECT image (2nd copy)             — control
  conflict_img  DIFFERENT scene image (2nd copy)     — if image is attended, answer MOVES off correct
Faithful iff wrong payloads drop accuracy vs their right-payload counterpart (payload is causally used).
Greedy, max_tokens 16384. Modes: smoke (8) | full (default 30). Guarded under __main__.
"""
import os, sys, json, time, re
from collections import defaultdict
from common import answer_correct, extract_boxed, canon_ans
from p2_common import render_scene_text, scramble

MODE=(sys.argv[1] if len(sys.argv)>1 else "full").lower()
N=int(os.environ.get("N_ITEMS", "8" if MODE=="smoke" else "30"))
MAXTOK=int(os.environ.get("MAX_TOK","16384"))
MODEL="/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR="/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT="/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED="\n\nPut your final answer in \\boxed{}."; THINK_CLOSE=151668
def log(*a): print(*a, flush=True)

def main():
    log("="*70, f"\nSet3 P2 WITHIN-vLLM FAITHFULNESS  MODE={MODE}  N={N}")
    from transformers import AutoProcessor
    from PIL import Image
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    VIS=re.search(r"<\|vision_start\|>.*?<\|vision_end\|>",
                  proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":"x"}]}],
                  tokenize=False, add_generation_prompt=False), re.DOTALL).group(0)
    orig=[json.loads(l) for l in open(f"{OUT}/set3_orig_records.jsonl")]
    O={r["qi"]:r for r in orig}
    correct=[r for r in orig if r["correct"]==1 and r["depth"]>=10][:N]      # CORRECT items (clean causal signal)
    log(f"correct items: {len(correct)}")
    by_count=defaultdict(list)
    for r in orig: by_count[len(r["scene"]["objects"])].append(r["qi"])
    for k in by_count: by_count[k].sort()
    def wrong_scene(qi):
        n=len(O[qi]["scene"]["objects"]); lst=by_count[n]
        return None if len(lst)<2 else O[lst[(lst.index(qi)+1)%len(lst)]]["scene"]
    def other_img(qi):
        n=len(O[qi]["scene"]["objects"]); lst=by_count[n]
        return None if len(lst)<2 else lst[(lst.index(qi)+1)%len(lst)]
    def img_of(qi): return Image.open(f"{IMGDIR}/{O[qi]['image_filename']}").convert("RGB")
    def user_tpl(qi): return proc.apply_chat_template([{"role":"user","content":[{"type":"image"},
                {"type":"text","text":O[qi]['question']+BOXED}]}], tokenize=False, add_generation_prompt=True)
    def prefix50(qi):
        resp=list(O[qi]["output_token_ids"]); ti=resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1
        return tok.decode(resp[:max(1,int(0.50*ti))])

    jobs=[]
    for r in correct:
        qi=r["qi"]; img=img_of(qi); base=user_tpl(qi)+prefix50(qi); scene=r["scene"]
        ws=wrong_scene(qi); oi=other_img(qi)
        jobs.append((qi,"V0", base, [img]))
        jobs.append((qi,"V_text", base+"\n\n"+render_scene_text(scene)+"\n", [img]))
        if ws: jobs.append((qi,"V_text_wrong", base+"\n\n"+render_scene_text(ws)+"\n", [img]))
        jobs.append((qi,"V1", base+VIS, [img,img]))
        if oi: jobs.append((qi,"conflict_img", base+VIS, [img, img_of(oi)]))
    log(f"jobs={len(jobs)}")

    import torch, vllm
    from vllm import LLM, SamplingParams
    t0=time.time()
    llm=LLM(model=MODEL,dtype="bfloat16",max_model_len=40960,gpu_memory_utilization=0.90,
            limit_mm_per_prompt={"image":2},trust_remote_code=False)
    log(f"engine up {time.time()-t0:.0f}s")
    outs=llm.generate([{"prompt":p,"multi_modal_data":{"image":im}} for (_,_,p,im) in jobs],
                      SamplingParams(temperature=0.0,max_tokens=MAXTOK))
    res=defaultdict(list)
    for (qi,c,_,_),o in zip(jobs,outs):
        res[c].append(answer_correct(o.outputs[0].text, O[qi]["gt_norm"]))
    def acc(c): return (sum(res[c])/len(res[c]), len(res[c])) if res[c] else (float("nan"),0)
    log("="*70, "\naccuracy on CORRECT items after injection (should stay high for RIGHT payloads, DROP for WRONG):")
    for c in ("V0","V1","V_text","V_text_wrong","conflict_img"):
        a,n=acc(c); log(f"  {c:14} acc={a:.2f}  (n={n})")
    at,_=acc("V_text"); aw,_=acc("V_text_wrong"); a1,_=acc("V1"); ac,_=acc("conflict_img")
    log(f"\nTEXT attended  (V_text {at:.2f} -> V_text_wrong {aw:.2f}): drop={at-aw:+.2f}")
    log(f"IMAGE attended (V1 {a1:.2f} -> conflict_img {ac:.2f}): drop={a1-ac:+.2f}")
    log("read: a clear accuracy DROP under wrong payloads => vLLM causally reads/uses the injection => faithful.")

if __name__=="__main__":
    main()
