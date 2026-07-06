#!/usr/bin/env python3
"""
Set 3 / Phase 2 — the decisive SWEEP (vLLM). 9 conditions on Pool-S AND Pool-P (V_viz2 dropped, record Part 3).
Greedy (temp 0), max_tokens 16384. Outcomes only; paired stats + strata are in p2_analyze.py (via stats.py).

POSITION-DEPENDENT (8), injected at p ∈ {f0.25,f0.50,f0.75} into the prefix chain[:p]:
  V0            none (greedy re-decode)                       — noise floor
  V1            original image (2nd copy)                     — pixel control
  V_scr         patch-shuffled image                          — pixel placebo
  V_self        model's own majority enumeration (set3_vself) — THE METHOD
  V_text        GT scene-graph text                           — oracle ceiling
  V_text_wrong  different-scene graph, matched object-count   — CONTENT placebo (A2)
  V_scaffold    original image + "Let me re-examine the image carefully:" cue — cue-vs-payload
  V_restart     keep prefix + "restart & solve from scratch" instruction + re-presented image — prefix-commitment probe
POSITION-INDEPENDENT (1), once per item:
  V_self_pre    enumeration placed in the USER turn BEFORE reasoning (fresh generation) — timing (A4)

INTERPRETATIONS FLAGGED (confirm on smoke): V_restart keeps prefix[:p] then instructs a from-scratch re-solve
with the image re-presented (tests whether the model is stuck on its committed prefix). V_self_pre is a fresh
generation with the enumeration in the user turn (no prefix), compared against mid-think V_self.

Modes (argv[1]): smoke (3 Pool-S + 2 Pool-P) | full. Guarded under __main__ (vLLM spawn).
"""
import os, sys, io, json, time, base64, re
from collections import defaultdict
from common import canon_ans, extract_boxed, answer_correct
from p2_common import vself_text, render_scene_text, scramble

MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
MAXTOK = int(os.environ.get("MAX_TOK", "16384"))
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED  = "\n\nPut your final answer in \\boxed{}."
THINK_CLOSE = 151668
POSFRACS = [0.25, 0.50, 0.75]
POS_CONDS = ["V0","V1","V_scr","V_self","V_text","V_text_wrong","V_scaffold","V_restart"]
def log(*a): print(*a, flush=True)

def main():
    log("="*70, f"\nSet3 P2 SWEEP  MODE={MODE}  pos={POSFRACS}  conds={POS_CONDS}+V_self_pre  MAXTOK={MAXTOK}")
    from transformers import AutoProcessor
    from PIL import Image
    proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer
    VIS=re.search(r"<\|vision_start\|>.*?<\|vision_end\|>",
                  proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":"x"}]}],
                  tokenize=False, add_generation_prompt=False), re.DOTALL).group(0)

    orig={json.loads(l)["qi"]:json.loads(l) for l in open(f"{OUT}/set3_orig_records.jsonl")}
    vself={json.loads(l)["qi"]:json.loads(l) for l in open(f"{OUT}/set3_vself.jsonl")}
    pools=[json.loads(l) for l in open(f"{OUT}/set3_pools.jsonl")]
    poolmap={r["qi"]:r["pool"] for r in pools}
    itemsS=[r["qi"] for r in pools if r["pool"]=="S"]; itemsP=[r["qi"] for r in pools if r["pool"]=="P"]
    if MODE=="smoke": itemsS, itemsP = itemsS[:3], itemsP[:2]
    items=itemsS+itemsP
    log(f"items: Pool-S={len(itemsS)} Pool-P={len(itemsP)}")

    # matched-object-count wrong-scene picker (deterministic; from all 3000 orig for options)
    by_count=defaultdict(list)
    for qi,r in orig.items(): by_count[len(r["scene"]["objects"])].append(qi)
    for k in by_count: by_count[k].sort()
    def wrong_scene(qi):
        n=len(orig[qi]["scene"]["objects"]); lst=by_count[n]
        if len(lst)<2: return None
        return orig[lst[(lst.index(qi)+1)%len(lst)]]["scene"]

    def img_of(qi): return Image.open(f"{IMGDIR}/{orig[qi]['image_filename']}").convert("RGB")
    def user_tpl(qi, extra_text=""):
        q=orig[qi]["question"]+extra_text+BOXED
        return proc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":q}]}],
                                        tokenize=False, add_generation_prompt=True)
    def positions(qi):
        resp=list(orig[qi]["output_token_ids"]); ti=resp.index(THINK_CLOSE) if THINK_CLOSE in resp else len(resp)-1
        return resp, [(f"f{f:.2f}", max(1,int(f*ti))) for f in POSFRACS]

    def build_pos(qi, base, cond, img):
        scene=orig[qi]["scene"]
        if cond=="V0":           return base, [img]
        if cond=="V1":           return base+VIS, [img,img]
        if cond=="V_scr":        return base+VIS, [img, scramble(img)]
        if cond=="V_self":       return base+"\n\n"+vself_text(vself[qi]["v_self_payload"])+"\n", [img]
        if cond=="V_text":       return base+"\n\n"+render_scene_text(scene)+"\n", [img]
        if cond=="V_text_wrong":
            ws=wrong_scene(qi)
            return (None if ws is None else (base+"\n\n"+render_scene_text(ws)+"\n", [img]))
        if cond=="V_scaffold":   return base+"\n\nLet me re-examine the image carefully:\n"+VIS, [img,img]
        if cond=="V_restart":    return base+"\n\nLet me disregard my reasoning so far and solve this again from scratch, looking carefully at the image:\n"+VIS, [img,img]

    # ---- build jobs ----
    jobs=[]
    for qi in items:
        img=img_of(qi); resp,pos=positions(qi)
        for plabel,ptok in pos:
            base=user_tpl(qi)+tok.decode(resp[:ptok])
            for c in POS_CONDS:
                built=build_pos(qi, base, c, img)
                if built is None:
                    log(f"  skip V_text_wrong qi={qi} (no matched-count scene)"); continue
                prompt,imgs=built
                jobs.append(dict(qi=qi,pool=poolmap[qi],pos=plabel,cond=c,prompt=prompt,images=imgs,gt=orig[qi]["gt_norm"]))
        # V_self_pre — enumeration in the user turn, fresh generation
        pre_prompt=user_tpl(qi, "\n\n"+vself_text(vself[qi]["v_self_payload"]))
        jobs.append(dict(qi=qi,pool=poolmap[qi],pos="pre",cond="V_self_pre",prompt=pre_prompt,images=[img],gt=orig[qi]["gt_norm"]))
    log(f"jobs={len(jobs)}")

    import torch, vllm
    from vllm import LLM, SamplingParams
    log(f"torch {torch.__version__} vllm {vllm.__version__}")
    t0=time.time()
    llm=LLM(model=MODEL,dtype="bfloat16",max_model_len=40960,gpu_memory_utilization=0.90,
            limit_mm_per_prompt={"image":2},trust_remote_code=False)
    log(f"engine up {time.time()-t0:.0f}s")
    sp=SamplingParams(temperature=0.0,max_tokens=MAXTOK)
    reqs=[{"prompt":j["prompt"],"multi_modal_data":{"image":j["images"]}} for j in jobs]
    t0=time.time(); outs=llm.generate(reqs,sp); log(f"generated {len(outs)} in {time.time()-t0:.0f}s")

    trunc=0
    with open(f"{OUT}/set3_p2sweep_{MODE}.jsonl","w") as f:
        for j,o in zip(jobs,outs):
            txt=o.outputs[0].text; ntok=len(o.outputs[0].token_ids); hit=int(ntok>=MAXTOK-2); trunc+=hit
            f.write(json.dumps(dict(qi=j["qi"],pool=j["pool"],pos=j["pos"],cond=j["cond"],gt=j["gt"],
                    ans=canon_ans(extract_boxed(txt)), ok=answer_correct(txt,j["gt"]), tok=ntok, trunc=hit))+"\n")
    log(f"saved -> {OUT}/set3_p2sweep_{MODE}.jsonl   truncated(hit MAXTOK): {trunc}/{len(jobs)}")

    # quick descriptive (NOT the analysis — that's p2_analyze.py with paired stats)
    recs=[json.loads(l) for l in open(f"{OUT}/set3_p2sweep_{MODE}.jsonl")]
    for pool in ("S","P"):
        log(f"\nPool-{pool} corrected-fraction (descriptive; paired stats in p2_analyze):")
        for pos in [f"f{f:.2f}" for f in POSFRACS]:
            row=" ".join(f"{c}={sum(r['ok'] for r in recs if r['pool']==pool and r['pos']==pos and r['cond']==c)/max(1,sum(1 for r in recs if r['pool']==pool and r['pos']==pos and r['cond']==c)):.2f}" for c in POS_CONDS)
            log(f"  {pos}: {row}")
        pre=[r for r in recs if r['pool']==pool and r['cond']=='V_self_pre']
        if pre: log(f"  pre : V_self_pre={sum(r['ok'] for r in pre)/len(pre):.2f}")

if __name__=="__main__":
    main()
