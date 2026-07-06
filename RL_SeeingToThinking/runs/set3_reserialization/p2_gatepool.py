#!/usr/bin/env python3
"""
Set 3 / Phase 2 — ROBUSTLY-CORRECT gate pool (vLLM). Items for the parity-gate v3 (Layer A + Layer B).
Symmetric to Pool-S construction: keep items that are greedy-correct AND >=4/5 resamples-correct — a clean,
stable-correct baseline so a wrong-payload accuracy drop (Layer B) is meaningful. Zero overlap with
Pool-S/Pool-P by construction (those are robust-WRONG); asserted defensively. Preserves pre-reg blindness.

Candidates: fixed-seed sample of correct-under-original items with depth>=10. Saves set3_gatepool.jsonl.
Modes: smoke (10 candidates) | full (default 70). Guarded under __main__.
"""
import os, sys, io, json, time, base64, random
from common import answer_correct

MODE=(sys.argv[1] if len(sys.argv)>1 else "full").lower()
N_CAND=int(os.environ.get("N_CAND", "10" if MODE=="smoke" else "70"))
MODEL="/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR="/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT="/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
BOXED="\n\nPut your final answer in \\boxed{}."; MAX_TOKENS=32768
def log(*a): print(*a, flush=True)

def main():
    orig=[json.loads(l) for l in open(f"{OUT}/set3_orig_records.jsonl")]
    pool_qi={json.loads(l)["qi"] for l in open(f"{OUT}/set3_pools.jsonl")}
    cand=[r for r in orig if r["correct"]==1 and r["depth"]>=10 and r["qi"] not in pool_qi]
    random.Random(0).shuffle(cand); cand=cand[:N_CAND]
    assert not ({r["qi"] for r in cand} & pool_qi), "gate candidates overlap Pool-S/P!"
    log("="*70, f"\nSet3 P2 GATEPOOL  MODE={MODE}  candidates={len(cand)} (correct, depth>=10, disjoint from pools)")
    import torch, vllm
    from PIL import Image
    from vllm import LLM, SamplingParams
    def b64(fn):
        im=Image.open(f"{IMGDIR}/{fn}").convert("RGB"); buf=io.BytesIO(); im.save(buf,"PNG")
        return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()
    convos=[[{"role":"user","content":[{"type":"image_url","image_url":{"url":b64(r["image_filename"])}},
                                       {"type":"text","text":r["question"]+BOXED}]}] for r in cand]
    t0=time.time()
    llm=LLM(model=MODEL,dtype="bfloat16",max_model_len=40960,gpu_memory_utilization=0.90,
            limit_mm_per_prompt={"image":1},trust_remote_code=False); log(f"engine up {time.time()-t0:.0f}s")
    g=llm.chat(convos, SamplingParams(temperature=0.0,max_tokens=MAX_TOKENS))
    r5=llm.chat(convos, SamplingParams(temperature=1.0,top_p=0.95,top_k=20,max_tokens=MAX_TOKENS,n=5,seed=1))

    n_rc=0
    with open(f"{OUT}/set3_gatepool.jsonl","w") as f:
        for r,go,ro in zip(cand,g,r5):
            gt=r["gt_norm"]; g_ok=answer_correct(go.outputs[0].text,gt)
            r_oks=[answer_correct(s.text,gt) for s in ro.outputs]; nrc=sum(r_oks)
            rc=(g_ok==1 and nrc>=4)                                        # robustly correct
            n_rc+=rc
            f.write(json.dumps(dict(qi=r["qi"],depth=r["depth"],
                    gt_type=("count" if gt.isdigit() else("bool" if gt in("yes","no") else "attr")),
                    greedy_ok=g_ok, resample_oks=r_oks, robust_correct=int(rc)))+"\n")
    log(f"saved -> {OUT}/set3_gatepool.jsonl   ROBUSTLY-CORRECT: {n_rc}/{len(cand)}")
    log("  (Layer B needs >=30; Layer A needs 10/payload-type — sufficient if robust-correct >= ~40)")

if __name__=="__main__":
    main()
