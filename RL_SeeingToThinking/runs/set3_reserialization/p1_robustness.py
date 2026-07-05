#!/usr/bin/env python3
"""
Set 3 / Phase 1, step 3 — robustness labeling (vLLM). Runs ONLY on items wrong under the original sample.

robust-wrong (directive §2.3) = wrong under (i) original sample [already true for these items]
  AND (ii) greedy re-decode-from-scratch (temp 0)
  AND (iii) >= 4/5 fresh temp-1.0 resamples (seed=1, n=5 — independent of the original's seed=0).
Items failing (ii) or (iii) are "flaky" (temp-1.0 sampling flukes) — logged, excluded from treatment pools.

Saves per wrong item: greedy_ok, greedy_pred, resample_oks (5), resample_preds (5), robust_wrong, flaky.
Modes: smoke (first 10 errors) | full. Guarded under __main__.
"""
import os, sys, io, json, time, base64
from common import answer_correct, canon_ans, extract_boxed

MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
ORIG   = f"{OUT}/set3_orig_records.jsonl"
RECOUT = f"{OUT}/set3_robustness.jsonl"
BOXED  = "\n\nPut your final answer in \\boxed{}."
MAX_TOKENS = 32768
def log(*a): print(*a, flush=True)

def main():
    recs = [json.loads(l) for l in open(ORIG)]
    errs = [r for r in recs if r["correct"] == 0]
    if MODE == "smoke": errs = errs[:10]
    log("="*70, f"\nSet3 P1 ROBUSTNESS  MODE={MODE}  wrong-items={len(errs)} (of {len(recs)})")
    import torch, vllm
    from PIL import Image
    from vllm import LLM, SamplingParams
    log(f"vllm {vllm.__version__}  gpu {torch.cuda.get_device_name(0)}")

    def b64(fn):
        im=Image.open(f"{IMGDIR}/{fn}").convert("RGB"); buf=io.BytesIO(); im.save(buf,"PNG")
        return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()
    convos=[[{"role":"user","content":[{"type":"image_url","image_url":{"url":b64(r["image_filename"])}},
                                       {"type":"text","text":r["question"]+BOXED}]}] for r in errs]

    t0=time.time()
    llm=LLM(model=MODEL,dtype="bfloat16",max_model_len=40960,gpu_memory_utilization=0.90,
            limit_mm_per_prompt={"image":1},trust_remote_code=False)
    log(f"engine up {time.time()-t0:.0f}s")

    sp_greedy   = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    sp_resample = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=MAX_TOKENS, n=5, seed=1)
    t0=time.time(); g_outs = llm.chat(convos, sp_greedy);   log(f"greedy {len(g_outs)} in {time.time()-t0:.0f}s")
    t0=time.time(); r_outs = llm.chat(convos, sp_resample); log(f"resamples {len(r_outs)}x5 in {time.time()-t0:.0f}s")

    n_robust=n_flaky=0
    with open(RECOUT,"w") as f:
        for r,go,ro in zip(errs,g_outs,r_outs):
            gt=r["gt_norm"]
            g_ok=answer_correct(go.outputs[0].text, gt)
            r_oks=[answer_correct(s.text, gt) for s in ro.outputs]              # 5 samples
            r_preds=[canon_ans(extract_boxed(s.text)) for s in ro.outputs]
            n_res_wrong=sum(1 for x in r_oks if x==0)
            robust = (g_ok==0) and (n_res_wrong>=4)                            # (i) already wrong
            n_robust+=robust; n_flaky+=(not robust)
            f.write(json.dumps(dict(qi=r["qi"], depth=r["depth"], gt_norm=gt,
                    gt_type=("count" if gt.isdigit() else ("bool" if gt in("yes","no") else "attr")),
                    orig_pred=r["pred"], greedy_ok=g_ok, greedy_pred=canon_ans(extract_boxed(go.outputs[0].text)),
                    resample_oks=r_oks, resample_preds=r_preds, n_resample_wrong=n_res_wrong,
                    robust_wrong=int(robust), flaky=int(not robust)))+"\n")
    log(f"saved -> {RECOUT}")

    # ---- diagnostics ----
    out=[json.loads(l) for l in open(RECOUT)]
    log("="*70, f"\nROBUST-WRONG: {n_robust}/{len(out)}   FLAKY: {n_flaky}/{len(out)}")
    for lo,hi,lab in [(10,13,"10-13"),(14,17,"14-17"),(18,99,"18+")]:
        g=[r for r in out if lo<=r["depth"]<=hi]
        if g: log(f"  depth {lab}: n={len(g)} robust={sum(r['robust_wrong'] for r in g)} flaky={sum(r['flaky'] for r in g)}")
    for t in ("count","bool","attr"):
        g=[r for r in out if r["gt_type"]==t]
        if g: log(f"  {t:5}: n={len(g)} robust={sum(r['robust_wrong'] for r in g)} ({sum(r['robust_wrong'] for r in g)/len(g)*100:.0f}% robust)")
    # sanity on the (ii)/(iii) agreement structure
    both=sum(1 for r in out if r["greedy_ok"]==0 and r["n_resample_wrong"]>=4)
    gonly=sum(1 for r in out if r["greedy_ok"]==0 and r["n_resample_wrong"]<4)
    ronly=sum(1 for r in out if r["greedy_ok"]==1 and r["n_resample_wrong"]>=4)
    log(f"  criteria overlap: greedy-wrong&resample-wrong(robust)={both}  greedy-wrong-only={gonly}  resample-wrong-only(greedy-fixed)={ronly}")

if __name__ == "__main__":
    main()
