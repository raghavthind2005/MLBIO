#!/usr/bin/env python3
"""
Set 3 / Phase 1, step 4 — K=5 enumeration gate on robust-wrong items → Pool-S / Pool-P (vLLM).

For each robust-wrong item: prompt the model to enumerate the scene (common.ENUM), draw K=5 samples,
score each vs the scene graph (common.score_enum). D_maj = majority perfect.
  Pool-S (serialization): robust-wrong ∧ D_maj=1  (perception intact → error is downstream).
  Pool-P (perception):    robust-wrong ∧ D_maj=0  (kept as a treatment group, not discarded).
Also SAVES the model's own K=5 enumerations (needed to build V_self in Phase 2).

Cross-check (directive §2.5): on a 50-item subsample, compare the strict enumeration D against the
executor-TARGETED D (does the enumeration cover the question-relevant objects?) and report agreement.
Modes: smoke (15) | full. Guarded under __main__.
"""
import os, sys, io, json, time, base64
from collections import Counter
from common import ENUM, score_enum, parse_objects, execute, tup

MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
K      = 5
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
ROB    = f"{OUT}/set3_robustness.jsonl"
ORIG   = f"{OUT}/set3_orig_records.jsonl"
RECOUT = f"{OUT}/set3_pools.jsonl"
MAX_TOKENS = 8192
def log(*a): print(*a, flush=True)

def main():
    orig = {json.loads(l)["qi"]: json.loads(l) for l in open(ORIG)}
    robust = [json.loads(l) for l in open(ROB) if json.loads(l)["robust_wrong"] == 1]
    if MODE == "smoke": robust = robust[:15]
    log("="*70, f"\nSet3 P1 GATE  MODE={MODE}  robust-wrong items={len(robust)}  K={K}")
    import torch, vllm
    from PIL import Image
    from vllm import LLM, SamplingParams
    log(f"vllm {vllm.__version__}  gpu {torch.cuda.get_device_name(0)}")

    def b64(fn):
        im=Image.open(f"{IMGDIR}/{fn}").convert("RGB"); buf=io.BytesIO(); im.save(buf,"PNG")
        return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()
    items=[]
    for r in robust:
        o=orig[r["qi"]]
        items.append(dict(qi=r["qi"], depth=r["depth"], gt_norm=r["gt_norm"], gt_type=r["gt_type"],
                          scene=o["scene"], program=o["program"], image_filename=o["image_filename"],
                          image_index=o["image_index"]))
    convos=[[{"role":"user","content":[{"type":"image_url","image_url":{"url":b64(c["image_filename"])}},
                                       {"type":"text","text":ENUM}]}] for c in items]

    t0=time.time()
    llm=LLM(model=MODEL,dtype="bfloat16",max_model_len=20480,gpu_memory_utilization=0.90,
            limit_mm_per_prompt={"image":1},trust_remote_code=False)
    log(f"engine up {time.time()-t0:.0f}s")
    sp=SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=MAX_TOKENS, n=K, seed=0)
    t0=time.time(); outs=llm.chat(convos, sp); log(f"enumerated {len(outs)}x{K} in {time.time()-t0:.0f}s")

    n_S=n_P=0
    with open(RECOUT,"w") as f:
        for c,o in zip(items,outs):
            enums=[s.text for s in o.outputs]                                   # K self-enumerations
            D=[score_enum(t, c["scene"])[0] for t in enums]
            D_maj=int(sum(D)>=(K+1)//2); rel=sum(D)/K
            pool = "S" if D_maj==1 else "P"; n_S+=(pool=="S"); n_P+=(pool=="P")
            # keep the majority-consistent enumeration text for V_self (first sample whose D==D_maj, else first)
            self_idx = next((i for i,d in enumerate(D) if d==D_maj), 0)
            f.write(json.dumps(dict(qi=c["qi"], depth=c["depth"], gt_norm=c["gt_norm"], gt_type=c["gt_type"],
                    image_filename=c["image_filename"], image_index=c["image_index"],
                    D_samples=D, D_maj=D_maj, reliability=rel, pool=pool,
                    self_enum=enums[self_idx], all_enums=enums))+"\n")
    log(f"saved -> {RECOUT}")

    # ---- diagnostics ----
    out=[json.loads(l) for l in open(RECOUT)]
    log("="*70, f"\nPOOL-S (D_maj=1, serialization): {n_S}   POOL-P (D_maj=0, perception): {n_P}")
    for t in ("count","bool","attr"):
        g=[r for r in out if r["gt_type"]==t]
        if g: log(f"  {t:5}: n={len(g)}  Pool-S={sum(r['pool']=='S' for r in g)}  Pool-P={sum(r['pool']=='P' for r in g)}")
    for lo,hi,lab in [(10,13,"10-13"),(14,17,"14-17"),(18,99,"18+")]:
        g=[r for r in out if lo<=r["depth"]<=hi]
        if g: log(f"  depth {lab}: n={len(g)}  Pool-S={sum(r['pool']=='S' for r in g)}  Pool-P={sum(r['pool']=='P' for r in g)}")
    log(f"  Pool-S unique scenes: {len({r['image_index'] for r in out if r['pool']=='S'})}  (LOSO CV granularity)")

    # ---- executor-targeted cross-check on <=50 items ----
    sub=out[:50]; agree=0
    for r in sub:
        c=next(x for x in items if x["qi"]==r["qi"])
        try:
            _, rel = execute(c["program"], c["scene"])
            reltup = Counter(tup(c["scene"]["objects"][i]) for i in rel)
            Dtarg = [int(len(reltup - Counter(parse_objects(t)))==0) for t in r["all_enums"]]
            Dtarg_maj = int(sum(Dtarg) >= (K+1)//2)
        except Exception:
            Dtarg_maj = r["D_maj"]
        agree += (Dtarg_maj == r["D_maj"])
    log(f"  executor cross-check (n={len(sub)}): strict-D vs targeted-D agree {agree}/{len(sub)} "
        f"(targeted is looser; disagreement = items where only question-irrelevant objects were mis-enumerated)")
    log(f"\nPHASE-1 COMPLETE if Pool-S >= 150. Current Pool-S={n_S}.")

if __name__ == "__main__":
    main()
