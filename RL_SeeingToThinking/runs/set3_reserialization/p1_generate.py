#!/usr/bin/env python3
"""
Set 3 / Phase 1, step 2 — generate ORIGINAL chains over the pool manifest (vLLM).

Set-2 recipe VERBATIM: temp 1.0, top_p 0.95, top_k 20, seed 0, max_tokens 32768, boxed prompt.
Scores with the validated common.answer_correct. Saves v2-compatible records (qi, full_text,
output_token_ids, gt_norm, scene, program, depth, image_*, question) so Phase-2/3 read them unchanged.
This is criterion (i) of robust-wrong; only the WRONG items go on to p1_robustness.py.

Modes (argv[1]): smoke (10 items, verbose) | full (all manifest). Guarded under __main__ (vLLM spawn).
"""
import os, sys, io, json, time, base64
from common import norm, answer_correct, extract_boxed, canon_ans

MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
ROOT   = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0"
IMGDIR = f"{ROOT}/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
MANIFEST = f"{OUT}/set3_pool_manifest.jsonl"
RECFILE  = f"{OUT}/set3_orig_records.jsonl"
BOXED    = "\n\nPut your final answer in \\boxed{}."
MAX_TOKENS = 32768

def log(*a): print(*a, flush=True)

def main():
    items = [json.loads(l) for l in open(MANIFEST)]
    if MODE == "smoke": items = items[:10]
    log("="*70, f"\nSet3 P1 GENERATE  MODE={MODE}  items={len(items)}")
    import torch, vllm, transformers
    from PIL import Image
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    log(f"torch {torch.__version__}  vllm {vllm.__version__}  transformers {transformers.__version__}  gpu {torch.cuda.get_device_name(0)}")
    tok = AutoTokenizer.from_pretrained(MODEL)

    def b64(fn):
        im = Image.open(f"{IMGDIR}/{fn}").convert("RGB"); buf = io.BytesIO(); im.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), im.size
    convos, sizes = [], []
    for c in items:
        bb, sz = b64(c["image_filename"]); sizes.append(sz)
        convos.append([{"role":"user","content":[{"type":"image_url","image_url":{"url":bb}},
                                                 {"type":"text","text":c["question"]+BOXED}]}])

    t0 = time.time()
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=40960, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image":1}, trust_remote_code=False)
    log(f"engine up {time.time()-t0:.0f}s")
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=MAX_TOKENS, seed=0)
    t0 = time.time(); outs = llm.chat(convos, sp); log(f"generated {len(outs)} in {time.time()-t0:.0f}s")

    n_err = n_trunc = 0
    with open(RECFILE, "w") as f:
        for c, o, sz in zip(items, outs, sizes):
            out = o.outputs[0]; text = out.text; tids = list(out.token_ids); fr = out.finish_reason
            ok = answer_correct(text, c["gt_norm"]); trunc = (fr == "length")
            n_err += (ok == 0); n_trunc += trunc
            rec = dict(qi=c["qi"], depth=c["depth"], question=c["question"], clevr_answer=c["clevr_answer"],
                       gt_norm=c["gt_norm"], program=c["program"], image_filename=c["image_filename"],
                       image_index=c["image_index"], image_size=sz, scene=c["scene"],
                       sampling=dict(temperature=1.0, top_p=0.95, top_k=20, max_tokens=MAX_TOKENS, seed=0),
                       finish_reason=fr, n_tok=len(tids), trunc=int(trunc),
                       full_text=text, output_token_ids=tids,
                       boxed=extract_boxed(text), pred=canon_ans(extract_boxed(text)), correct=ok)
            f.write(json.dumps(rec, default=str) + "\n")
            if MODE == "smoke":
                log(f"  qi={c['qi']:>6} d={c['depth']:>2} gt={c['gt_norm']!r:>7} pred={rec['pred']!r:>8} "
                    f"ok={ok} trunc={int(trunc)} n_tok={len(tids)} | {c['question'][:40]}")
    log(f"saved -> {RECFILE}")

    # ---- diagnostics ----
    recs = [json.loads(l) for l in open(RECFILE)]
    acc = sum(r["correct"] for r in recs)/len(recs)
    log("="*70, f"\nACCURACY {acc:.3f}  errors={n_err}/{len(recs)}  truncated={n_trunc}")
    for lo,hi,lab in [(10,13,"10-13"),(14,17,"14-17"),(18,99,"18+")]:
        g=[r for r in recs if lo<=r["depth"]<=hi]
        if g: log(f"  depth {lab}: n={len(g)} acc={sum(x['correct'] for x in g)/len(g):.2f} errors={sum(1-x['correct'] for x in g)}")
    et={}
    for r in recs:
        if not r["correct"]:
            k="count" if r["gt_norm"].isdigit() else ("bool" if r["gt_norm"] in("yes","no") else "attr")
            et[k]=et.get(k,0)+1
    log(f"  error subtype mix: {et}")
    if n_trunc: log(f"  NOTE: {n_trunc} truncated (finish=length) — flagged (trunc=1), excluded-from-error-analysis candidates")

if __name__ == "__main__":
    main()
