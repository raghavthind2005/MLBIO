#!/usr/bin/env python3
"""
E1 audit-fix #3 — MULTI-SAMPLE capability gate D.

E1 measured D (does the model perceive the whole scene?) from a SINGLE enumeration sample
at temp=1.0, so a lucky/unlucky draw can flip D and thus the RIPE count. This re-measures D
over K independent enumeration samples per item and reports perception RELIABILITY, then
recomputes RIPE under robust D definitions (all-K / majority / any) to show how stable 17/21 is.

A (answer correctness) is re-derived from the saved reasoning chain (synonym-aware, matches
rescore/clevr_exec). D from K fresh enumeration samples. Guarded under __main__ (vLLM spawn).
Modes (argv[1]): smoke (5 items) | full (200).
"""
import os, sys, io, re, json, time, base64
import numpy as np
from collections import Counter

MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
DSET   = os.environ.get("DSET", "full")                    # which v2 dataset to gate: full | hard
K      = 5
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
RECIN  = f"{OUT}/v2_{DSET}_records.jsonl"
RECOUT = f"{OUT}/e1_gate_multi_{DSET}.jsonl"
MAX_TOKENS = 8192
ENUM = ("List every object in the image. Output ONE object per line in EXACTLY this format:\n"
        "<size> <color> <material> <shape>\n"
        "using size in {large, small}, material in {metal, rubber}, shape in {cube, sphere, cylinder}, "
        "and the usual 8 colors. Example line: large red metal cube\nList only the objects, nothing else.")

SIZES  = {"large": "large", "big": "large", "small": "small", "tiny": "small", "little": "small"}
COLORS = {"gray": "gray", "grey": "gray", "red": "red", "blue": "blue", "green": "green",
          "brown": "brown", "purple": "purple", "cyan": "cyan", "yellow": "yellow"}
MATS   = {"rubber": "rubber", "matte": "rubber", "metal": "metal", "metallic": "metal", "shiny": "metal"}
SHAPES = {"cube": "cube", "block": "cube", "sphere": "sphere", "ball": "sphere", "cylinder": "cylinder"}

def log(*a): print(*a, flush=True)

# ---- answer re-scoring (synonym-aware, matches clevr_exec/rescore) ----
def extract_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx < 0: return None
    i, depth, buf = idx + 7, 1, []
    while i < len(text) and depth:
        ch = text[i]
        if ch == "{": depth += 1
        elif ch == "}": depth -= 1
        if depth: buf.append(ch)
        i += 1
    return "".join(buf)

def canon_ans(s):
    if not s: return ""
    s = re.sub(r"\\(?:text|mathrm|mathbf|textbf|mathsf|textit)\s*\{([^{}]*)\}", r"\1", s).replace("\\", " ")
    p = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    if re.search(r"\bmatte\b", p) or "non metal" in p or "not metal" in p: return "rubber"
    if re.search(r"\b(metallic|shiny)\b", p): return "metal"
    p = re.sub(r"\bbig\b", "large", p); p = re.sub(r"\b(tiny|little)\b", "small", p)
    return p

def answer_correct(full_text, gt):
    p = canon_ans(extract_boxed(full_text))
    return int(bool(gt) and (p == gt or gt in p.split() or p in gt.split()))

# ---- enumeration scoring ----
def first(m, t):
    for w, v in m.items():
        if re.search(rf"\b{w}\b", t): return v
    return None
def parse_objects(text):
    objs = []
    for line in text.splitlines():
        t = line.lower(); sz, co, ma, sh = first(SIZES, t), first(COLORS, t), first(MATS, t), first(SHAPES, t)
        if sz and co and ma and sh: objs.append((sz, co, ma, sh))
    return objs
def scene_tuples(scene): return [(o["size"], o["color"], o["material"], o["shape"]) for o in scene["objects"]]
def score_enum(text, scene):
    m = parse_objects(text.split("</think>")[-1]); sc = scene_tuples(scene)
    matched = sum((Counter(m) & Counter(sc)).values())
    return int(Counter(m) == Counter(sc)), (matched / len(sc) if sc else 0.0)


def main():
    os.makedirs(OUT, exist_ok=True)
    log("="*70, f"\nE1 MULTI-SAMPLE D  MODE={MODE}  K={K}")
    import torch, vllm
    from PIL import Image
    log(f"torch {torch.__version__}  vllm {vllm.__version__}  gpu {torch.cuda.get_device_name(0)}")
    rows = [json.loads(l) for l in open(RECIN)]
    if MODE == "smoke": rows = rows[:5]
    log(f"items: {len(rows)}")

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=20480, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=False)
    log(f"engine up in {time.time()-t0:.0f}s")
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=MAX_TOKENS, n=K, seed=0)

    def b64(fn):
        im = Image.open(f"{IMGDIR}/{fn}").convert("RGB"); buf = io.BytesIO(); im.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    convos = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": b64(r["image_filename"])}},
                                            {"type": "text", "text": ENUM}]}] for r in rows]
    t0 = time.time(); outs = llm.chat(convos, sp); log(f"enumerated {len(outs)}x{K} in {time.time()-t0:.0f}s")

    recs = []
    with open(RECOUT, "w") as f:
        for r, o in zip(rows, outs):
            Ds = [score_enum(c.text, r["scene"])[0] for c in o.outputs]     # K samples
            A = answer_correct(r["full_text"], r["gt_norm"])
            rel = float(np.mean(Ds))
            rec = dict(qi=r["qi"], depth=r["depth"], gt=r["gt_norm"], A=A,
                       D_samples=Ds, reliability=rel,
                       D_all=int(all(Ds)), D_maj=int(sum(Ds) >= (K + 1) // 2), D_any=int(any(Ds)))
            recs.append(rec); f.write(json.dumps(rec) + "\n")
    log(f"saved -> {RECOUT}")

    # ---- how stable is RIPE under different D definitions? ----
    log("="*70, "\nRIPE STABILITY under multi-sample D")
    err = [x for x in recs if x["A"] == 0]; cor = [x for x in recs if x["A"] == 1]
    n = len(recs)
    def ripe(key): return sum(1 for x in err if x[key] == 1)
    log(f"items={n} correct={len(cor)} errors={len(err)}")
    log(f"RIPE (single-sample equivalent = D_any) : {ripe('D_any')}/{len(err)}")
    log(f"RIPE (majority of {K})                  : {ripe('D_maj')}/{len(err)}   <- ROBUST headline")
    log(f"RIPE (all {K} — strict)                 : {ripe('D_all')}/{len(err)}")
    if cor:
        log(f"control mean perception reliability: correct={np.mean([x['reliability'] for x in cor]):.3f}  "
            f"errors={np.mean([x['reliability'] for x in err]) if err else 0:.3f}")
    log("\nper-error perception reliability (fraction of K enumerations that were perfect):")
    for x in sorted(err, key=lambda z: z["reliability"]):
        tag = "RIPE(maj)" if x["D_maj"] else ("percept-fail" if not x["D_any"] else "borderline")
        log(f"  qi={x['qi']:>6} d={x['depth']:>2} gt={x['gt']!r:>7} reliability={x['reliability']:.2f} "
            f"D={x['D_samples']} -> {tag}")
    json.dump(dict(mode=MODE, K=K, n=n, n_err=len(err),
                   ripe_any=ripe('D_any'), ripe_maj=ripe('D_maj'), ripe_all=ripe('D_all')),
              open(f"{OUT}/e1_gate_multi_{DSET}_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
