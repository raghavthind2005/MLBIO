#!/usr/bin/env python3
"""
E1 v1 — behavioral RIPE via a full-scene ENUMERATION capability gate.

For each v2 item (all 200 → errors + correct controls):
  A = was the FULL-REASONING answer correct?  (re-scored from the saved chain, synonym-aware)
  D = can the model perceive the WHOLE scene in isolation?  (NEW short probe: enumerate every
      object as size·color·material·shape, matched to the scene graph — deterministic, no judge)
  RIPE = {A=0 AND D=1}  → wrong answer despite intact perception → reasoning-induced (drift OR logic).

Reports RIPE rate, the control P(D=1|A=1), and a per-error breakdown. Saves every probe
(full text + parsed objects + D) so nothing needs re-running.  Caveat: full-scene enumeration
is a STRICT gate (miss one object → D=0) → RIPE is a conservative LOWER bound.

Guarded under __main__ (vLLM spawn). Modes (argv[1]): smoke (5) | full (all 200).
"""
import os, sys, io, re, json, time, base64
import numpy as np
from collections import Counter

MODE   = (sys.argv[1] if len(sys.argv) > 1 else "smoke").lower()
MODEL  = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
IMGDIR = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
RECIN  = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out/v2_full_records.jsonl"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
RECOUT = f"{OUT}/e1_gate_{MODE}.jsonl"
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

# ---- answer re-scoring (synonym-aware, matches rescore.py) ----
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
    return bool(gt) and (p == gt or gt in p.split() or p in gt.split())

# ---- perception scoring ----
def first(mapping, text):
    for w, v in mapping.items():
        if re.search(rf"\b{w}\b", text): return v
    return None

def parse_objects(text):
    objs = []
    for line in text.splitlines():
        t = line.lower()
        sz, co, ma, sh = first(SIZES, t), first(COLORS, t), first(MATS, t), first(SHAPES, t)
        if sz and co and ma and sh: objs.append((sz, co, ma, sh))
    return objs

def scene_tuples(scene):
    return [(o["size"], o["color"], o["material"], o["shape"]) for o in scene["objects"]]


def main():
    os.makedirs(OUT, exist_ok=True)
    log("="*70, f"\nE1 GATE  MODE={MODE}")
    import torch, vllm, transformers
    from PIL import Image
    log(f"torch {torch.__version__}  vllm {vllm.__version__}  transformers {transformers.__version__}  gpu {torch.cuda.get_device_name(0)}")

    rows = [json.loads(l) for l in open(RECIN)]
    if MODE == "smoke": rows = rows[:5]
    log(f"items: {len(rows)}")

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=20480, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=False)
    log(f"engine up in {time.time()-t0:.0f}s")
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=MAX_TOKENS, seed=0)

    def b64(fn):
        im = Image.open(f"{IMGDIR}/{fn}").convert("RGB"); buf = io.BytesIO(); im.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    convos = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": b64(r["image_filename"])}},
                                            {"type": "text", "text": ENUM}]}] for r in rows]
    t0 = time.time(); outs = llm.chat(convos, sp); log(f"enumerated {len(outs)} in {time.time()-t0:.0f}s")

    recs = []
    with open(RECOUT, "w") as f:
        for r, o in zip(rows, outs):
            text = o.outputs[0].text; trunc = o.outputs[0].finish_reason == "length"
            model_objs = parse_objects(text.split("</think>")[-1])          # parse the answer, not the reasoning
            sc = scene_tuples(r["scene"])
            matched = sum((Counter(model_objs) & Counter(sc)).values())
            pscore = matched / len(sc) if sc else 0.0
            D = int(Counter(model_objs) == Counter(sc))
            A = int(answer_correct(r["full_text"], r["gt_norm"]))
            rec = dict(qi=r["qi"], depth=r["depth"], gt=r["gt_norm"], chain_tok=r["parsed"]["chain_tok"],
                       A=A, D=D, perception_score=round(pscore, 3), n_scene=len(sc), n_model=len(model_objs),
                       trunc=int(trunc), model_objs=model_objs, enum_text=text.split("</think>")[-1][:800])
            recs.append(rec); f.write(json.dumps(rec, default=str) + "\n")
            if MODE == "smoke":
                log(f"[qi{r['qi']}] A={A} D={D} pscore={pscore:.2f} n_scene={len(sc)} n_model={len(model_objs)} trunc={int(trunc)}")
    log(f"saved -> {RECOUT}")

    # ---- RIPE ----
    log("="*70, "\nRIPE SUMMARY")
    n = len(recs); err = [x for x in recs if x["A"] == 0]; cor = [x for x in recs if x["A"] == 1]
    ripe = [x for x in err if x["D"] == 1]; pfail = [x for x in err if x["D"] == 0]
    log(f"items={n}  correct={len(cor)}  errors={len(err)}")
    log(f"perception gate D=1 (perfect scene enumeration): overall {sum(x['D'] for x in recs)}/{n}"
        f"   | among correct {sum(x['D'] for x in cor)}/{len(cor)}   | among errors {len(ripe)}/{len(err)}")
    log(f"mean perception_score: correct={np.mean([x['perception_score'] for x in cor]) if cor else 0:.3f}  "
        f"errors={np.mean([x['perception_score'] for x in err]) if err else 0:.3f}")
    log(f"RIPE (A=0 & D=1) = {len(ripe)}/{n} = {len(ripe)/n:.3f}   ({len(ripe)}/{len(err)} of errors are reasoning-induced, not perception-inability)")
    log(f"perception-failure errors (A=0 & D=0) = {len(pfail)}/{len(err)}")
    log("\nerror breakdown (D=1 => RIPE/reasoning-induced; D=0 => perception failure):")
    for x in sorted(err, key=lambda z: (-z["D"], z["depth"])):
        tag = "RIPE" if x["D"] == 1 else "percept-fail"
        log(f"  qi={x['qi']:>6} d={x['depth']:>2} chain={x['chain_tok']:>6} pscore={x['perception_score']:.2f} "
            f"n={x['n_model']}/{x['n_scene']} gt={x['gt']!r:>8} -> {tag}")
    json.dump(dict(mode=MODE, n=n, n_err=len(err), ripe=len(ripe), ripe_rate=len(ripe)/n if n else 0,
                   ripe_of_errors=len(ripe)/len(err) if err else 0, D_given_correct=sum(x['D'] for x in cor)/len(cor) if cor else 0),
              open(f"{OUT}/e1_gate_{MODE}_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
