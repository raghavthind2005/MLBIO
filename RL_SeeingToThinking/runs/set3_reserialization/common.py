#!/usr/bin/env python3
"""
Set 3 — shared module. The Set-2 synonym-aware SCORER, enumeration GATE, and CLEVR EXECUTOR,
copied VERBATIM from Set 2 (e1_ripe/e1_gate_multi.py and e1_ripe/clevr_exec.py) so every Set-3
script scores identically — no silent drift. Stdlib only (importable on the bare login node).

Run `python3 common.py` on the cluster to SELF-TEST against saved Set-2 data:
  (1) executor reproduces GT answer on all v2 records (must be 100%),
  (2) this module's `answer_correct` reproduces Set-2's stored `A` labels exactly (scorer regression).
Both must pass before any Set-3 scaling (directive §2.5 guardrail).
"""
import re, json, os
from collections import Counter

OUT = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"

# ============================================================ normalization + scorer
# (verbatim: v2_accuracy.norm ; e1_gate_multi.extract_boxed / canon_ans / answer_correct)
def norm(s): return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def extract_boxed(text):
    if text is None: return None
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
    """1 if the model's boxed answer (synonym-canonicalized) matches gt (already norm()'d)."""
    p = canon_ans(extract_boxed(full_text))
    return int(bool(gt) and (p == gt or gt in p.split() or p in gt.split()))

# ============================================================ enumeration gate (verbatim e1_gate_multi)
SIZES  = {"large": "large", "big": "large", "small": "small", "tiny": "small", "little": "small"}
COLORS = {"gray": "gray", "grey": "gray", "red": "red", "blue": "blue", "green": "green",
          "brown": "brown", "purple": "purple", "cyan": "cyan", "yellow": "yellow"}
MATS   = {"rubber": "rubber", "matte": "rubber", "metal": "metal", "metallic": "metal", "shiny": "metal"}
SHAPES = {"cube": "cube", "block": "cube", "sphere": "sphere", "ball": "sphere", "cylinder": "cylinder"}
ENUM = ("List every object in the image. Output ONE object per line in EXACTLY this format:\n"
        "<size> <color> <material> <shape>\n"
        "using size in {large, small}, material in {metal, rubber}, shape in {cube, sphere, cylinder}, "
        "and the usual 8 colors. Example line: large red metal cube\nList only the objects, nothing else.")

def _first(m, t):
    for w, v in m.items():
        if re.search(rf"\b{w}\b", t): return v
    return None
def parse_objects(text):
    objs = []
    for line in text.splitlines():
        t = line.lower(); sz, co, ma, sh = _first(SIZES, t), _first(COLORS, t), _first(MATS, t), _first(SHAPES, t)
        if sz and co and ma and sh: objs.append((sz, co, ma, sh))
    return objs
def scene_tuples(scene): return [(o["size"], o["color"], o["material"], o["shape"]) for o in scene["objects"]]
def score_enum(text, scene):
    """returns (perfect:int, matched_fraction:float) — perfect==1 iff the enumeration exactly equals the scene."""
    m = parse_objects(text.split("</think>")[-1]); sc = scene_tuples(scene)
    matched = sum((Counter(m) & Counter(sc)).values())
    return int(Counter(m) == Counter(sc)), (matched / len(sc) if sc else 0.0)

# ============================================================ CLEVR executor (verbatim clevr_exec)
def execute(program, scene):
    """return (answer, set_of_relevant_object_indices). Relevant set drives V_viz2 (question-referenced crop)."""
    objs, rel, n = scene["objects"], scene["relationships"], len(scene["objects"])
    vals, relevant = [], set()
    for node in program:
        f = node.get("function") or node.get("type")
        vi = node.get("value_inputs", []) or []
        ins = [vals[j] for j in (node.get("inputs", []) or [])]
        if f == "scene":                 out = set(range(n))
        elif f.startswith("filter_"):    a = f[7:]; out = {i for i in ins[0] if objs[i][a] == vi[0]}
        elif f == "unique":              out = next(iter(ins[0]))
        elif f == "relate":              out = set(rel[vi[0]][ins[0]])
        elif f.startswith("same_"):      a = f[5:]; i = ins[0]; out = {j for j in range(n) if j != i and objs[j][a] == objs[i][a]}
        elif f in ("union", "or"):       out = set(ins[0]) | set(ins[1])
        elif f in ("intersect", "and"):  out = set(ins[0]) & set(ins[1])
        elif f == "count":               out = len(ins[0])
        elif f == "exist":               out = "yes" if len(ins[0]) > 0 else "no"
        elif f.startswith("query_"):     out = objs[ins[0]][f[6:]]
        elif f == "equal_integer":       out = "yes" if ins[0] == ins[1] else "no"
        elif f.startswith("equal_"):     out = "yes" if ins[0] == ins[1] else "no"
        elif f == "less_than":           out = "yes" if ins[0] < ins[1] else "no"
        elif f == "greater_than":        out = "yes" if ins[0] > ins[1] else "no"
        else:                            out = None
        if isinstance(out, set): relevant |= out
        if f == "unique":        relevant.add(out)
        vals.append(out)
    return vals[-1], relevant
def tup(o): return (o["size"], o["color"], o["material"], o["shape"])


# ============================================================ SELF-TEST
def _selftest():
    dsets = [d for d in ("full", "hard") if os.path.exists(f"{OUT}/v2_{d}_records.jsonl")]
    assert dsets, f"no v2_*_records.jsonl under {OUT} — cannot self-test"
    V2, GATE = {}, {}
    for d in dsets:
        for l in open(f"{OUT}/v2_{d}_records.jsonl"):
            r = json.loads(l); V2.setdefault(r["qi"], r)
        gp = f"{OUT}/e1_gate_multi_{d}.jsonl"
        if os.path.exists(gp):
            for l in open(gp):
                g = json.loads(l); GATE.setdefault(g["qi"], g)
    print(f"loaded {len(V2)} v2 records, {len(GATE)} gate records from {dsets}")

    # (1) executor reproduces GT
    ok, bad = 0, []
    for qi, r in V2.items():
        try:
            ans, rel = execute(r["program"], r["scene"])
            if norm(ans) == r["gt_norm"] and len(rel) > 0: ok += 1
            else: bad.append((qi, norm(ans), r["gt_norm"], len(rel)))
        except Exception as ex:
            bad.append((qi, f"ERR:{type(ex).__name__}", r["gt_norm"], -1))
    print(f"(1) EXECUTOR self-test: {ok}/{len(V2)} reproduce GT (and relevant-set nonempty)")
    if bad[:5]: print("    mismatches:", bad[:5])

    # (2) scorer regression: recompute A, must equal Set-2 stored A
    match, mism = 0, []
    for qi, g in GATE.items():
        if qi not in V2: continue
        A = answer_correct(V2[qi]["full_text"], V2[qi]["gt_norm"])
        if A == g["A"]: match += 1
        else: mism.append((qi, A, g["A"]))
    print(f"(2) SCORER regression: {match}/{len(GATE)} A-labels reproduce Set-2 exactly")
    if mism[:5]: print("    mismatches:", mism[:5])

    passed = (ok == len(V2)) and (not mism) and (match == len([qi for qi in GATE if qi in V2]))
    print("\nSELF-TEST", "PASSED" if passed else "FAILED — do not scale until fixed")
    return passed

if __name__ == "__main__":
    _selftest()
