#!/usr/bin/env python3
"""
E1 v2 — TARGETED capability gate (OFFLINE, stdlib only; runs on the login node).

Builds a CLEVR functional-program EXECUTOR (self-validating: its computed answer must equal
the dataset ground-truth), extracts each question's RELEVANT objects (the ones the program
touches), and re-scores the ALREADY-SAVED enumeration against only those objects.

Gives RIPE_targeted (UPPER bound) to bracket RIPE_strict (lower bound = 17/21 from E1 v1):
    true RIPE  ∈  [strict, targeted].
No new generation. Executor is reusable for E2/E3 (identifies each question's hinge objects).

Limitation (stated): relevance uses GROUND-TRUTH scene facts, so a model that mis-sees an
IRRELEVANT object *into* the answer isn't flagged → this is the loose/upper side of the bracket.
"""
import json, re
from collections import Counter

V2 = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out/v2_full_records.jsonl"
E1 = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out/e1_gate_full.jsonl"

def norm(s): return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def execute(program, scene):
    """return (answer, set_of_relevant_object_indices). Self-tested against GT below."""
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

# ---- load saved records ----
v2 = {}
for l in open(V2):
    r = json.loads(l); v2[r["qi"]] = r
e1 = {}
for l in open(E1):
    r = json.loads(l); e1[r["qi"]] = r

# ---- census: every function type used, and whether the executor implements it ----
def known(f):
    if f in ("scene", "unique", "relate", "union", "or", "intersect", "and", "count",
             "exist", "equal_integer", "less_than", "greater_than"): return True
    return any(f.startswith(p) for p in ("filter_", "same_", "query_", "equal_"))
funcs = Counter((node.get("function") or node.get("type"))
                for r in v2.values() for node in r["program"])
unknown = [f for f in funcs if not known(f)]
print("function types in 200 programs:", dict(funcs))
print("UNIMPLEMENTED functions:", unknown if unknown else "none")

# ---- self-test the executor against GT (must be 200/200 or the executor is wrong) ----
ok, bad = 0, []
for qi, r in v2.items():
    try:
        ans, _ = execute(r["program"], r["scene"])
        if norm(ans) == r["gt_norm"]: ok += 1
        else: bad.append((qi, norm(ans), r["gt_norm"]))
    except Exception as ex:
        bad.append((qi, f"ERR:{type(ex).__name__}", r["gt_norm"]))
print(f"EXECUTOR self-test: {ok}/{len(v2)} programs reproduce GT answer")
if bad[:8]:
    print("  mismatches (first 8):", bad[:8])
TRUST = (ok == len(v2)) and not unknown
if not TRUST:
    print("\n!!! DO NOT TRUST targeted numbers below: self-test<100% or unimplemented function.")
    print("!!! Executor has a bug — fix it until self-test = 200/200 first.\n")

# ---- targeted gate on the error items ----
strict_ripe = sum(1 for qi, r in e1.items() if r["A"] == 0 and r["D"] == 1)
n_err = sum(1 for r in e1.values() if r["A"] == 0)
print(f"\nRIPE_strict (E1 v1) = {strict_ripe}/{n_err} errors")

print("\nper-error targeted re-check (only the perception-fails can change):")
targ_ripe = 0
for qi, r in sorted(e1.items(), key=lambda kv: kv[1]["depth"]):
    if r["A"] != 0: continue
    _, rel = execute(v2[qi]["program"], v2[qi]["scene"])
    relevant_tuples = Counter(tup(v2[qi]["scene"]["objects"][i]) for i in rel)
    model = Counter(tuple(o) for o in r["model_objs"])
    missing = relevant_tuples - model                       # relevant objects the model failed to perceive
    D_targ = int(len(missing) == 0)
    targ_ripe += (D_targ == 1)
    if r["D"] != D_targ or r["D"] == 0:                     # show the ones that were/would flip
        print(f"  qi={qi:>6} d={r['depth']:>2} gt={r['gt']!r:>7} strictD={r['D']} targD={D_targ} "
              f"|rel|={len(rel)} miss={dict(missing)}  {'-> FLIPS to RIPE' if (r['D']==0 and D_targ==1) else ''}")

print(f"\nRIPE_targeted = {targ_ripe}/{n_err} errors")
print(f"==> true RIPE  ∈  [{strict_ripe}/{n_err}, {targ_ripe}/{n_err}]  "
      f"= [{strict_ripe/n_err:.2f}, {targ_ripe/n_err:.2f}] of errors are reasoning-induced")
