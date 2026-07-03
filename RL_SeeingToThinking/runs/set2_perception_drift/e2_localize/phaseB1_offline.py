#!/usr/bin/env python3
"""
E2b Phase 1 — OFFLINE subtype breakdown (pure stdlib, runs on login node; no numpy).

Purpose: make the hard-pool NEGATIVE airtight. The headline said RIPE flip (2/39) <= control
flip (4/39), i.e. NO error-specific belief-flip. The suspicion is that every apparent "flip"
lives in the COUNT-0 subtype and is a PRIOR artifact (default belief "0 / nothing there"),
because the SAME early->think-drop appears in CORRECT count-0 controls too. This script proves
it by breaking early/think/ans margins and flip-counts down BY (group x answer-subtype).

Reads the saved trajectory JSONL(s) written by phaseB1_trajectory.py:
  rec = {qi,is_ripe,gt,model_ans,wrong_target,depth, traj:{"30|f0.05":{pgt,pwrong},...,"30|think":..,"30|ans":..}}

Usage (login node):
  python3 phaseB1_offline.py /iopsstor/.../out/phaseB1_hard_full_traj.jsonl
  python3 phaseB1_offline.py .../phaseB1_hard_full_traj.jsonl .../phaseB1_full_full_traj.jsonl   # pool both
"""
import sys, json

FRACS_EARLY = ["30|f0.05", "30|f0.15", "30|f0.25"]   # same "early" window as the online summary
COLORS = {"gray","red","blue","green","brown","purple","cyan","yellow"}

def subtype(gt):
    if gt == "0": return "count0"
    if gt.isdigit(): return "count+"
    if gt in ("yes","no"): return "bool"
    if gt in ("large","small"): return "size"
    if gt in ("rubber","metal"): return "material"
    if gt in ("cube","sphere","cylinder"): return "shape"
    if gt in COLORS: return "color"
    return "other"

def marg(rec, key):
    d = rec["traj"].get(key)
    return None if d is None else d["pgt"] - d["pwrong"]

def early(rec):
    vs = [marg(rec, k) for k in FRACS_EARLY if marg(rec, k) is not None]
    return sum(vs)/len(vs) if vs else 0.0

def mean(xs): return sum(xs)/len(xs) if xs else 0.0

def main():
    files = sys.argv[1:] or ["/iopsstor/scratch/cscs/raghavthind/set2_pilot/out/phaseB1_hard_full_traj.jsonl"]
    recs = []
    for f in files:
        for l in open(f):
            l = l.strip()
            if l: recs.append(json.loads(l))
    print(f"loaded {len(recs)} trajectories from {len(files)} file(s)\n")

    # per record: subtype, early margin, think margin, ans margin, flip flags
    for r in recs:
        r["_st"]    = subtype(r["gt"])
        r["_early"] = early(r)
        r["_think"] = marg(r, "30|think")
        r["_ans"]   = marg(r, "30|ans")
        # </think> flip: leaned correct early, negative at </think>   (the online headline criterion)
        r["_flip_think"] = int(r["_early"] > 0.01 and (r["_think"] is not None and r["_think"] < -0.01))
        # emission flip: leaned correct early, committed wrong at ans  (the flip that actually matters)
        r["_flip_ans"]   = int(r["_early"] > 0.01 and (r["_ans"]   is not None and r["_ans"]   < -0.01))

    hdr = f"{'group':7} {'subtype':8} {'n':>3} {'early':>7} {'think':>7} {'ans':>7} {'flip</t>':>8} {'flipAns':>8}"
    for is_ripe, gname in [(True,"RIPE"),(False,"correct")]:
        g = [r for r in recs if r["is_ripe"] == is_ripe]
        print("="*len(hdr)); print(f"{gname}  (n={len(g)})"); print(hdr); print("-"*len(hdr))
        sts = sorted({r["_st"] for r in g})
        for st in sts:
            gg = [r for r in g if r["_st"] == st]
            print(f"{'':7} {st:8} {len(gg):>3} {mean([r['_early'] for r in gg]):>+7.3f} "
                  f"{mean([r['_think'] for r in gg]):>+7.3f} {mean([r['_ans'] for r in gg]):>+7.3f} "
                  f"{sum(r['_flip_think'] for r in gg):>8} {sum(r['_flip_ans'] for r in gg):>8}")
        print("-"*len(hdr))
        print(f"{'':7} {'ALL':8} {len(g):>3} {mean([r['_early'] for r in g]):>+7.3f} "
              f"{mean([r['_think'] for r in g]):>+7.3f} {mean([r['_ans'] for r in g]):>+7.3f} "
              f"{sum(r['_flip_think'] for r in g):>8} {sum(r['_flip_ans'] for r in g):>8}\n")

    # THE decisive contrast: is the early-belief / flip concentrated in count0, and present in BOTH groups?
    print("="*len(hdr))
    print("DECISIVE CONTRAST — count0 vs everything-else, RIPE vs correct:")
    for is_ripe, gname in [(True,"RIPE"),(False,"correct")]:
        g = [r for r in recs if r["is_ripe"] == is_ripe]
        c0  = [r for r in g if r["_st"] == "count0"]
        rest= [r for r in g if r["_st"] != "count0"]
        print(f"  {gname:7}: count0  n={len(c0):>2} early={mean([r['_early'] for r in c0]):+.3f} "
              f"flip</t>={sum(r['_flip_think'] for r in c0)} flipAns={sum(r['_flip_ans'] for r in c0)}   |  "
              f"non-count0 n={len(rest):>2} early={mean([r['_early'] for r in rest]):+.3f} "
              f"flip</t>={sum(r['_flip_think'] for r in rest)} flipAns={sum(r['_flip_ans'] for r in rest)}")
    print("\nread: if BOTH count0 rows (RIPE and correct) carry the positive early-margin & the flips,")
    print("      while non-count0 early-margins ~0 with ~no flips => the 'flip' is a count-0 PRIOR artifact,")
    print("      not an error-specific belief drift. Confirms Method B is closed on the varied pool.")

if __name__ == "__main__":
    main()
