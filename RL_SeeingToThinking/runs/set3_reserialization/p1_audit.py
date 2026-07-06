#!/usr/bin/env python3
"""
Set 3 / Phase 1 — DEEP INTEGRITY AUDIT (login node, stdlib). Re-derives every labeled quantity from its
source and asserts consistency, so any silent/propagating error surfaces as FAIL. Reads ONLY saved files.

Checks: file counts & internal consistency · qi uniqueness · qi set-nesting across the pipeline ·
orig 'correct' re-scored from full_text · executor reproduces GT on all items (program/scene integrity) ·
robust_wrong re-derived from greedy+resamples · Pool D_maj/label re-derived from D_samples ·
gt_norm & gt_type consistency across all files · V_self modal payload re-derived + Pool-S GT-perfect ·
enumeration truncation health · staged-image existence.
"""
import json, os
from collections import Counter
from common import answer_correct, execute, norm, score_enum, parse_objects
from p1_vself import modal_index, enum_body

OUT="/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
IMGDIR="/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0/images/val"
def load(f): return [json.loads(l) for l in open(f"{OUT}/{f}")]
def gt_type(g): return "count" if g.isdigit() else ("bool" if g in("yes","no") else "attr")

RESULTS=[]
def check(name, passed, detail=""):
    RESULTS.append((passed, name, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))

def main():
    man   = load("set3_pool_manifest.jsonl")
    orig  = load("set3_orig_records.jsonl")
    rob   = load("set3_robustness.jsonl")
    pools = load("set3_pools.jsonl")
    vself = load("set3_vself.jsonl")
    M={r["qi"]:r for r in man}; O={r["qi"]:r for r in orig}
    R={r["qi"]:r for r in rob}; P={r["qi"]:r for r in pools}; V={r["qi"]:r for r in vself}
    print("="*70, "\nSet3 Phase-1 DEEP AUDIT")
    print(f"rows: manifest={len(man)} orig={len(orig)} robustness={len(rob)} pools={len(pools)} vself={len(vself)}")

    # ---- C1 counts + uniqueness ----
    check("manifest N=3000", len(man)==3000, f"got {len(man)}")
    check("orig N=3000", len(orig)==3000, f"got {len(orig)}")
    for nm,d,lst in [("manifest",M,man),("orig",O,orig),("rob",R,rob),("pools",P,pools),("vself",V,vself)]:
        check(f"{nm} qi unique", len(d)==len(lst), f"{len(lst)-len(d)} dup")

    # ---- C2 qi set-nesting through the pipeline ----
    errs = {r["qi"] for r in orig if r["correct"]==0}
    robust_qi = {r["qi"] for r in rob if r["robust_wrong"]==1}
    check("orig qi == manifest qi", set(O)==set(M))
    check("robustness qi == orig-error qi", set(R)==errs, f"rob={len(R)} errs={len(errs)}")
    check("pools qi == robust-wrong qi", set(P)==robust_qi, f"pools={len(P)} robust={len(robust_qi)}")
    check("vself qi == pools qi", set(V)==set(P))

    # ---- C3 orig 'correct' re-scored from full_text (scoring integrity) ----
    bad=[qi for qi,r in O.items() if answer_correct(r["full_text"], r["gt_norm"])!=r["correct"]]
    check("orig 'correct' reproducible via answer_correct", not bad, f"{len(bad)} mismatch {bad[:5]}")
    check("orig error count == 407", sum(1-r["correct"] for r in orig)==407, f"got {sum(1-r['correct'] for r in orig)}")

    # ---- C4 executor reproduces GT on ALL items (program/scene integrity) ----
    exbad=[]; exrel0=[]
    for qi,r in O.items():
        try:
            ans,rel=execute(r["program"], r["scene"])
            if norm(ans)!=r["gt_norm"]: exbad.append((qi,norm(ans),r["gt_norm"]))
            if not rel: exrel0.append(qi)
        except Exception as e: exbad.append((qi,f"ERR:{type(e).__name__}",r["gt_norm"]))
    check("executor reproduces GT on all 3000 (program/scene integrity)", not exbad, f"{len(exbad)} fail {exbad[:3]}")
    check("executor relevant-set nonempty on all (for V_viz2)", not exrel0, f"{len(exrel0)} empty")

    # ---- C5 robustness logic re-derived ----
    r5=[qi for qi,r in R.items() if len(r["resample_oks"])!=5]
    check("robustness has 5 resamples each", not r5, f"{len(r5)} bad")
    nrw=[qi for qi,r in R.items() if r["n_resample_wrong"]!=sum(1-x for x in r["resample_oks"])]
    check("n_resample_wrong == sum(wrong resamples)", not nrw, f"{len(nrw)} mismatch")
    rwbad=[qi for qi,r in R.items() if r["robust_wrong"]!=int(r["greedy_ok"]==0 and r["n_resample_wrong"]>=4)]
    check("robust_wrong == (greedy_wrong & >=4/5 resample_wrong)", not rwbad, f"{len(rwbad)} mismatch {rwbad[:5]}")
    fl=[qi for qi,r in R.items() if r["flaky"]!=(1-r["robust_wrong"])]
    check("flaky == not robust_wrong", not fl, f"{len(fl)} mismatch")
    check("all robustness items are orig-wrong", all(O[qi]["correct"]==0 for qi in R), "")
    check("robust+flaky == 407", sum(r["robust_wrong"] for r in rob)+sum(r["flaky"] for r in rob)==407)

    # ---- C6 pool logic re-derived ----
    ds=[qi for qi,r in P.items() if len(r["D_samples"])!=5]
    check("pools have 5 D-samples each", not ds, f"{len(ds)} bad")
    dmaj=[qi for qi,r in P.items() if r["D_maj"]!=int(sum(r["D_samples"])>=3)]
    check("D_maj == majority of 5 D-samples", not dmaj, f"{len(dmaj)} mismatch")
    pl=[qi for qi,r in P.items() if r["pool"]!=("S" if r["D_maj"]==1 else "P")]
    check("pool label == (S if D_maj else P)", not pl, f"{len(pl)} mismatch")
    check("Pool-S count == 149", sum(r["pool"]=="S" for r in pools)==149, f"got {sum(r['pool']=='S' for r in pools)}")

    # ---- C7 gt_norm / gt_type consistency across files ----
    gnb=[qi for qi in P if not (M[qi]["gt_norm"]==O[qi]["gt_norm"]==R[qi]["gt_norm"]==P[qi]["gt_norm"]==V[qi]["gt_norm"])]
    check("gt_norm identical across manifest/orig/rob/pools/vself", not gnb, f"{len(gnb)} mismatch {gnb[:5]}")
    gtb=[qi for qi,r in P.items() if r["gt_type"]!=gt_type(r["gt_norm"])]
    check("gt_type consistent with gt_norm", not gtb, f"{len(gtb)} mismatch")

    # ---- C8 scene consistency (orig scene == manifest scene, same qi) ----
    scb=[qi for qi in P if len(O[qi]["scene"]["objects"])!=len(M[qi]["scene"]["objects"])
                        or O[qi]["image_index"]!=M[qi]["image_index"]]
    check("orig scene/image matches manifest (pool items)", not scb, f"{len(scb)} mismatch")

    # ---- C9 V_self modal payload re-derived + Pool-S 100% GT-perfect + non-empty ----
    vbad=[]; vs_perf=0; vs_n=0; empty=[]
    for qi,r in P.items():
        idx,_=modal_index(r["all_enums"]); payload=enum_body(r["all_enums"][idx])
        if payload!=V[qi]["v_self_payload"]: vbad.append(qi)
        if not payload.strip() or not parse_objects(payload): empty.append((qi,r["pool"]))
        if r["pool"]=="S":
            vs_n+=1; vs_perf+=score_enum(r["all_enums"][idx], O[qi]["scene"])[0]
    check("V_self payload reproducible (modal rule)", not vbad, f"{len(vbad)} mismatch {vbad[:5]}")
    check("Pool-S V_self is GT-perfect (post-hoc) 149/149", vs_perf==vs_n==149, f"{vs_perf}/{vs_n}")
    check("no empty/unparseable V_self payload", not empty, f"{len(empty)} empty {empty[:5]}")

    # ---- C10 enumeration truncation health (gate max_tokens=8192) ----
    noclose=sum(1 for r in pools for t in r["all_enums"] if "</think>" not in t)
    tot=sum(len(r["all_enums"]) for r in pools)
    check("enumeration samples closed </think> (truncation health)", noclose==0,
          f"{noclose}/{tot} samples lack </think> (possible enum truncation) — note if >0")

    # ---- C11 staged images exist for all pool items ----
    present=set(os.listdir(IMGDIR))
    imb=[qi for qi,r in P.items() if r["image_filename"] not in present]
    check("all pool images staged", not imb, f"{len(imb)} missing")

    # ---- summary ----
    npass=sum(1 for p,_,_ in RESULTS if p); n=len(RESULTS)
    print("="*70, f"\nAUDIT: {npass}/{n} checks passed")
    fails=[(nm,d) for p,nm,d in RESULTS if not p]
    if fails:
        print("FAILURES:");  [print(f"  - {nm}: {d}") for nm,d in fails]
        print("\n!! DO NOT PROCEED until failures are understood/fixed.")
    else:
        print("ALL CHECKS PASSED — Phase-1 data is internally consistent and reproducible from source.")
    # informative (non-gating) note
    print("\nNOTE (unverifiable from saved data): p1_robustness.py did not log finish_reason, so greedy/resample "
          "truncation is not directly auditable. Mitigant: orig had 0/3000 truncation at max_tokens=32768; "
          "greedy(temp0)/resamples are not longer. Risk low but unproven.")

if __name__=="__main__":
    main()
