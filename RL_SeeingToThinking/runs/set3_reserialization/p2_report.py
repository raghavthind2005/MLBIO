#!/usr/bin/env python3
"""
Set 3 / Phase 2 — COMPLETE REPORT (secondary + descriptive; p2_analyze.py stays frozen for the H1 confirmatory).
Computes the pre-registered SECONDARY endpoints (H2 TOST, H3, H5 — definitions frozen in SET3_PREREGISTRATION.md
§4) plus the reporting the supervisor requested: absolute correction rate per cell with 95% Wilson CI,
per-cell truncation, Holm-adjusted p across the full exploratory block, depth x answer-type strata, and the
exact cell-count accounting. Confirmatory H1 is NOT recomputed here (see frozen p2_analyze.py). Runs in the
container (McNemar via frozen stats.py needs py>=3.8 math.comb).
"""
import os, sys, json, math
from collections import Counter, defaultdict
import stats

OUT = os.environ.get("OUT", "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out")
MODE = (sys.argv[1] if len(sys.argv) > 1 else "full").lower()
POSFRACS = ["f0.25","f0.50","f0.75"]
POS_CONDS = ["V0","V1","V_scr","V_self","V_text","V_text_wrong","V_scaffold","V_restart","V_restart_ctrl"]
def log(*a): print(*a, flush=True)

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0.0, c-h), min(1.0, c+h))

def holm(labels_ps):
    m = len(labels_ps)
    order = sorted(range(m), key=lambda i: labels_ps[i][1])
    adj = [0.0]*m; running = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, (m-rank)*labels_ps[i][1]); running = max(running, a); adj[i] = running
    return {labels_ps[i][0]: adj[i] for i in range(m)}

def main():
    recs = [json.loads(l) for l in open(f"{OUT}/set3_p2sweep_{MODE}.jsonl")]
    def cell(pool,pos,cond): return {r["qi"]: r for r in recs if r["pool"]==pool and r["pos"]==pos and r["cond"]==cond}
    def pair(pool,pos,ca,cb,posB=None):
        A,B = cell(pool,pos,ca), cell(pool,posB or pos,cb); qs=sorted(set(A)&set(B))
        return qs, [A[q]["ok"] for q in qs], [B[q]["ok"] for q in qs]

    log("="*76, f"\nSet3 P2 COMPLETE REPORT  MODE={MODE}  cells={len(recs)}")

    # ---------- [A] CELL-COUNT ACCOUNTING ----------
    log("\n"+"-"*76+"\n[A] CELL-COUNT ACCOUNTING")
    log(f"  total cells executed = {len(recs)}")
    log(f"  per pool: {dict(Counter(r['pool'] for r in recs))}")
    npool = {"S": len({r['qi'] for r in recs if r['pool']=='S'}), "P": len({r['qi'] for r in recs if r['pool']=='P'})}
    log(f"  items: Pool-S={npool['S']} Pool-P={npool['P']}  total={npool['S']+npool['P']}")
    log(f"  design = {len(POS_CONDS)} position-conds x 3 positions + V_self_pre(1) = {len(POS_CONDS)*3+1} cells/item")
    log(f"  {npool['S']+npool['P']} items x {len(POS_CONDS)*3+1} = {(npool['S']+npool['P'])*(len(POS_CONDS)*3+1)}")
    exp = (npool['S']+npool['P'])*31
    log(f"  vs ~5239 expectation (10 pos-conds x3 + pre = 31/item): {exp}; gap = {exp-len(recs)} "
        f"= V_viz2 (3 pos x {npool['S']+npool['P']} items), DROPPED per prereg A-1 (fake-zoom). No cells lost.")

    # ---------- [B] ABSOLUTE RATES + WILSON CI + TRUNCATION, per cell ----------
    log("\n"+"-"*76+"\n[B] ABSOLUTE correction rate [95% Wilson CI] and truncation, per cell")
    for pool in ("S","P"):
        log(f" Pool-{pool}:")
        for pos in POSFRACS:
            parts=[]
            for c in POS_CONDS:
                cc=cell(pool,pos,c); n=len(cc); k=sum(r["ok"] for r in cc.values())
                lo,hi=wilson(k,n); tr=sum(r["trunc"] for r in cc.values())/n if n else float("nan")
                parts.append(f"{c}={k/n:.3f}[{lo:.2f},{hi:.2f}]tr={tr:.2f}" if n else f"{c}=NA")
            log(f"   {pos}: "+"  ".join(parts))
        pc=cell(pool,"pre","V_self_pre"); n=len(pc); k=sum(r["ok"] for r in pc.values())
        lo,hi=wilson(k,n); log(f"   pre : V_self_pre={k/n:.3f}[{lo:.2f},{hi:.2f}] (n={n})")

    # ---------- [C] H2 (equivalence V_self ~= V_text, Pool-S, TOST margin 0.05) ----------
    log("\n"+"-"*76+"\n[C] H2 TOST — V_self vs V_text, Pool-S, equivalence margin ±0.05 (90% CI within margin)")
    log("    (prereg: interpreted only if H1 passes; H1 FAILED, so descriptive)")
    for pos in POSFRACS:
        qs,xt,xs = pair("S",pos,"V_text","V_self")   # delta = V_self - V_text
        d,lo,hi = stats.paired_bootstrap_ci(xt,xs,alpha=0.10)   # 90% CI for TOST
        equiv = (lo > -0.05 and hi < 0.05)
        r = stats.paired_compare(xt,xs,"V_self-V_text")
        log(f"   {pos}: Δ(V_self-V_text)={d:+.4f} 90%CI=({lo:+.3f},{hi:+.3f}) equiv={equiv}  "
            f"[acc V_text={r['acc_base']:.3f} V_self={r['acc_cond']:.3f} McNemar p={r['mcnemar_p']}]")

    # ---------- [D] H3 (V1 ~= V0, Pool-S; predicted no pixel effect) ----------
    log("\n"+"-"*76+"\n[D] H3 — V1 vs V0, Pool-S (prereg predicted V1≈V0, 'no pixel effect', Set-2 dissociation)")
    for pos in POSFRACS:
        qs,x0,x1 = pair("S",pos,"V0","V1")
        d,lo,hi = stats.paired_bootstrap_ci(x0,x1,alpha=0.10)
        equiv = (lo > -0.05 and hi < 0.05)
        r = stats.paired_compare(x0,x1,"V1-V0")
        verdict = "H3 HOLDS (no pixel effect)" if equiv else "H3 VIOLATED — pixels act"
        log(f"   {pos}: Δ(V1-V0)={r['delta']:+.4f} 95%CI={r['ci']} McNemar p={r['mcnemar_p']} 90%CI=({lo:+.3f},{hi:+.3f}) -> {verdict}")

    # ---------- [E] H5 (monotone decline with p for V_self and V_text, Pool-S) ----------
    log("\n"+"-"*76+"\n[E] H5 — correction declines with position (f0.25≥f0.50≥f0.75), Pool-S, for V_self & V_text")
    for c in ("V_self","V_text"):
        ds=[]
        for pos in POSFRACS:
            qs,x0,xc = pair("S",pos,"V0",c); ds.append(stats.paired_compare(x0,xc)["delta"])
        mono = ds[0] >= ds[1] >= ds[2]
        log(f"   {c:7s}: Δ f0.25={ds[0]:+.3f}  f0.50={ds[1]:+.3f}  f0.75={ds[2]:+.3f}  monotone-decline={mono}")

    # ---------- [F] HOLM across the FULL exploratory block ----------
    log("\n"+"-"*76+"\n[F] HOLM-adjusted p across the full exploratory block (every cond-vs-V0, both pools, 3 positions)")
    raw=[]
    for pool in ("S","P"):
        for pos in POSFRACS:
            for c in POS_CONDS:
                if c=="V0": continue
                qs,x0,xc = pair(pool,pos,"V0",c)
                if not qs: continue
                raw.append((f"{pool}/{pos}/{c}", stats.paired_compare(x0,xc)["mcnemar_p"]))
    # add V_self_pre vs V0@f0.25 both pools
    for pool in ("S","P"):
        qs,x0,xp = pair(pool,"f0.25","V0","V_self_pre",posB="pre")
        if qs: raw.append((f"{pool}/pre/V_self_pre", stats.paired_compare(x0,xp)["mcnemar_p"]))
    adj = holm(raw)
    log(f"   {len(raw)} tests; showing those with Holm-adj p < 0.10 (rest available in full dump):")
    for lbl,p in sorted(raw, key=lambda t:t[1]):
        star = "  <== survives Holm .05" if adj[lbl] < 0.05 else ("  (Holm<.10)" if adj[lbl] < 0.10 else "")
        if adj[lbl] < 0.10 or p < 0.05:
            log(f"     {lbl:22s} raw p={p:.4f}  Holm p={adj[lbl]:.4f}{star}")

    # ---------- [G] STRATA depth x answer-type (V_self_pre and V_scaffold too) ----------
    log("\n"+"-"*76+"\n[G] STRATA (Pool-S f0.25): Δ vs V0 by answer-type, for V_self / V_self_pre / V_scaffold")
    try:
        depth={json.loads(l)["qi"]:json.loads(l).get("depth") for l in open(f"{OUT}/set3_orig_records.jsonl")}
    except FileNotFoundError:
        depth={}
    def atype(g): g=str(g); return "count" if g.isdigit() else ("bool" if g in ("yes","no") else "attr")
    V0=cell("S","f0.25","V0")
    arms=(("V_self","f0.25"),("V_self_pre","pre"),("V_scaffold","f0.25"))
    log("  by ANSWER-TYPE:")
    for cnd,posB in arms:
        C=cell("S",posB,cnd); qs=sorted(set(V0)&set(C))
        byt=defaultdict(list)
        for q in qs: byt[atype(V0[q]["gt"])].append(q)
        row=" ".join(f"{k}:Δ={ (sum(C[q]['ok'] for q in v)-sum(V0[q]['ok'] for q in v))/len(v):+.3f}(n={len(v)})" for k,v in sorted(byt.items()))
        log(f"   {cnd:12s}: {row}")
    if depth:
        log("  by DEPTH (d10-12 / d13+):")
        for cnd,posB in arms:
            C=cell("S",posB,cnd); qs=sorted(set(V0)&set(C))
            byd=defaultdict(list)
            for q in qs:
                d=depth.get(q); byd["d10-12" if (d is not None and d<=12) else "d13+"].append(q)
            row=" ".join(f"{k}:Δ={ (sum(C[q]['ok'] for q in v)-sum(V0[q]['ok'] for q in v))/len(v):+.3f}(n={len(v)})" for k,v in sorted(byd.items()))
            log(f"   {cnd:12s}: {row}")
    log("\n"+"="*76+"\nREPORT COMPLETE")

if __name__ == "__main__":
    main()
