#!/usr/bin/env python3
"""
INDEPENDENT verification of the sweep numbers quoted to the supervisor. Reads the raw
set3_p2sweep_full.jsonl and recomputes accuracies / deltas / discordant counts / truncation rates
FROM SCRATCH (no import of p2_analyze or stats; pure stdlib -> runs on the login node). If these
match the frozen analyzer's block output, the quoted figures are trustworthy. No p-values here
(McNemar needs math.comb / py>=3.8); we verify the FACTUAL quantities the p-values are computed from.
"""
import os, json
from collections import Counter, defaultdict
OUT = os.environ.get("OUT", "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out")

def main():
    recs = [json.loads(l) for l in open(f"{OUT}/set3_p2sweep_full.jsonl")]
    print(f"total cells = {len(recs)}   (expect 4732)")
    print(f"total truncated = {sum(r['trunc'] for r in recs)}   (expect 438)")
    pools = Counter(r["pool"] for r in recs)
    print("cells per pool:", dict(pools))
    # cells per (pool,pos,cond) — expect Pool-S 149 / Pool-P 20, minus V_text_wrong skips
    cnt = Counter((r["pool"], r["pos"], r["cond"]) for r in recs)
    print("\ncell counts (Pool-S f0.25, all conds) — expect 149 except V_text_wrong (skips):")
    for c in ["V0","V1","V_scr","V_self","V_text","V_text_wrong","V_scaffold","V_restart","V_restart_ctrl"]:
        print(f"   {c:14s} S/f0.25 = {cnt[('S','f0.25',c)]:3d}   P/f0.25 = {cnt[('P','f0.25',c)]:2d}")
    print(f"   V_self_pre     S/pre   = {cnt[('S','pre','V_self_pre')]:3d}   P/pre   = {cnt[('P','pre','V_self_pre')]:2d}")

    def cell(pool, pos, cond): return {r["qi"]: r for r in recs if r["pool"]==pool and r["pos"]==pos and r["cond"]==cond}
    def cmp(pool, pos, ca, cb, restrict=None):
        A, B = cell(pool,pos,ca), cell(pool,pos,cb)
        qs = sorted(set(A) & set(B))
        if restrict: qs = [q for q in qs if restrict(A[q], B[q])]
        n = len(qs)
        accA = sum(A[q]["ok"] for q in qs)/n; accB = sum(B[q]["ok"] for q in qs)/n
        bwin = sum(1 for q in qs if B[q]["ok"]==1 and A[q]["ok"]==0)   # cond(B) wins
        awin = sum(1 for q in qs if B[q]["ok"]==0 and A[q]["ok"]==1)   # base(A) wins
        tA = sum(A[q]["trunc"] for q in qs)/n; tB = sum(B[q]["trunc"] for q in qs)/n
        print(f"   {cb:14s}-{ca:12s} @{pool}/{pos}: n={n:3d}  acc({ca})={accA:.3f} acc({cb})={accB:.3f}  "
              f"Δ={accB-accA:+.4f}  +{bwin}/-{awin}  trunc({ca})={tA:.3f} trunc({cb})={tB:.3f}")

    print("\n[H1 + trigger cell] V_self vs V0, Pool-S f0.25:")
    cmp("S","f0.25","V0","V_self")
    print("   trunc-status discordance (b=V_self-only, c=V0-only):")
    A, B = cell("S","f0.25","V0"), cell("S","f0.25","V_self"); qs=sorted(set(A)&set(B))
    b = sum(1 for q in qs if B[q]["trunc"]==1 and A[q]["trunc"]==0)
    c = sum(1 for q in qs if B[q]["trunc"]==0 and A[q]["trunc"]==1)
    print(f"      b={b} c={c}   (analyzer said b=15 c=8)")
    print("\n[concluded-only] both closed </think>:")
    cmp("S","f0.25","V0","V_self", restrict=lambda a,b: a["trunc"]==0 and b["trunc"]==0)
    print("\n[placebo A2]:")
    cmp("S","f0.25","V0","V_text_wrong"); cmp("S","f0.25","V_text_wrong","V_self")
    print("\n[exploratory Pool-S f0.25 vs V0]:")
    for c in ["V1","V_scr","V_text","V_scaffold","V_restart","V_restart_ctrl"]:
        cmp("S","f0.25","V0",c)
    print("\n[restart decomposition]:")
    cmp("S","f0.25","V_restart_ctrl","V_restart")
    print("\n[V_self_pre vs V0 @f0.25]:")
    cmp("S","f0.25","V0","V_self_pre"); cmp("P","f0.25","V0","V_self_pre")
    print("\n[strata by answer-type, V_self vs V0 Pool-S f0.25]:")
    A, B = cell("S","f0.25","V0"), cell("S","f0.25","V_self"); qs=sorted(set(A)&set(B))
    def atype(g): g=str(g); return "count" if g.isdigit() else ("bool" if g in ("yes","no") else "attr")
    byt=defaultdict(list)
    for q in qs: byt[atype(A[q]["gt"])].append(q)
    for k in sorted(byt):
        qk=byt[k]; accA=sum(A[q]["ok"] for q in qk)/len(qk); accB=sum(B[q]["ok"] for q in qk)/len(qk)
        bw=sum(1 for q in qk if B[q]["ok"]==1 and A[q]["ok"]==0); aw=sum(1 for q in qk if B[q]["ok"]==0 and A[q]["ok"]==1)
        print(f"   {k:6s} n={len(qk):3d} Δ={accB-accA:+.3f} +{bw}/-{aw}")

if __name__ == "__main__":
    main()
