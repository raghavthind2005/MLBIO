#!/usr/bin/env python3
"""
Set 3 / Phase 2 — the SWEEP analysis (frozen before it is run on the completed sweep; see PREREG_FREEZE).
Consumes set3_p2sweep_{MODE}.jsonl (compact: qi,pool,pos,cond,gt,ans,ok,tok,ptok,trunc). Every paired delta
flows through the FROZEN harness stats.py. Reads the completed sweep ONCE; reports EVERY cell regardless of
outcome (analysis lockdown, amendment 11).

OUTPUT ORDER IS DELIBERATE AND H1-BLIND (amendment 12):
  [0] Descriptive accuracy + per-condition truncation tables (all cells).
  [1] TRUNCATION RE-RUN TRIGGER block — computed and printed BEFORE any H1 verdict. Trigger fires iff
      (a) McNemar p<0.05 on TRUNCATION STATUS between V_self and V0 @f0.25 (Pool-S), OR
      (b) truncation >25% in either confirmatory cell (V_self or V0 @f0.25, Pool-S).
      The H1 outcome is NOT an input here.
  [2] H1 PRIMARY: V_self-V0, Pool-S, f0.25 (Δ>=0.10 AND CI-lo>0.03 AND McNemar p<0.05).
  [3] Placebo gate A2: V_self > V_text_wrong ~= V0.
  [4] Concluded-only sensitivity: H1 on the paired subset that closed </think> in BOTH arms.
  [5] Strata (depth x answer-type) — exploratory, underpowered, descriptive.
  [6] Exploratory conditions vs V0 (all pos/pool); V_restart decomposition; V_self_pre.
  [7] Discordant-pair lists for the confirmatory cell (qualitative review via _full.jsonl).
"""
import os, sys, json
from collections import defaultdict
import stats

MODE = (sys.argv[1] if len(sys.argv) > 1 else "full").lower()
OUT  = os.environ.get("OUT", "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out")
POSFRACS = ["f0.25", "f0.50", "f0.75"]
POS_CONDS = ["V0","V1","V_scr","V_self","V_text","V_text_wrong","V_scaffold","V_restart","V_restart_ctrl"]
CONF_POS, CONF_POOL = "f0.25", "S"   # confirmatory cell
def log(*a): print(*a, flush=True)

def atype(gt):
    return "count" if str(gt).isdigit() else ("bool" if gt in ("yes","no") else "attr")

def main():
    path = f"{OUT}/set3_p2sweep_{MODE}.jsonl"
    recs = [json.loads(l) for l in open(path)]
    log("="*74, f"\nSet3 P2 SWEEP ANALYSIS  MODE={MODE}  file={path}  cells={len(recs)}")
    # depth join (exploratory strata only; not in the confirmatory endpoint)
    depth = {}
    try:
        for l in open(f"{OUT}/set3_orig_records.jsonl"):
            r = json.loads(l); depth[r["qi"]] = r.get("depth")
    except FileNotFoundError:
        log("  (orig records not found — depth strata skipped)")

    # index cells
    def cell(pool, pos, cond):
        return {r["qi"]: r for r in recs if r["pool"]==pool and r["pos"]==pos and r["cond"]==cond}
    def paired(pool, pos, ca, cb, field="ok", restrict=None):
        A, B = cell(pool,pos,ca), cell(pool,pos,cb)
        qs = sorted(set(A) & set(B))
        if restrict is not None: qs = [q for q in qs if restrict(q, A, B)]
        return qs, [A[q][field] for q in qs], [B[q][field] for q in qs]
    def rate(pool, pos, cond, field="trunc"):
        c = cell(pool,pos,cond)
        return (sum(r[field] for r in c.values())/len(c)) if c else float("nan"), len(c)

    pools = sorted({r["pool"] for r in recs})

    # ---------- [0] DESCRIPTIVE + TRUNCATION TABLES ----------
    log("\n" + "-"*74 + "\n[0] DESCRIPTIVE accuracy (corrected-fraction) and TRUNCATION rate per cell")
    for pool in pools:
        log(f"\n Pool-{pool}  accuracy:")
        for pos in POSFRACS:
            row = " ".join(f"{c}={rate(pool,pos,c,'ok')[0]:.2f}" for c in POS_CONDS)
            log(f"   {pos}: {row}")
        pre_ok = rate(pool, "pre", "V_self_pre", "ok"); log(f"   pre : V_self_pre={pre_ok[0]:.2f} (n={pre_ok[1]})")
        log(f" Pool-{pool}  truncation:")
        for pos in POSFRACS:
            row = " ".join(f"{c}={rate(pool,pos,c,'trunc')[0]:.2f}" for c in POS_CONDS)
            log(f"   {pos}: {row}")

    # ---------- [1] TRUNCATION RE-RUN TRIGGER (H1-BLIND) ----------
    log("\n" + "-"*74 + "\n[1] TRUNCATION RE-RUN TRIGGER  (H1-blind; amendment 12) — confirmatory cell "
        f"V_self vs V0 @ {CONF_POS} Pool-{CONF_POOL}")
    r_v0,  n_v0  = rate(CONF_POOL, CONF_POS, "V0",     "trunc")
    r_vs,  n_vs  = rate(CONF_POOL, CONF_POS, "V_self", "trunc")
    qs_t, t_v0, t_vs = paired(CONF_POOL, CONF_POS, "V0", "V_self", field="trunc")
    b = sum(1 for i in range(len(qs_t)) if t_vs[i]==1 and t_v0[i]==0)   # V_self trunc, V0 not
    c = sum(1 for i in range(len(qs_t)) if t_vs[i]==0 and t_v0[i]==1)   # V0 trunc, V_self not
    p_trunc = stats.mcnemar_exact(b, c)
    trig_a = p_trunc < 0.05
    trig_b = (r_v0 > 0.25) or (r_vs > 0.25)
    fired  = trig_a or trig_b
    log(f"   V0 trunc rate    = {r_v0:.3f}  (n={n_v0})")
    log(f"   V_self trunc rate= {r_vs:.3f}  (n={n_vs})")
    log(f"   trunc-status discordance: b(V_self-only)={b}  c(V0-only)={c}  McNemar p={p_trunc:.6f}")
    log(f"   trigger (a) differential McNemar p<0.05 : {trig_a}")
    log(f"   trigger (b) either cell trunc >25%      : {trig_b}  (V0>{0.25}={r_v0>0.25}, V_self>{0.25}={r_vs>0.25})")
    log(f"   >>> TRUNCATION RE-RUN TRIGGER: {'FIRED — whole-sweep re-run required (identical params)' if fired else 'NOT FIRED — primary + concluded-only stand'}")

    # ---------- [2] H1 PRIMARY ----------
    log("\n" + "-"*74 + "\n[2] H1 PRIMARY (confirmatory): V_self - V0, Pool-S, f0.25")
    qs, x, y = paired(CONF_POOL, CONF_POS, "V0", "V_self")   # x=V0, y=V_self
    v = stats.h1_verdict(x, y)
    log(f"   n={v['n']}  acc V0={v['acc_base']:.3f}  acc V_self={v['acc_cond']:.3f}")
    log(f"   Δ={v['delta']:+.4f}  95%CI={v['ci']}  McNemar p={v['mcnemar_p']}  (V_self-wins={v['cond_wins']} V0-wins={v['base_wins']})")
    log(f"   thresholds: Δ>=0.10 [{v['delta']>=0.10}]  CI-lo>0.03 [{v['ci'][0]>0.03}]  p<0.05 [{v['mcnemar_p']<0.05}]")
    log(f"   >>> H1 {'PASS' if v['H1_pass'] else 'FAIL'}")

    # ---------- [3] PLACEBO GATE A2 ----------
    log("\n" + "-"*74 + "\n[3] PLACEBO GATE A2: expect V_self > V_text_wrong ~= V0  (Pool-S, f0.25)")
    _, xw0, yw = paired(CONF_POOL, CONF_POS, "V0", "V_text_wrong")
    rw = stats.paired_compare(xw0, yw, "V_text_wrong-V0")
    _, xs, ys = paired(CONF_POOL, CONF_POS, "V_text_wrong", "V_self")
    rs = stats.paired_compare(xs, ys, "V_self-V_text_wrong")
    log(f"   V_text_wrong - V0     : Δ={rw['delta']:+.4f} CI={rw['ci']} p={rw['mcnemar_p']}  (expect ~0)")
    log(f"   V_self - V_text_wrong : Δ={rs['delta']:+.4f} CI={rs['ci']} p={rs['mcnemar_p']}  (expect >0)")
    vtw_flat = abs(rw['delta']) < 0.05 or (rw['ci'][0] <= 0 <= rw['ci'][1])   # placebo does not move accuracy
    a2_ok = vtw_flat and rs['delta'] > 0
    log(f"   A2 pattern (V_text_wrong flat AND V_self>placebo): {a2_ok}  (interpretive, not a hard gate)")

    # ---------- [4] CONCLUDED-ONLY SENSITIVITY ----------
    log("\n" + "-"*74 + "\n[4] CONCLUDED-ONLY sensitivity: H1 on the paired subset closing </think> in BOTH arms")
    A0, As = cell(CONF_POOL,CONF_POS,"V0"), cell(CONF_POOL,CONF_POS,"V_self")
    qcc = [q for q in sorted(set(A0)&set(As)) if A0[q]["trunc"]==0 and As[q]["trunc"]==0]
    xcc = [A0[q]["ok"] for q in qcc]; ycc = [As[q]["ok"] for q in qcc]
    if qcc:
        vcc = stats.h1_verdict(xcc, ycc)
        drop = len(set(A0)&set(As)) - len(qcc)
        log(f"   n_concluded={vcc['n']} (dropped {drop} truncated-in-either)  acc V0={vcc['acc_base']:.3f} V_self={vcc['acc_cond']:.3f}")
        log(f"   Δ={vcc['delta']:+.4f} CI={vcc['ci']} p={vcc['mcnemar_p']}  >>> H1(concluded-only) {'PASS' if vcc['H1_pass'] else 'FAIL'}")
    else:
        log("   no fully-concluded pairs (unexpected)")

    # ---------- [5] STRATA (exploratory, descriptive) ----------
    log("\n" + "-"*74 + "\n[5] STRATA (exploratory, underpowered): H1 Δ within depth x answer-type (Pool-S, f0.25)")
    if depth:
        buckets = defaultdict(list)
        for q in qs:  # qs = confirmatory paired items
            d = depth.get(q); dk = "d?" if d is None else ("d10-12" if d<=12 else "d13+")
            buckets[(dk,)].append(q)
        A0, As = cell(CONF_POOL,CONF_POS,"V0"), cell(CONF_POOL,CONF_POS,"V_self")
        for key in sorted(buckets):
            qk = buckets[key]; xk=[A0[q]["ok"] for q in qk]; yk=[As[q]["ok"] for q in qk]
            rk = stats.paired_compare(xk, yk, str(key))
            log(f"   depth {key[0]:6s} n={rk['n']:3d}: Δ={rk['delta']:+.3f} (V_self-wins={rk['cond_wins']} V0-wins={rk['base_wins']})")
    at = defaultdict(list)
    A0, As = cell(CONF_POOL,CONF_POS,"V0"), cell(CONF_POOL,CONF_POS,"V_self")
    for q in qs: at[atype(A0[q]["gt"])].append(q)
    for k in sorted(at):
        qk=at[k]; xk=[A0[q]["ok"] for q in qk]; yk=[As[q]["ok"] for q in qk]
        rk=stats.paired_compare(xk,yk,k)
        log(f"   atype {k:6s} n={rk['n']:3d}: Δ={rk['delta']:+.3f} (V_self-wins={rk['cond_wins']} V0-wins={rk['base_wins']})")

    # ---------- [6] EXPLORATORY conditions vs V0 ----------
    log("\n" + "-"*74 + "\n[6] EXPLORATORY: each condition vs V0 (paired), all positions/pools")
    for pool in pools:
        for pos in POSFRACS:
            log(f"  Pool-{pool} {pos}:")
            for cnd in POS_CONDS:
                if cnd=="V0": continue
                _, xb, yb = paired(pool, pos, "V0", cnd)
                if not xb: continue
                rr = stats.paired_compare(xb, yb, cnd)
                log(f"     {cnd:14s} Δ={rr['delta']:+.3f} CI={rr['ci']} p={rr['mcnemar_p']} (+{rr['cond_wins']}/-{rr['base_wins']})")
    # V_restart decomposition + V_self_pre
    log("  V_restart decomposition (Pool-S f0.25):")
    for a,bnd in [("V0","V_restart_ctrl"),("V_restart_ctrl","V_restart"),("V0","V_restart")]:
        _, xa, ya = paired("S","f0.25",a,bnd)
        if xa:
            rr=stats.paired_compare(xa,ya,f"{bnd}-{a}")
            log(f"     {bnd} - {a:14s} Δ={rr['delta']:+.3f} CI={rr['ci']} p={rr['mcnemar_p']}")
    for pool in pools:
        Apre = cell(pool,"pre","V_self_pre"); A0 = cell(pool,CONF_POS,"V0")
        qs2 = sorted(set(Apre)&set(A0))
        if qs2:
            rr = stats.paired_compare([A0[q]["ok"] for q in qs2], [Apre[q]["ok"] for q in qs2], "V_self_pre-V0")
            log(f"  Pool-{pool} V_self_pre - V0 (vs f0.25 V0): Δ={rr['delta']:+.3f} CI={rr['ci']} p={rr['mcnemar_p']}")

    # ---------- [7] DISCORDANT PAIRS (confirmatory cell) ----------
    log("\n" + "-"*74 + "\n[7] DISCORDANT PAIRS — confirmatory cell (for qualitative review via _full.jsonl)")
    A0, As = cell(CONF_POOL,CONF_POS,"V0"), cell(CONF_POOL,CONF_POS,"V_self")
    fixed  = [q for q in qs if As[q]["ok"]==1 and A0[q]["ok"]==0]
    broke  = [q for q in qs if As[q]["ok"]==0 and A0[q]["ok"]==1]
    log(f"   V_self FIXED (V0 wrong -> V_self right), n={len(fixed)}: {fixed}")
    log(f"   V_self BROKE (V0 right -> V_self wrong), n={len(broke)}: {broke}")
    log("\n" + "="*74 + "\nANALYSIS COMPLETE")

if __name__ == "__main__":
    main()
