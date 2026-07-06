#!/usr/bin/env python3
"""
Set 3 / Phase 1, step 5 — build the GT-FREE V_self payload (login node, stdlib).

CORRECTION (supervisor sign-off, before any Phase-2 generation): the V_self payload must be selected
WITHOUT ever comparing a sample to the GT scene graph (selecting a "GT-perfect" sample is oracle leakage
and would invalidate the training-free/deployable claim). p1_gate.py's `self_enum` used the GT-based D and
is therefore DISCARDED for Phase 2.

Rule (frozen): V_self payload = the sample whose parsed object-multiset is the MODAL exact multiset among
the K=5 self-enumerations; ties broken by FIRST occurrence. No GT comparison anywhere in selection.

On Pool-S (D_maj=1 ⇒ ≥3/5 samples perfect ⇒ the modal multiset is the correct one) this provably yields a
correct enumeration anyway — GT-free costs nothing. On Pool-P the consensus may be wrong; that is left
uncorrected (informative). We ALSO record, post-hoc and NOT used in selection, whether the chosen payload
happens to be GT-perfect — purely to document that the two agree on Pool-S.

Output: set3_vself.jsonl {qi, pool, v_self_payload, modal_agreement(/5), gtperfect_posthoc, matches_old_self_enum}.
"""
import json
from collections import Counter
from common import parse_objects, score_enum

OUT   = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
POOLS = f"{OUT}/set3_pools.jsonl"
ORIG  = f"{OUT}/set3_orig_records.jsonl"
RECOUT= f"{OUT}/set3_vself.jsonl"

def enum_body(text): return text.split("</think>")[-1].strip()   # verbatim enumeration (post-think)

def modal_index(enums):
    """index of the sample whose parsed multiset is the MODAL exact multiset among the 5;
    ties broken by FIRST occurrence. GT-free (no scene comparison)."""
    ms = [frozenset(Counter(parse_objects(t)).items()) for t in enums]
    counts = Counter(ms)
    maxc = max(counts.values())
    first_idx = min(ms.index(m) for m in counts if counts[m] == maxc)   # earliest index among modal multisets
    return first_idx, maxc

def main():
    orig = {json.loads(l)["qi"]: json.loads(l) for l in open(ORIG)}
    pools = [json.loads(l) for l in open(POOLS)]
    nS=nP=0; s_perfect=p_perfect=0; changed=0
    with open(RECOUT, "w") as f:
        for r in pools:
            enums = r["all_enums"]
            idx, agree = modal_index(enums)
            payload = enum_body(enums[idx])
            scene = orig[r["qi"]]["scene"]
            gtperfect = score_enum(enums[idx], scene)[0]                      # POST-HOC only, not selection
            matches_old = (enum_body(r["self_enum"]) == payload)
            f.write(json.dumps(dict(qi=r["qi"], pool=r["pool"], gt_type=r["gt_type"], depth=r["depth"],
                    v_self_payload=payload, modal_agreement=agree, gtperfect_posthoc=gtperfect,
                    matches_old_self_enum=int(matches_old)))+"\n")
            if r["pool"]=="S": nS+=1; s_perfect+=gtperfect
            else: nP+=1; p_perfect+=gtperfect
            changed += (not matches_old)
    log = print
    log("="*70, f"\nGT-FREE V_self built -> {RECOUT}")
    log(f"Pool-S n={nS}: modal payload GT-perfect (post-hoc) = {s_perfect}/{nS} "
        f"({s_perfect/max(1,nS)*100:.0f}%)  <- confirms 'GT-free costs nothing on Pool-S'")
    log(f"Pool-P n={nP}: modal payload GT-perfect (post-hoc) = {p_perfect}/{nP} "
        f"({p_perfect/max(1,nP)*100:.0f}%)  <- consensus can be wrong; left uncorrected (informative)")
    log(f"payload changed vs the (leaky) old self_enum on {changed}/{len(pools)} items "
        f"(shows the GT-based selection's practical impact)")
    # modal-agreement distribution
    allrec=[json.loads(l) for l in open(RECOUT)]
    from collections import Counter as C
    log(f"modal agreement (/5) distribution: {dict(sorted(C(r['modal_agreement'] for r in allrec).items()))}")

if __name__ == "__main__":
    main()
