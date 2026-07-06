#!/usr/bin/env python3
"""
Set 3 — targeted inspection of enumeration samples lacking </think> (the one audit WARN).
Confirms whether the malformed sample propagates: is it the V_self-selected (modal) sample?
does it inflate D_maj? Prints the propagation-relevant facts per affected item. Stdlib.
"""
import json
from common import parse_objects, score_enum
from p1_vself import modal_index, enum_body
OUT="/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
def load(f): return {json.loads(l)["qi"]: json.loads(l) for l in open(f"{OUT}/{f}")}

def main():
    pools=load("set3_pools.jsonl"); orig=load("set3_orig_records.jsonl")
    affected=0
    for qi,r in pools.items():
        bad=[i for i,t in enumerate(r["all_enums"]) if "</think>" not in t]
        if not bad: continue
        affected+=1
        idx,agree=modal_index(r["all_enums"])
        print("="*60)
        print(f"qi={qi} pool={r['pool']} D_samples={r['D_samples']} D_maj={r['D_maj']} modal_agree={agree}/5")
        for i in bad:
            t=r["all_enums"][i]
            print(f"  BAD sample #{i}: len={len(t)} chars, D_of_sample={r['D_samples'][i]}, "
                  f"IS_MODAL_SELECTED={i==idx}, parses_to={len(parse_objects(enum_body(t)))} objs")
        # propagation facts
        payload=enum_body(r["all_enums"][idx])
        perfect=score_enum(r["all_enums"][idx], orig[qi]["scene"])[0]
        others_perfect=sum(r["D_samples"][j] for j in range(5) if j not in bad)
        print(f"  -> V_self from sample #{idx} (bad? {idx in bad}); payload GT-perfect={perfect}; "
              f"payload_len={len(payload)}")
        print(f"  -> D_maj drivers: {others_perfect} of the {5-len(bad)} GOOD samples are perfect "
              f"(a bad sample scores D=0, so it can only DEFLATE D_maj, never inflate -> Pool-S membership is safe/conservative)")
    print("="*60)
    print(f"affected items: {affected}")
    print("VERDICT: benign iff for every affected item — (a) the bad sample is NOT the modal/V_self source,")
    print("         and (b) the item's Pool-S status is driven by >=3 GOOD perfect samples (bad sample D=0 can't inflate).")

if __name__=="__main__":
    main()
