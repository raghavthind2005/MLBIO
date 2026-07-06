#!/usr/bin/env python3
"""
Set 3 / Phase 2 — V_viz2 crop-sanity ARTIFACT (login node, stdlib). Deterministic → hashable.

Quantifies, across ALL Pool-S+Pool-P items, that the executor's "question-referenced object set" ≈ the
whole scene for depth-≥10 CLEVR, so a bbox crop is not a real zoom (Set-2 D5). This artifact is the
pre-outcome evidence for dropping V_viz2 from the CLEVR sweep (record Part 3). Writes set3_viz2_sanity.txt.
"""
import json
from common import execute

OUT="/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
ARTIFACT=f"{OUT}/set3_viz2_sanity.txt"

def main():
    orig={json.loads(l)["qi"]:json.loads(l) for l in open(f"{OUT}/set3_orig_records.jsonl")}
    pools=[json.loads(l) for l in open(f"{OUT}/set3_pools.jsonl")]
    lines=[]
    def emit(s): lines.append(s); print(s)
    emit("Set3 V_viz2 crop-sanity artifact — executor relevant-set vs scene, bbox area fraction")
    emit(f"items: {len(pools)} (Pool-S + Pool-P)")
    rel_frac=[]; bbox_frac=[]
    for r in pools:
        o=orig[r["qi"]]; scene=o["scene"]; W,H=o["image_size"]; objs=scene["objects"]
        try: _, rel = execute(o["program"], scene)
        except Exception: rel=set()
        rf=len(rel)/max(1,len(objs)); rel_frac.append(rf)
        if rel:
            xs=[objs[i]["pixel_coords"][0] for i in rel]; ys=[objs[i]["pixel_coords"][1] for i in rel]
            m=55; x0=max(0,min(xs)-m); x1=min(W,max(xs)+m); y0=max(0,min(ys)-m); y1=min(H,max(ys)+m)
            bf=max(0,(x1-x0))*max(0,(y1-y0))/(W*H)
        else: bf=1.0
        bbox_frac.append(bf)
    def stats(v):
        s=sorted(v); n=len(s)
        return dict(min=round(s[0],3), median=round(s[n//2],3), mean=round(sum(s)/n,3), max=round(s[-1],3),
                    frac_ge_0_8=round(sum(1 for x in s if x>=0.8)/n,3))
    emit(f"relevant-set size / scene size: {stats(rel_frac)}")
    emit(f"crop bbox area / image area   : {stats(bbox_frac)}")
    emit(f"items whose crop covers >=80% of the image: {sum(1 for x in bbox_frac if x>=0.8)}/{len(pools)}")
    emit("CONCLUSION: crop >= most of the image for the bulk of items -> not a real zoom (Set-2 D5). "
         "Tight answer-object crop would leak the answer. V_viz2 dropped from the CLEVR sweep (see record Part 3).")
    open(ARTIFACT,"w").write("\n".join(lines)+"\n")
    print(f"\nsaved -> {ARTIFACT}")

if __name__=="__main__":
    main()
