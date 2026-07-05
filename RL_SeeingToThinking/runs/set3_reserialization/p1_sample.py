#!/usr/bin/env python3
"""
Set 3 / Phase 1, step 1 — sample the error-pool candidate set (login node, stdlib).

Fixed-seed FLAT random sample of N depth>=MIN_DEPTH questions whose image is staged (the "22,118
candidates" from Set 2). Saves a manifest with everything downstream needs (qi, depth, question,
program, scene, image). `qi` = integer index into CLEVR_val_questions.json (Set-2 convention).

Diagnostics: candidate count (expect ~22,118 — consistency with Set 2), depth histogram,
unique-scene count (for leave-one-scene-out CV later). Deterministic given SEED.
"""
import os, json, random
from common import norm

ROOT   = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0"
QJSON  = f"{ROOT}/questions/CLEVR_val_questions.json"
SJSON  = f"{ROOT}/scenes/CLEVR_val_scenes.json"
IMGDIR = f"{ROOT}/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
MANIFEST = f"{OUT}/set3_pool_manifest.jsonl"

N         = int(os.environ.get("N_ITEMS", "3000"))
MIN_DEPTH = int(os.environ.get("MIN_DEPTH", "10"))
SEED      = int(os.environ.get("SEED", "0"))

def main():
    for p in (QJSON, SJSON, IMGDIR):
        assert os.path.exists(p), f"missing {p} — did CLEVR staging finish?"
    Q  = json.load(open(QJSON))["questions"]
    SC = {s["image_index"]: s for s in json.load(open(SJSON))["scenes"]}
    PRESENT = set(os.listdir(IMGDIR))
    print(f"val questions={len(Q)}  scenes={len(SC)}  staged images={len(PRESENT)}")

    cand = [i for i in range(len(Q)) if Q[i]["image_filename"] in PRESENT and len(Q[i]["program"]) >= MIN_DEPTH]
    print(f"candidates (image staged & depth>={MIN_DEPTH}): {len(cand)}   (Set-2 saw 22,118)")
    if abs(len(cand) - 22118) > 500:
        print(f"  !! WARNING: candidate count deviates from Set-2's 22,118 by >500 — staging may have changed.")
    assert len(cand) >= N, f"only {len(cand)} candidates < N={N}"

    rng = random.Random(SEED)
    chosen = sorted(rng.sample(cand, N))     # flat random, fixed seed; sorted for stable manifest order

    with open(MANIFEST, "w") as f:
        for i in chosen:
            q = Q[i]
            f.write(json.dumps(dict(
                qi=i, depth=len(q["program"]), question=q["question"],
                clevr_answer=q["answer"], gt_norm=norm(q["answer"]),
                program=q["program"], image_filename=q["image_filename"],
                image_index=q["image_index"], scene=SC[q["image_index"]])) + "\n")

    # ---- diagnostics ----
    depths = [len(Q[i]["program"]) for i in chosen]
    scenes = {Q[i]["image_index"] for i in chosen}
    imgs_present = all(Q[i]["image_filename"] in PRESENT for i in chosen)
    print(f"\nMANIFEST: N={len(chosen)}  seed={SEED}  min_depth={MIN_DEPTH}  -> {MANIFEST}")
    print(f"  depth: min={min(depths)} max={max(depths)} mean={sum(depths)/len(depths):.1f}")
    hist = {}
    for d in depths:
        b = "10-13" if d<=13 else ("14-17" if d<=17 else "18+")   # the A3 analysis bins
        hist[b] = hist.get(b,0)+1
    print(f"  depth bins (A3): {hist}")
    print(f"  unique scenes: {len(scenes)}  (avg {len(chosen)/len(scenes):.2f} questions/scene; LOSO CV granularity)")
    print(f"  all sampled images staged: {imgs_present}")
    ans_hist = {}
    for i in chosen:
        a = norm(Q[i]["answer"]); k = "count" if a.isdigit() else ("bool" if a in("yes","no") else "attr")
        ans_hist[k] = ans_hist.get(k,0)+1
    print(f"  answer-type mix: {ans_hist}")

if __name__ == "__main__":
    main()
