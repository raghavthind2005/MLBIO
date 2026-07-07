#!/usr/bin/env python3
"""
Set-3b / Phase-1b, step 1 — sample the FRESH, DISJOINT error-pool candidate set (login node, stdlib).

Same candidate universe as Set-3 (depth>=MIN_DEPTH, image staged), but EXCLUDING every Set-3 item AND every
Set-3 scene (image_index), per SET3B_PREREGISTRATION §2. Deterministic-PREFIX sampling: shuffle the eligible
set once with SEED and take [:N], so the pool-extension rule (§2: +2000 until Pool-S>=200) is a nested prefix
(6000 subset of 8000 subset of ...) — an extension only regenerates the new tail. Saves set3b_pool_manifest.jsonl.

Hard asserts: zero qi overlap with Set-3, zero scene (image_index) overlap with Set-3.
"""
import os, json, random
from common import norm

ROOT   = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/CLEVR_v1.0"
QJSON  = f"{ROOT}/questions/CLEVR_val_questions.json"
SJSON  = f"{ROOT}/scenes/CLEVR_val_scenes.json"
IMGDIR = f"{ROOT}/images/val"
OUT    = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
SET3_MANIFEST = f"{OUT}/set3_pool_manifest.jsonl"
MANIFEST      = f"{OUT}/set3b_pool_manifest.jsonl"

N         = int(os.environ.get("N_ITEMS", "6000"))
MIN_DEPTH = int(os.environ.get("MIN_DEPTH", "10"))
SEED      = int(os.environ.get("SEED", "100"))   # Set-3b sample seed (distinct from Set-3's 0)

def main():
    for p in (QJSON, SJSON, IMGDIR, SET3_MANIFEST):
        assert os.path.exists(p), f"missing {p}"
    Q  = json.load(open(QJSON))["questions"]
    SC = {s["image_index"]: s for s in json.load(open(SJSON))["scenes"]}
    PRESENT = set(os.listdir(IMGDIR))
    cand = [i for i in range(len(Q)) if Q[i]["image_filename"] in PRESENT and len(Q[i]["program"]) >= MIN_DEPTH]
    print(f"candidates (staged & depth>={MIN_DEPTH}): {len(cand)}   (Set-2/3 saw 22,118)")

    set3 = [json.loads(l) for l in open(SET3_MANIFEST)]
    set3_qi     = {r["qi"] for r in set3}
    set3_scenes = {r["image_index"] for r in set3}
    print(f"Set-3 footprint: items={len(set3_qi)}  scenes={len(set3_scenes)}")

    drop_item  = [i for i in cand if i in set3_qi]
    drop_scene = [i for i in cand if i not in set3_qi and Q[i]["image_index"] in set3_scenes]
    eligible   = [i for i in cand if i not in set3_qi and Q[i]["image_index"] not in set3_scenes]
    print(f"excluded: {len(drop_item)} Set-3 items + {len(drop_scene)} more sharing a Set-3 scene "
          f"-> eligible={len(eligible)}  (untouched pool)")
    assert len(eligible) >= N, f"eligible {len(eligible)} < N={N} — lower N or relax (do NOT touch Set-3 scenes)"

    rng = random.Random(SEED)
    order = eligible[:]; rng.shuffle(order)      # deterministic-prefix: chosen = order[:N] nested across N
    chosen = sorted(order[:N])

    # HARD disjointness asserts (belt-and-suspenders)
    assert not (set(chosen) & set3_qi), "OVERLAP: chosen qi intersect Set-3 items!"
    chosen_scenes = {Q[i]["image_index"] for i in chosen}
    assert not (chosen_scenes & set3_scenes), "OVERLAP: chosen scenes intersect Set-3 scenes!"

    with open(MANIFEST, "w") as f:
        for i in chosen:
            q = Q[i]
            f.write(json.dumps(dict(
                qi=i, depth=len(q["program"]), question=q["question"],
                clevr_answer=q["answer"], gt_norm=norm(q["answer"]),
                program=q["program"], image_filename=q["image_filename"],
                image_index=q["image_index"], scene=SC[q["image_index"]])) + "\n")

    depths = [len(Q[i]["program"]) for i in chosen]; scenes = {Q[i]["image_index"] for i in chosen}
    print(f"\nMANIFEST set3b: N={len(chosen)} seed={SEED} min_depth={MIN_DEPTH} -> {MANIFEST}")
    print(f"  DISJOINTNESS OK: 0 item overlap, 0 scene overlap with Set-3")
    print(f"  depth: min={min(depths)} max={max(depths)} mean={sum(depths)/len(depths):.1f}")
    print(f"  unique scenes: {len(scenes)} ({len(chosen)/len(scenes):.2f} q/scene)")
    ans_hist = {}
    for i in chosen:
        a = norm(Q[i]["answer"]); k = "count" if a.isdigit() else ("bool" if a in ("yes","no") else "attr")
        ans_hist[k] = ans_hist.get(k, 0) + 1
    print(f"  answer-type mix: {ans_hist}")
    print(f"  (extension: re-run with N_ITEMS=8000 -> chosen is a superset of this 6000)")

if __name__ == "__main__":
    main()
