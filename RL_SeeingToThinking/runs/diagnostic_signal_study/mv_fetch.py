#!/usr/bin/env python3
"""
Stage MathVerse Vision-Intensive images + questions on the cluster (in-container, offline).

Login-node prefetch (internet there; compute nodes offline):
  curl -L -o {MVDIR}/testmini.parquet \
    https://huggingface.co/datasets/AI4Math/MathVerse/resolve/main/testmini.parquet
Then this (in-container, pyarrow+PIL) writes:
  {MVDIR}/images/{pid}.png   the shared VI diagram per problem
  {MVDIR}/mv_vi.jsonl        {pid, question, answer, question_type}  (VI version only)
"""
import os, io, json
import pyarrow.parquet as pq
from PIL import Image

MVDIR   = os.environ.get("MVDIR", "/iopsstor/scratch/cscs/raghavthind/set2_pilot/mv")
PARQUET = os.environ.get("MV_PARQUET", f"{MVDIR}/testmini.parquet")

def img_bytes(v):
    if isinstance(v, dict):  return v.get("bytes")
    if isinstance(v, (bytes, bytearray)): return bytes(v)
    return None

def main():
    os.makedirs(f"{MVDIR}/images", exist_ok=True)
    t = pq.read_table(PARQUET); rows = t.to_pylist()
    print("columns:", t.column_names, " n:", len(rows), flush=True)
    n = 0
    with open(f"{MVDIR}/mv_vi.jsonl", "w") as f:
        for r in rows:
            if r["problem_version"] != "Vision Intensive":
                continue
            pid = str(r["problem_index"]); b = img_bytes(r.get("image"))
            if b:
                Image.open(io.BytesIO(b)).convert("RGB").save(f"{MVDIR}/images/{pid}.png"); n += 1
            f.write(json.dumps(dict(pid=pid, question=r["question"], answer=r["answer"],
                                    question_type=r["question_type"])) + "\n")
    print(f"wrote {MVDIR}/mv_vi.jsonl   images={n}", flush=True)

if __name__ == "__main__":
    main()
