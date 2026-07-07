#!/usr/bin/env python3
"""
A5 step 0 — MATERIALIZE MathVista testmini from a LOCAL parquet (offline; runs in the container).
The parquet is downloaded once on the login node via curl (internet there; nodes are offline):
  curl -L -o {A5DIR}/testmini.parquet \
    https://huggingface.co/datasets/AI4Math/MathVista/resolve/main/data/testmini-00000-of-00001-725687bf7a18d64b.parquet
This reads it with pyarrow (ships with `datasets`), writes:
  {A5DIR}/images/{pid}.png        decoded image per item
  {A5DIR}/testmini.jsonl          pid, question, choices, answer, question_type, answer_type,
                                  precision, unit, query, has_image   (NO model outputs)
and prints schema + answer_type/question_type distributions + 3 examples (for scorer authoring).
"""
import os, io, json
from collections import Counter
import pyarrow.parquet as pq
from PIL import Image

A5DIR = os.environ.get("A5DIR", "/iopsstor/scratch/cscs/raghavthind/set2_pilot/a5")
PARQUET = os.environ.get("A5_PARQUET", f"{A5DIR}/testmini.parquet")
def log(*a): print(*a, flush=True)

def img_bytes(v):
    if v is None: return None
    if isinstance(v, dict): return v.get("bytes")
    if isinstance(v, (bytes, bytearray)): return bytes(v)
    return None

def main():
    os.makedirs(f"{A5DIR}/images", exist_ok=True)
    t = pq.read_table(PARQUET)
    cols = t.column_names
    log("columns:", cols)
    rows = t.to_pylist()
    log("n:", len(rows))
    img_col = next((c for c in ("decoded_image","image") if c in cols), None)
    log("image column:", img_col)
    keep = ["pid","question","choices","answer","question_type","answer_type","precision","unit","query"]
    n_img = 0
    with open(f"{A5DIR}/testmini.jsonl","w") as f:
        for r in rows:
            pid = str(r["pid"])
            b = img_bytes(r.get(img_col)) if img_col else None
            has = False
            if b:
                try:
                    Image.open(io.BytesIO(b)).convert("RGB").save(f"{A5DIR}/images/{pid}.png"); n_img += 1; has = True
                except Exception as e:
                    log(f"  WARN pid={pid} image decode failed: {e}")
            row = {k: r.get(k) for k in keep}; row["has_image"] = has
            f.write(json.dumps(row, default=str)+"\n")
    log(f"wrote {A5DIR}/testmini.jsonl  images={n_img}/{len(rows)}")
    log("\n--- distributions ---")
    log("answer_type:", dict(Counter(str(r.get("answer_type")) for r in rows)))
    log("question_type:", dict(Counter(str(r.get("question_type")) for r in rows)))
    log("\n--- 3 example items ---")
    for r in rows[:3]:
        log(json.dumps({k:(str(r.get(k))[:220]) for k in keep}, indent=1, default=str))

if __name__ == "__main__":
    main()
