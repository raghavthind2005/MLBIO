#!/usr/bin/env python3
"""
A5 step 0 — fetch MathVista testmini (1000 items) and MATERIALIZE it for the offline compute nodes.
Runs wherever there is internet + `datasets` (login node or an online container). Downloads once, writes:
  {A5DIR}/images/{pid}.png            decoded image per item
  {A5DIR}/testmini.jsonl              one row/item: pid, question, choices, answer, question_type,
                                      answer_type, precision, unit, query  (NO model outputs here)
Then the sweep/prep scripts read these files with HF_HUB_OFFLINE=1 — no runtime download.
Also prints the schema + 3 example items so the scorer (a5_common.py) is written to the real fields.
"""
import os, io, json
from datasets import load_dataset

A5DIR = os.environ.get("A5DIR", "/iopsstor/scratch/cscs/raghavthind/set2_pilot/a5")
os.makedirs(f"{A5DIR}/images", exist_ok=True)

def main():
    ds = load_dataset("AI4Math/MathVista", split="testmini")
    print("columns:", ds.column_names)
    print("n:", len(ds))
    keep = ["pid","question","choices","answer","question_type","answer_type","precision","unit","query"]
    n_img = 0
    with open(f"{A5DIR}/testmini.jsonl","w") as f:
        for r in ds:
            pid = str(r["pid"])
            img = r.get("decoded_image") or r.get("image")
            if img is not None and hasattr(img, "save"):
                img.convert("RGB").save(f"{A5DIR}/images/{pid}.png"); n_img += 1
            row = {k: r.get(k) for k in keep}
            row["has_image"] = bool(img is not None and hasattr(img,"save"))
            f.write(json.dumps(row)+"\n")
    print(f"wrote {A5DIR}/testmini.jsonl  images={n_img}")
    # schema dump for scorer authoring
    print("\n--- answer_type / question_type distribution ---")
    from collections import Counter
    at = Counter(str(r["answer_type"]) for r in ds); qt = Counter(str(r["question_type"]) for r in ds)
    print("answer_type:", dict(at)); print("question_type:", dict(qt))
    print("\n--- 3 example items ---")
    for r in list(ds)[:3]:
        print(json.dumps({k:(str(r.get(k))[:200]) for k in keep}, indent=1))

if __name__ == "__main__":
    main()
