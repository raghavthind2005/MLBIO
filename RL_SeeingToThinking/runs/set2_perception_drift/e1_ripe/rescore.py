#!/usr/bin/env python3
"""Re-score v2 records OFFLINE (login node, stdlib only) — fixes \\boxed{\\text{...}}
false negatives WITHOUT re-running (uses the saved full_text). Proves the value of
capturing everything. Reports true accuracy, by-depth, artifact flips, and the clean
E1 error pool (items genuinely wrong)."""
import json, re
from collections import defaultdict

REC = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out/v2_full_records.jsonl"

def extract_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx < 0: return None
    i, depth, buf = idx + 7, 1, []
    while i < len(text) and depth:
        ch = text[i]
        if ch == "{": depth += 1
        elif ch == "}": depth -= 1
        if depth: buf.append(ch)
        i += 1
    return "".join(buf)

def clean(s):
    if not s: return ""
    s = re.sub(r"\\(?:text|mathrm|mathbf|textbf|mathsf|textit)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\", " ")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def canon(s):
    """map CLEVR visual synonyms to canonical answer terms (rubber/metal, large/small)."""
    p = clean(s)
    if re.search(r"\bmatte\b", p) or "non metal" in p or "not metal" in p: return "rubber"
    if re.search(r"\b(metallic|shiny)\b", p): return "metal"
    p = re.sub(r"\bbig\b", "large", p); p = re.sub(r"\b(tiny|little)\b", "small", p)
    return p

def ok(pred, gt):
    p = canon(pred)
    return bool(gt) and (p == gt or gt in p.split() or p in gt.split())

CATS = {**{str(i): "count" for i in range(21)}, "yes": "bool", "no": "bool",
        "rubber": "material", "metal": "material", "large": "size", "small": "size",
        "cube": "shape", "sphere": "shape", "cylinder": "shape"}
def category(gt): return CATS.get(gt, "color")

rows = [json.loads(l) for l in open(REC)]
n = len(rows)
old = sum(r["parsed"]["correct"] for r in rows)
by = defaultdict(lambda: [0, 0]); flips = 0; errors = []; catn = defaultdict(int)
CLEAN = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out/v2_error_pool.jsonl"
fclean = open(CLEAN, "w")
for r in rows:
    b = extract_boxed(r["full_text"]); good = ok(b, r["gt_norm"])
    by[r["level"]][0] += good; by[r["level"]][1] += 1
    if good and not r["parsed"]["correct"]: flips += 1
    if not good:
        cat = category(r["gt_norm"]); catn[cat] += 1; ctk = r["parsed"]["chain_tok"]
        errors.append((r["qi"], r["depth"], cat, r["gt_norm"], canon(b), ctk, r["question"]))
        fclean.write(json.dumps(dict(qi=r["qi"], depth=r["depth"], category=cat, gt=r["gt_norm"],
                                     pred=canon(b), chain_tok=ctk, question=r["question"])) + "\n")
fclean.close()
new = sum(v[0] for v in by.values())

print(f"records: {n}")
print(f"OLD accuracy (script)       = {old}/{n} = {old/n:.3f}")
print(f"NEW accuracy (+synonyms)    = {new}/{n} = {new/n:.3f}   (recovered {flips + (new-old-flips)} false negatives total)")
print("by level (fixed):")
for lv in sorted(by):
    c, t = by[lv]; print(f"  L{lv}: {c}/{t} = {c/t:.2f}")
print(f"\nTRUE error pool = {len(errors)} items;  by type: {dict(catn)}")
for qi, d, cat, gt, pred, ctk, q in sorted(errors, key=lambda x: (x[2], x[1])):
    print(f"  qi={qi:>6} d={d:>2} {cat:<9} chain={ctk:>6} gt={gt!r:>8} pred={pred!r:>12} | {q[:52]}")
print(f"\nclean error pool -> {CLEAN}")
