#!/usr/bin/env python3
"""
Canonical Track-T pool + TD-VI delta extractor (frozen per TRACK_T_T1_RECORD.md).
Imported by the placebo builder, the sweep, and analysis so all stages share ONE definition.

  * extract_delta(td_q, vi_q): the relaxed round-trip-pure TD-VI delta (T1 §5), or None if the
    item is empty / impure. Distinct removed spans are joined with '; ' for a readable payload.
  * pool_items(by): the 559 pool (round-trip-pure minus the 8 reasoning/directive leaks, T1 §9b).
"""
import re, difflib, string, json
from collections import defaultdict

PUNC = string.punctuation + ' '
EXCL_REASONING = {'132', '361', '366', '403', '413', '547', '598', '683'}   # T1 §9b

def stem(q):
    return re.split(r'\n\s*Choices?\s*:', q, maxsplit=1)[0].rstrip()

def _n1(t):
    return t.strip(PUNC).lower()

def extract_delta(td_q, vi_q):
    a = stem(td_q).split(); b = stem(vi_q).split()
    if a == b:
        return None
    na = [_n1(t) for t in a]; nb = [_n1(t) for t in b]
    ops = difflib.SequenceMatcher(None, na, nb, autojunk=False).get_opcodes()
    if {t for t, *_ in ops} - {'equal', 'delete'}:
        return None
    kept = [a[k] for t, i1, i2, j1, j2 in ops if t == 'equal' for k in range(i1, i2)]
    if [x for x in (_n1(t) for t in kept) if x] != [x for x in nb if x]:
        return None
    spans = [' '.join(a[i1:i2]).strip(PUNC)
             for t, i1, i2, j1, j2 in ops if t == 'delete' and any(_n1(a[k]) for k in range(i1, i2))]
    return '; '.join(s for s in spans if s)

def load(path):
    by = defaultdict(dict)
    for r in json.load(open(path)):
        by[r['problem_index']][r['problem_version']] = r
    return by

def pool_items(by):
    out = []
    for pi in by:
        if pi in EXCL_REASONING:
            continue
        if extract_delta(by[pi]['Text Dominant']['question'], by[pi]['Vision Intensive']['question']) is not None:
            out.append(pi)
    return sorted(out, key=int)

def deltas(by, pids=None):
    if pids is None:
        pids = pool_items(by)
    return {pi: extract_delta(by[pi]['Text Dominant']['question'], by[pi]['Vision Intensive']['question'])
            for pi in pids}
