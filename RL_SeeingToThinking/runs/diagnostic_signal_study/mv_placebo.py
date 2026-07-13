#!/usr/bin/env python3
"""
Wrong-item placebo assignment for Track T.

Each target item is assigned a donor item (from the same scored pool) whose delta is:
  * LENGTH-matched   -> so the placebo payload is ~the same size as the item's own privileged delta,
  * CONTENT-mismatched -> a DIFFERENT figure's givens (never near-identical to the target's).

Deterministic: sort by delta length, walk nearest-length offsets, take the first donor with token
Jaccard < THRESH (avoids accidentally-applicable near-duplicates). Prints a validation report;
with --write, emits placebo_assignment.json (frozen).
"""
import os, re, sys, json, statistics
import mv_pool, mv_score

TESTMINI = os.environ.get('TESTMINI', '/tmp/mathverse_testmini.json')
THRESH = 0.6   # max allowed content-token Jaccard between an item and its placebo donor

def content_tokens(delta):
    return set(re.findall(r'[a-z0-9]+', delta.lower()))

def jac(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0

def scored_pool(by):
    pool = mv_pool.pool_items(by)
    def V(pi): return by[pi]['Vision Intensive']
    def chb(pi): return re.split(r'\n\s*Choices?\s*:', V(pi)['question'], maxsplit=1)
    def choices(pi):
        m = chb(pi); return m[1] if len(m) > 1 else ''
    mc = [pi for pi in pool if V(pi)['question_type'] == 'multi-choice'
          and mv_score.mc_scorable(V(pi)['answer'], choices(pi))]
    ff = [pi for pi in pool if V(pi)['question_type'] == 'free-form'
          and mv_score.ff_scorable(V(pi)['answer'])]
    return sorted(mc + ff, key=int)

def assign(pids, dlt):
    order = sorted(pids, key=lambda p: (len(dlt[p]), int(p)))
    n = len(order); pos = {p: k for k, p in enumerate(order)}
    tok = {p: content_tokens(dlt[p]) for p in pids}
    offsets = [o for k in range(1, n) for o in (k, -k)]   # 1,-1,2,-2,... nearest length first
    donor = {}
    for p in pids:
        k = pos[p]; chosen = None
        for off in offsets:
            q = order[(k + off) % n]
            if q != p and jac(tok[p], tok[q]) < THRESH:
                chosen = q; break
        donor[p] = chosen if chosen else order[(k + 1) % n]
    return donor

def main():
    by = mv_pool.load(TESTMINI)
    pids = scored_pool(by)
    dlt = mv_pool.deltas(by, pids)
    donor = assign(pids, dlt)
    tok = {p: content_tokens(dlt[p]) for p in pids}

    self_asg = sum(donor[p] == p for p in pids)
    ldiff = [abs(len(dlt[p]) - len(dlt[donor[p]])) for p in pids]
    jacs = [jac(tok[p], tok[donor[p]]) for p in pids]
    over = [(p, donor[p], round(j, 2)) for p, j in zip(pids, jacs) if j >= THRESH]

    print(f'scored pool = {len(pids)}   placebo assigned = {len(donor)}')
    print(f'self-assignments (must be 0): {self_asg}')
    print(f'length |Δchars|: mean={statistics.mean(ldiff):.1f} median={statistics.median(ldiff)} '
          f'p90={sorted(ldiff)[int(0.9*len(ldiff))]} max={max(ldiff)}')
    print(f'content Jaccard (item vs its donor): mean={statistics.mean(jacs):.3f} '
          f'p90={sorted(jacs)[int(0.9*len(jacs))]:.3f} max={max(jacs):.3f}   over-threshold({THRESH}): {len(over)}')
    print(f'donor reuse: {len(set(donor.values()))} distinct donors for {len(pids)} items')
    print('\nexamples (target delta  ||  placebo donor delta):')
    for p in pids[:4]:
        print(f'  [{p}] {dlt[p][:80]!r}\n        <- [{donor[p]}] {dlt[donor[p]][:80]!r}  (jac={jac(tok[p],tok[donor[p]]):.2f}, Δlen={abs(len(dlt[p])-len(dlt[donor[p]]))})')
    if over:
        print('\nOVER-THRESHOLD pairs (inspect):')
        for p, q, j in over[:10]:
            print(f'  [{p}] {dlt[p][:60]!r}  ~~  [{q}] {dlt[q][:60]!r}  jac={j}')

    if '--write' in sys.argv:
        json.dump({p: donor[p] for p in pids}, open('placebo_assignment.json', 'w'), indent=0)
        print('\nwrote placebo_assignment.json')

if __name__ == '__main__':
    main()
