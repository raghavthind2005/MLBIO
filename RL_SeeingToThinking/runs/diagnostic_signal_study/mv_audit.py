#!/usr/bin/env python3
"""
Track-T integrity audit — independent re-verification of every load-bearing claim before freeze.
Run: TESTMINI=/path python3 mv_audit.py    (exits non-zero on any FAIL).
"""
import os, re, sys, hashlib, difflib, string, subprocess
from collections import Counter, defaultdict
import mv_pool, mv_score, mv_placebo

TESTMINI = os.environ.get('TESTMINI', '/tmp/mathverse_testmini.json')
PUNC = string.punctuation + ' '
fails = []
def check(name, ok, detail=''):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  — {detail}' if detail else ''))
    if not ok: fails.append(name)

def ntok(s): return [t for t in (w.strip(PUNC).lower() for w in s.split()) if t]
def stem(q): return re.split(r'\n\s*Choices?\s*:', q, maxsplit=1)[0].rstrip()
def choices(q):
    m = re.split(r'\n\s*Choices?\s*:', q, maxsplit=1); return m[1] if len(m) > 1 else ''

by = mv_pool.load(TESTMINI)
def V(pi): return by[pi]['Vision Intensive']
def TD(pi): return by[pi]['Text Dominant']['question']

print('== 0. data provenance ==')
h = hashlib.sha256(open(TESTMINI, 'rb').read()).hexdigest()
check('testmini sha256 matches T1', h == 'f4ce9b18d111b23d5950dcbc8f377c6a05955a458db6a3103aec706fa63b0e9b', h[:16])

print('== 1. pool definition & cross-module consistency ==')
pool = mv_pool.pool_items(by)
check('pool size == 559', len(pool) == 559, str(len(pool)))
check('8 reasoning/directive leaks excluded', all(pi not in pool for pi in mv_pool.EXCL_REASONING))
# independent purity via mv_score._is_pure over ALL items minus EXCL
indep = [pi for pi in by if pi not in mv_pool.EXCL_REASONING
         and mv_score._is_pure(TD(pi), V(pi)['question'])]
check('mv_pool.pool_items == mv_score._is_pure set', set(pool) == set(indep),
      f'pool={len(pool)} indep={len(indep)} sym-diff={set(pool)^set(indep)}')

print('== 2. delta completeness (independent: TD-tokens − VI-tokens == delta-tokens) ==')
dl = mv_pool.deltas(by, pool)
bad = []
for pi in pool:
    removed = Counter(ntok(stem(TD(pi)))) - Counter(ntok(stem(V(pi)['question'])))
    got = Counter(ntok(dl[pi]))
    if removed != got: bad.append(pi)
check('every delta == exactly TD−VI removed tokens', not bad, f'{len(bad)} mismatched: {bad[:6]}')
check('no empty delta', all(dl[pi].strip() for pi in pool))

print('== 3. perception-only re-screen on the 559 (must be 0 reasoning markers) ==')
MARK = re.compile(r'\btherefore\b|\bthus\b|\bhence\b|\bconsequently\b|\bimpl(y|ies)\b|it follows|'
                  r'we (get|have|obtain|can)|can be (obtained|calculated|derived|computed)|deduce|'
                  r'conclude|as a result|\bfind\b|\bcalculate\b|\bsolve\b|\bprove\b|\bdetermine\b|'
                  r'\bbecause\b|\bsince\b', re.I)
hits = [pi for pi in pool if MARK.search(dl[pi])]
check('0 reasoning/solution markers in pool deltas', not hits, f'hits={hits}')

print('== 4. answer-leakage re-screen (automated; genuine leaks must be 0) ==')
def norm_val(s): return re.sub(r'\s', '', str(s).strip().lower().rstrip('.'))
def opt_text(chb, letter):
    for ln in chb.splitlines():
        m = re.match(r'\s*\(?' + re.escape(letter) + r'\)?\s*[:.]\s*(.+)', ln)
        if m: return m.group(1).strip()
flagged = []
for pi in pool:
    a = V(pi)['answer']; nd = norm_val(dl[pi])
    if V(pi)['question_type'] == 'free-form':
        tgt = norm_val(a)
    else:
        ct = opt_text(choices(V(pi)['question']), str(a).strip().strip('()')); tgt = norm_val(ct) if ct else ''
    if tgt and len(tgt) >= 2 and tgt in nd: flagged.append((pi, a, dl[pi][:50]))
check('answer-leak flags == 4 known coincidences (234/268/421/537)',
      sorted(p for p, *_ in flagged) == ['234', '268', '421', '537'], f'{[p for p,*_ in flagged]}')

print('== 5. scored pool consistency (mv_placebo vs recomputed) ==')
sp = mv_placebo.scored_pool(by)
mc = [pi for pi in pool if V(pi)['question_type'] == 'multi-choice' and mv_score.mc_scorable(V(pi)['answer'], choices(V(pi)['question']))]
ff = [pi for pi in pool if V(pi)['question_type'] == 'free-form' and mv_score.ff_scorable(V(pi)['answer'])]
check('scored pool == 497', len(sp) == 497, str(len(sp)))
check('scored == mc(373)+ff(124) recomputed', set(sp) == set(mc + ff) and len(mc) == 373 and len(ff) == 124,
      f'mc={len(mc)} ff={len(ff)} sp={len(sp)}')

print('== 6. placebo determinism & properties ==')
d1 = mv_placebo.assign(sp, {pi: dl.get(pi) or mv_pool.extract_delta(TD(pi), V(pi)['question']) for pi in sp})
d2 = mv_placebo.assign(sp, {pi: dl.get(pi) or mv_pool.extract_delta(TD(pi), V(pi)['question']) for pi in sp})
check('placebo assignment deterministic', d1 == d2)
check('placebo no self-assignment', all(d1[p] != p for p in sp))
alld = {pi: (dl.get(pi) or mv_pool.extract_delta(TD(pi), V(pi)['question'])) for pi in sp}
def jac(a, b):
    A, B = set(re.findall(r'[a-z0-9]+', a.lower())), set(re.findall(r'[a-z0-9]+', b.lower()))
    u = A | B; return len(A & B) / len(u) if u else 0.0
maxj = max(jac(alld[p], alld[d1[p]]) for p in sp)
check('placebo content Jaccard max < 0.60', maxj < 0.60, f'max={maxj:.3f}')

print('== 7. scorer self-test + adversarial (via mv_score, assert-guarded) ==')
r = subprocess.run([sys.executable, 'mv_score.py'], capture_output=True, text=True,
                   env={**os.environ, 'TESTMINI': TESTMINI})
ok = r.returncode == 0 and 'FP=0' in r.stdout and 'wrong-number  FP=0' in r.stdout
check('mv_score.py self-test+adversarial pass (0 FP)', ok, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[:120])

print('\n' + ('ALL CHECKS PASSED' if not fails else f'{len(fails)} FAILURE(S): {fails}'))
sys.exit(1 if fails else 0)
