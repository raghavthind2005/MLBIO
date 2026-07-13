#!/usr/bin/env python3
"""
MathVerse rule-based scorer — Track T (Diagnostic Signal Study).

Scoring is deliberately restricted to answer forms that can be matched with ZERO ambiguity:
  * multi-choice : model emits \\boxed{<letter>}; compare to the GT option letter.
                   An item is SCORABLE iff its GT answer is a single option letter [A-F]
                   that appears in the item's Choices block (drops malformed GT: 'False',
                   two-letter 'C\\nD', bearing-as-mc, etc.).
  * free-form    : NUMERIC-SCALAR only. model emits \\boxed{<value>}; reduce to a float via
                   to_number() (units / degree / bearing / \\sqrt / \\frac / \\pi / label=value),
                   compare to to_number(GT) with tolerance. Non-scalar GT (tuples, intervals,
                   domain/range, function defs, EQUATIONS, prose) -> UNSCORABLE (dropped).

Frozen scoring parameters: num tolerance rtol=0.01, atol=0.05.

Run as __main__ (local): self-tests on testmini.json — GT recognition, format-robustness,
counts, and a discordant audit sample. Import elsewhere for score_mc / score_ff.
"""
import os, re, math, json, difflib, string
from collections import defaultdict, Counter

NUM_RTOL, NUM_ATOL = 0.01, 0.05

# ---------------- \boxed extractor (balanced braces, last box) ----------------
def extract_boxed(text):
    out = None; i = 0
    while True:
        j = text.find('\\boxed', i)
        if j < 0: break
        k = text.find('{', j)
        if k < 0: break
        depth = 0; buf = []
        for c in text[k:]:
            if c == '{': depth += 1
            elif c == '}': depth -= 1
            if not (c == '{' and depth == 1) and not (c == '}' and depth == 0):
                buf.append(c)
            if depth == 0: break
        out = ''.join(buf); i = j + 6
    return out

# ---------------- numeric-scalar reduction ----------------
_STRUCT = re.compile(
    r',|\\cup|\\infty|\[|\bdomain\b|\brange\b|increasing|decreasing|constant|local|'
    r'maximum|minimum|\beven\b|\bodd\b|when|every|complete|center|bunny|'
    r'\bf\s*\(|\bg\s*\(|\bh\s*\(|<|>|\\leq|\\geq|\\le\b|\\ge\b', re.I)

def _clean(a):
    s = str(a).strip().replace('$', '').replace('\\[', '').replace('\\]', '')
    s = re.sub(r'\\(?:text|mathrm|mbox|operatorname)\s*\{[^{}]*\}', ' ', s)
    s = re.sub(r'\^?\s*\{?\s*\\?circ\s*\}?', '', s)
    s = s.replace('°', '').replace('掳', '').replace('\\%', '').replace('%', '')
    s = re.sub(r'\^\s*\{?\s*[23]\s*\}?', '', s)                    # unit exponents cm^2, m^{3}
    s = re.sub(r'\bunits?\b|\bmetres?\b|\bmeters?\b|\bkm\b|\bmm\b|\bcm\b|\blitres?\b|\bh\b|\bm\b|~',
               ' ', s, flags=re.I)
    return s.strip()

def to_number(a):
    """Reduce a scalar answer to float, or None if not a single scalar."""
    if a is None: return None
    s = _clean(a)
    if _STRUCT.search(s): return None
    if s.count('=') == 1:                                          # label=value, but not an equation
        lhs, rhs = s.split('=', 1)
        if not re.fullmatch(r'[A-Za-z][A-Za-z ]*', lhs.strip()):   # lhs must be a pure label word
            return None
        s = rhs.strip()
    if _STRUCT.search(s) or not s: return None
    s = s.replace('\\left', '').replace('\\right', '')
    s = re.sub(r'\\sqrt\s*\{([^{}]*)\}', r'sqrt(\1)', s)           # sqrt BEFORE frac (nesting)
    s = re.sub(r'\\sqrt\s*(\d+)', r'sqrt(\1)', s)
    for _ in range(3):
        s = re.sub(r'\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'((\1)/(\2))', s)
    s = s.replace('\\pi', 'pi').replace('\\cdot', '*').replace('\\times', '*').replace('^', '**')
    s = re.sub(r'\b[NSEWT]\b', '', s)                              # bearing letters
    s = re.sub(r'(?<=\d)\s*[NSEWT]\s*$', '', s)
    s = s.replace('\\', ' ').strip()
    s = re.sub(r'(\d)\s*(sqrt|pi|\()', r'\1*\2', s)                # implicit multiplication
    s = re.sub(r'(pi|\))\s*(sqrt|pi|\()', r'\1*\2', s)
    s = re.sub(r'\)\s*(\d)', r')*\1', s)
    if re.search(r'[a-zA-Z]', re.sub(r'sqrt|pi', '', s)): return None
    try:
        return float(eval(s, {'__builtins__': {}}, {'sqrt': math.sqrt, 'pi': math.pi}))
    except Exception:
        return None

def num_match(a, b, rtol=NUM_RTOL, atol=NUM_ATOL):
    return abs(a - b) <= max(atol, rtol * abs(b))

# ---------------- multi-choice ----------------
def option_letters(choices_block):
    out = set()
    for ln in (choices_block or '').splitlines():
        m = re.match(r'\s*\(?([A-F])\)?\s*[:.]', ln)
        if m: out.add(m.group(1))
    return out

def gt_letter(ans):
    s = str(ans).strip().strip('()').strip()
    return s if re.fullmatch(r'[A-F]', s) else None

def mc_scorable(gt_ans, choices_block):
    L = gt_letter(gt_ans)
    return L is not None and L in option_letters(choices_block)

# explicit answer-cue fallback (only used when no letter is in \boxed{}); deterministic, no bare-letter scan
_ANS_CUE = re.compile(r'(?:[Aa]nswer|[Cc]orrect|[Cc]hoice)[^A-F\n]{0,15}([A-F])\b')

def score_mc(model_out, gt_ans):
    g = gt_letter(gt_ans)
    if g is None:
        return False
    b = extract_boxed(model_out)
    if b is not None:                                   # 1) letter inside \boxed{}
        m = re.search(r'([A-F])', b.strip().strip('()'))
        if m:
            return m.group(1) == g
    ms = _ANS_CUE.findall(model_out)                    # 2) explicit "answer/correct/choice ... X" (last)
    if ms:
        return ms[-1] == g
    return False                                        # 3) unparseable -> wrong (conservative; rate tracked in smoke)

# ---------------- free-form (numeric) ----------------
def ff_scorable(gt_ans):
    return to_number(gt_ans) is not None

def score_ff(model_out, gt_ans):
    gv = to_number(gt_ans)
    if gv is None: return None                                    # unscorable item
    b = extract_boxed(model_out)
    pv = to_number(b if b is not None else model_out)
    if pv is None: return False
    return num_match(pv, gv)

# ==================== self-test ====================
def _split_choices(q):
    m = re.split(r'\n\s*Choices?\s*:', q, maxsplit=1)
    return (m[0].rstrip(), (m[1] if len(m) > 1 else None))

def _n1(t): return t.strip(string.punctuation + ' ').lower()
def _is_pure(td, vi):
    a = _split_choices(td)[0].split(); b = _split_choices(vi)[0].split()
    if a == b: return False
    na = [_n1(t) for t in a]; nb = [_n1(t) for t in b]
    ops = difflib.SequenceMatcher(None, na, nb, autojunk=False).get_opcodes()
    if {t for t, *_ in ops} - {'equal', 'delete'}: return False
    kept = [a[k] for t, i1, i2, j1, j2 in ops if t == 'equal' for k in range(i1, i2)]
    return [x for x in (_n1(t) for t in kept) if x] == [x for x in nb if x]

def _main():
    path = os.environ.get('TESTMINI', '/tmp/mathverse_testmini.json')
    d = json.load(open(path))
    by = defaultdict(dict)
    for r in d: by[r['problem_index']][r['problem_version']] = r
    EXCL = {'132', '361', '366', '403', '413', '547', '598', '683'}   # reasoning/directive leaks (T1 §9b)
    pool = [pi for pi in by if pi not in EXCL and
            _is_pure(by[pi]['Text Dominant']['question'], by[pi]['Vision Intensive']['question'])]
    mc = [pi for pi in pool if by[pi]['Vision Intensive']['question_type'] == 'multi-choice']
    ff = [pi for pi in pool if by[pi]['Vision Intensive']['question_type'] == 'free-form']

    def V(pi): return by[pi]['Vision Intensive']
    def choices(pi): return _split_choices(V(pi)['question'])[1] or ''

    mc_ok = [pi for pi in mc if mc_scorable(V(pi)['answer'], choices(pi))]
    mc_drop = [pi for pi in mc if pi not in set(mc_ok)]
    ff_ok = [pi for pi in ff if ff_scorable(V(pi)['answer'])]
    ff_drop = [pi for pi in ff if pi not in set(ff_ok)]

    print(f'pool(559) = mc {len(mc)} + ff {len(ff)}')
    print(f'SCORABLE  = mc {len(mc_ok)} (drop {len(mc_drop)}) + ff {len(ff_ok)} (drop {len(ff_drop)})'
          f'  => scored pool = {len(mc_ok)+len(ff_ok)}')

    # self-test 1: GT recognition (model emits \boxed{GT})
    r1 = sum(score_mc('...\\boxed{' + V(pi)['answer'] + '}', V(pi)['answer']) for pi in mc_ok)
    print(f'  self-test mc  GT-recognition : {r1}/{len(mc_ok)}')
    r2 = sum(bool(score_ff('...\\boxed{' + V(pi)['answer'] + '}', V(pi)['answer'])) for pi in ff_ok)
    print(f'  self-test ff  GT-recognition : {r2}/{len(ff_ok)}')

    # self-test 2: format robustness — value / rounded / +unit renderings all score correct
    def robust(pi):
        v = to_number(V(pi)['answer'])
        outs = [f'\\boxed{{{v}}}', f'\\boxed{{{round(v,2)}}}', f'\\boxed{{{round(v,1)}}}',
                f'the answer is \\boxed{{{round(v,2)}}} cm', f'\\boxed{{{V(pi)["answer"]}}}']
        return all(score_ff(o, V(pi)['answer']) for o in outs)
    r3 = sum(robust(pi) for pi in ff_ok)
    print(f'  self-test ff  format-robustness: {r3}/{len(ff_ok)}')
    for pi in ff_ok:
        if not robust(pi): print('     ROBUST-FAIL', repr(V(pi)['answer']))

    # ---- ADVERSARIAL SECURITY: false-positives must be ZERO ----
    def _chb(pi): return _split_choices(V(pi)['question'])[1] or ''
    mc_fp = sum(score_mc('reasoning \\boxed{' + w + '}', V(pi)['answer'])
                for pi in mc_ok for w in option_letters(_chb(pi))
                if w != gt_letter(V(pi)['answer']))
    mc_tp = sum(score_mc('Reasoning about angle A, line B. \\boxed{' + gt_letter(V(pi)['answer']) + '}',
                         V(pi)['answer']) for pi in mc_ok)
    def _wrong(v): return v * 1.5 + 3.0 if abs(v * 0.5 + 3.0) > max(NUM_ATOL, NUM_RTOL * abs(v)) else v + 10.0
    ff_fp = sum(bool(score_ff('\\boxed{' + str(round(_wrong(to_number(V(pi)['answer'])), 3)) + '}', V(pi)['answer']))
                for pi in ff_ok)
    ff_tp = sum(bool(score_ff('reasoning ... \\boxed{' + str(round(to_number(V(pi)['answer']), 2)) + '}', V(pi)['answer']))
                for pi in ff_ok)
    print(f'  adversarial mc: wrong-letter FP={mc_fp} (want 0)  reasoning-wrapper TP={mc_tp}/{len(mc_ok)}')
    print(f'  adversarial ff: wrong-number  FP={ff_fp} (want 0)  reasoning-wrapper TP={ff_tp}/{len(ff_ok)}')
    assert mc_fp == 0 and ff_fp == 0, 'FALSE-POSITIVE DETECTED — scorer is unsafe'

    print('\n  mc dropped (malformed GT):', [repr(V(pi)['answer']) for pi in mc_drop])
    print('\n  ff dropped (non-scalar) count:', len(ff_drop))

if __name__ == '__main__':
    _main()
