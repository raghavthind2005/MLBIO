import json, collections, sys
import tp_common as C

base = f'{C.OUT}/full'
rows = [json.loads(l) for l in open(f'{base}/scored.jsonl')]
byarm = collections.defaultdict(list)
for r in rows: byarm[r['arm']].append(r)

print("=== fraction of generations with near-zero thinking (think_tok<=5), and conditional accuracy ===")
for arm in ('T0','T1','T2','T3'):
    rs = [r for r in byarm[arm] if r.get('think_tok') is not None]
    short = [r for r in rs if r['think_tok'] <= 5]
    long_ = [r for r in rs if r['think_tok'] > 5]
    frac_short = len(short)/len(rs)
    acc_short = sum(r['correct'] for r in short)/len(short) if short else float('nan')
    acc_long = sum(r['correct'] for r in long_)/len(long_) if long_ else float('nan')
    print(f"{arm}: n={len(rs)} frac_think_tok<=5={frac_short:.3f} ({len(short)}/{len(rs)})  "
          f"acc|short={acc_short:.4f}  acc|long={acc_long:.4f}  acc_gap(long-short)={acc_long-acc_short:+.4f}")

print()
print("=== T1/T2 specifically: what does a think_tok<=5 generation actually look like? (raw sample) ===")
gen = {}
for arm in ('T1','T2'):
    g, _ = C.read_jsonl(f'{base}/gen_{arm}.jsonl')
    g, _ = C.dedup_rows(g, lambda r: (r['index'], r['draw']))
    gen[arm] = g
for arm in ('T1','T2'):
    ex = [r for r in gen[arm] if r.get('think_tok') is not None and r['think_tok']<=5][:2]
    for r in ex:
        print(f"--- {arm} idx={r['index']} draw={r['draw']} think_tok={r['think_tok']} ---")
        print(repr(r['text'][:300]))
