import json, collections, re, sys
sys.path.insert(0, '.')
import tp_common as C

base = f'{C.OUT}/full'
rows = [json.loads(l) for l in open(f'{base}/scored.jsonl')]
gen = {}
for arm in ('T0','T1','T2','T3','A5'):
    g, _ = C.read_jsonl(f'{base}/gen_{arm}.jsonl')
    g, _ = C.dedup_rows(g, lambda r: (r['index'], r['draw']))
    gen[arm] = {(r['index'], r['draw']): r for r in g}

df = C.load_mmstar()
gt = {int(r['index']): r['answer'] for _, r in df.iterrows()}
ch = {int(r['index']): C.parse_choices(r['question']) for _, r in df.iterrows()}
cat = {int(r['index']): r['category'] for _, r in df.iterrows()}
l2 = {int(r['index']): r['l2_category'] for _, r in df.iterrows()}

caps = {}
for r in C.read_jsonl(f'{base}/captions.jsonl')[0]:
    caps.setdefault(r['index'], {})[r['caption_idx']] = r['caption']
capsq = {}
for r in C.read_jsonl(f'{base}/captions_q.jsonl')[0]:
    capsq.setdefault(r['index'], {})[r['caption_idx']] = r['caption']

REASONING = {'logical reasoning','math','science & technology','instance reasoning'}
PERCEPTION = {'coarse perception','fine-grained perception'}

byarm = collections.defaultdict(list)
for r in rows:
    byarm[r['arm']].append(r)

print("="*80)
print("SECTION 1: reasoning length vs correctness, per arm (think_tok)")
print("="*80)
for arm in ('T0','T1','T2','T3'):
    rs = byarm[arm]
    c1 = [r['think_tok'] for r in rs if r['correct']==1 and r.get('think_tok') is not None]
    c0 = [r['think_tok'] for r in rs if r['correct']==0 and r.get('think_tok') is not None]
    import statistics as st
    m1 = st.mean(c1) if c1 else float('nan')
    m0 = st.mean(c0) if c0 else float('nan')
    med1 = st.median(c1) if c1 else float('nan')
    med0 = st.median(c0) if c0 else float('nan')
    # point-biserial-ish correlation: correctness vs think_tok
    all_tok = [r['think_tok'] for r in rs if r.get('think_tok') is not None]
    all_cor = [r['correct'] for r in rs if r.get('think_tok') is not None]
    n = len(all_tok)
    mean_tok = sum(all_tok)/n
    mean_cor = sum(all_cor)/n
    cov = sum((t-mean_tok)*(c-mean_cor) for t,c in zip(all_tok,all_cor))/n
    sd_tok = (sum((t-mean_tok)**2 for t in all_tok)/n)**0.5
    sd_cor = (sum((c-mean_cor)**2 for c in all_cor)/n)**0.5
    r_pb = cov/(sd_tok*sd_cor) if sd_tok>0 and sd_cor>0 else float('nan')
    print(f"{arm}: n_correct={len(c1)} n_wrong={len(c0)}  mean_think_tok(correct)={m1:.1f} mean_think_tok(wrong)={m0:.1f}  "
          f"median(correct)={med1:.0f} median(wrong)={med0:.0f}  point-biserial r={r_pb:+.4f}")

print()
print("="*80)
print("SECTION 2: T3 vs T0 item-level churn on REASONING axis (majority vote across K=3)")
print("="*80)
def majvote(arm, idx):
    draws = [gen[arm].get((idx,d)) for d in range(3)]
    draws = [d for d in draws if d is not None]
    scored = [r for r in byarm[arm] if r['index']==idx]
    if not scored: return None
    s = sum(r['correct'] for r in scored)
    n = len(scored)
    if s*2 > n: return 1
    if s*2 < n: return 0
    return None

ridx = sorted(set(cat) - set())
ridx = [i for i in cat if cat[i] in REASONING]
b_items, c_items = [], []  # b = T3 right, T0 wrong ; c = T3 wrong, T0 right
for i in ridx:
    m3, m0 = majvote('T3', i), majvote('T0', i)
    if m3 is None or m0 is None: continue
    if m3==1 and m0==0: b_items.append(i)
    if m3==0 and m0==1: c_items.append(i)

print(f"T3-fixes (b): {len(b_items)} items | T0-fixes (c, T3 breaks): {len(c_items)} items")

def option_overlap_flags(idx, arm='T3'):
    capset = capsq if arm in ('T3',) else caps
    slots = capset.get(idx, {})
    gold = str(gt[idx]).upper()
    opts = {k: str(v).lower() for k,v in ch.get(idx, {}).items() if str(v).strip()}
    hit_corr = hit_incorr = 0
    for ci, text in slots.items():
        t = text.lower()
        if any(k==gold and len(v)>8 and v in t for k,v in opts.items()): hit_corr += 1
        if any(k!=gold and len(v)>8 and v in t for k,v in opts.items()): hit_incorr += 1
    n = max(len(slots),1)
    return hit_corr/n, hit_incorr/n

print("\n--- caption/correct-option overlap: T3-fixes (b) vs T3-breaks (c) ---")
for label, items in (("T3-fixes(b)", b_items), ("T3-breaks(c)", c_items)):
    co = [option_overlap_flags(i)[0] for i in items]
    ic = [option_overlap_flags(i)[1] for i in items]
    import statistics as st
    print(f"{label}: n={len(items)}  mean_correct-option-overlap={sum(co)/len(co) if co else float('nan'):.3f}  "
          f"mean_incorrect-option-overlap={sum(ic)/len(ic) if ic else float('nan'):.3f}")

print("\n--- category breakdown of b vs c items ---")
bcat = collections.Counter(cat[i] for i in b_items)
ccat = collections.Counter(cat[i] for i in c_items)
for c_ in sorted(REASONING):
    print(f"  {c_:24s} T3-fixes={bcat.get(c_,0):3d}  T3-breaks={ccat.get(c_,0):3d}")

print("\n--- l2_category breakdown (top 8 by combined count) ---")
bl2 = collections.Counter(l2[i] for i in b_items)
cl2 = collections.Counter(l2[i] for i in c_items)
combined = collections.Counter()
for k,v in bl2.items(): combined[k]+=v
for k,v in cl2.items(): combined[k]+=v
for k,_ in combined.most_common(8):
    print(f"  {k:28s} T3-fixes={bl2.get(k,0):3d}  T3-breaks={cl2.get(k,0):3d}")

print("\n--- caption length (chars) for b vs c items ---")
import statistics as st
def cap_len(idx):
    slots = capsq.get(idx, {})
    return st.mean([len(v) for v in slots.values()]) if slots else float('nan')
bl = [cap_len(i) for i in b_items]
cl = [cap_len(i) for i in c_items]
print(f"T3-fixes(b) mean caption len={sum(bl)/len(bl):.0f} chars | T3-breaks(c) mean caption len={sum(cl)/len(cl):.0f} chars")

# sample a few concrete examples of each for qualitative read
print("\n--- 4 concrete T3-fixes examples (T0 wrong, T3 right) ---")
for i in b_items[:4]:
    print(f"\n### item {i} | cat={cat[i]} | l2={l2[i]} | gold={gt[i]}")
    q = str(df[df['index']==i]['question'].values[0])[:200]
    print(f"Q: {q}")
    slot0 = sorted(capsq.get(i,{}).items())[0][1] if capsq.get(i) else "?"
    print(f"T3 caption used: {slot0[:350]}")
    t0text = [gen['T0'].get((i,d)) for d in range(3)]
    t3text = [gen['T3'].get((i,d)) for d in range(3)]
    print(f"T0 answers (post-think tails): {[ (t['text'].split(chr(60)+'/think'+chr(62))[-1] if t and '</think>' in t['text'] else (t['text'][-60:] if t else None))[:60] for t in t0text]}")
    print(f"T3 answers (post-think tails): {[ (t['text'].split(chr(60)+'/think'+chr(62))[-1] if t and '</think>' in t['text'] else (t['text'][-60:] if t else None))[:60] for t in t3text]}")

print("\n--- 4 concrete T3-breaks examples (T0 right, T3 wrong) ---")
for i in c_items[:4]:
    print(f"\n### item {i} | cat={cat[i]} | l2={l2[i]} | gold={gt[i]}")
    q = str(df[df['index']==i]['question'].values[0])[:200]
    print(f"Q: {q}")
    slot0 = sorted(capsq.get(i,{}).items())[0][1] if capsq.get(i) else "?"
    print(f"T3 caption used: {slot0[:350]}")
    t0text = [gen['T0'].get((i,d)) for d in range(3)]
    t3text = [gen['T3'].get((i,d)) for d in range(3)]
    print(f"T0 answers (post-think tails): {[ (t['text'].split(chr(60)+'/think'+chr(62))[-1] if t and '</think>' in t['text'] else (t['text'][-60:] if t else None))[:60] for t in t0text]}")
    print(f"T3 answers (post-think tails): {[ (t['text'].split(chr(60)+'/think'+chr(62))[-1] if t and '</think>' in t['text'] else (t['text'][-60:] if t else None))[:60] for t in t3text]}")

print()
print("="*80)
print("SECTION 3: T1 vs T0 churn (blind), same analysis, for comparison")
print("="*80)
b1, c1 = [], []
for i in cat:
    m1v, m0v = majvote('T1', i), majvote('T0', i)
    if m1v is None or m0v is None: continue
    if m1v==1 and m0v==0: b1.append(i)
    if m1v==0 and m0v==1: c1.append(i)
print(f"T1-fixes (b): {len(b1)} | T1-breaks (c): {len(c1)}  (ALL categories, n={len(cat)})")
def option_overlap_blind(idx):
    slots = caps.get(idx, {})
    gold = str(gt[idx]).upper()
    opts = {k: str(v).lower() for k,v in ch.get(idx, {}).items() if str(v).strip()}
    hit_corr = hit_incorr = 0
    for ci, text in slots.items():
        t = text.lower()
        if any(k==gold and len(v)>8 and v in t for k,v in opts.items()): hit_corr += 1
        if any(k!=gold and len(v)>8 and v in t for k,v in opts.items()): hit_incorr += 1
    n = max(len(slots),1)
    return hit_corr/n, hit_incorr/n
co1 = [option_overlap_blind(i)[0] for i in b1]
ic1 = [option_overlap_blind(i)[1] for i in b1]
co0 = [option_overlap_blind(i)[0] for i in c1]
ic0 = [option_overlap_blind(i)[1] for i in c1]
print(f"T1-fixes: mean_correct-overlap={sum(co1)/len(co1):.3f} mean_incorrect-overlap={sum(ic1)/len(ic1):.3f}")
print(f"T1-breaks: mean_correct-overlap={sum(co0)/len(co0):.3f} mean_incorrect-overlap={sum(ic0)/len(ic0):.3f}")

print()
print("="*80)
print("SECTION 4: think_tok distribution comparison T0 vs T3 (does caption shorten or lengthen reasoning)")
print("="*80)
for arm in ('T0','T1','T2','T3'):
    rs = [r for r in byarm[arm] if r.get('think_tok') is not None]
    toks = sorted(r['think_tok'] for r in rs)
    n = len(toks)
    print(f"{arm}: n={n} mean={sum(toks)/n:.1f} p25={toks[n//4]} median={toks[n//2]} p75={toks[3*n//4]} max={toks[-1]}")

print()
print("="*80)
print("SECTION 5: does caption length correlate with whether T3 beats T0 on that item (all reasoning items)")
print("="*80)
all_r_items = [i for i in ridx if majvote('T3',i) is not None and majvote('T0',i) is not None]
same = [i for i in all_r_items if majvote('T3',i)==majvote('T0',i)]
print(f"same outcome (both right or both wrong): {len(same)} / {len(all_r_items)}")
both_right = [i for i in all_r_items if majvote('T3',i)==1 and majvote('T0',i)==1]
both_wrong = [i for i in all_r_items if majvote('T3',i)==0 and majvote('T0',i)==0]
print(f"both right: {len(both_right)}  both wrong: {len(both_wrong)}  T3-fixes: {len(b_items)}  T3-breaks: {len(c_items)}")
