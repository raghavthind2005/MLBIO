"""
Pass 4 — paired analysis. Mirrors Track-T's mv_analyze: continuous paired delta of per-item p-hat
with a 10k TWO-LEVEL bootstrap (resample items, then resample draws within item, so decode
variance enters the CI), plus exact McNemar on the majority-vote binarisation, Holm across the
pre-registered family.

Pre-registered contrast family, THINKING-ONLY (3), amended 2026-08-06:
    T1-T0, T1-T2, T2-T0
Instruct (I0-I3) was dropped from the full-scale run: the smoke measured an arm-dependent,
payload-length-correlated non-convergence rate (unextract_rate spread 0.271; I0 0.021 < I1 0.090
< I3 0.146, scaling with payload complexity, unchanged in character even after raising max_tokens)
that is a real confound, not a metric artifact -- see submit_full.sh's SCOPE note. The ORIGINAL
7-contrast family (T1-T0, I1-I0, INTERACTION, T1-T2, I1-I2, T2-T0, I2-I0) and the INTERACTION
bootstrap in run() are left in place, unused by FAMILY_A now, so the already-collected Instruct
SMOKE data (n=48) can still be reported as an explicitly-labelled exploratory footnote if wanted.
A5 is DESCRIPTIVE ONLY and is not in the test family (Q4).

Reported separately, pre-specified, not mined: the 500 perception items
(coarse perception + fine-grained perception).

  python tp_pass4_analyze.py [--tag smoke] [--boot 10000]
"""
import argparse, collections, json, math, random
import tp_common as C

# TWO pre-registered families, Holm-corrected SEPARATELY. They answer different questions, and
# folding the T3 contrasts into family A would weaken the original endpoints under Holm for no
# scientific reason. Both were fixed pre-outcome (amended 2026-08-06 to drop Instruct, also
# pre-outcome relative to the full-scale run -- no full-scale generation has happened yet).
#   A: does a strong QUESTION-BLIND articulation help?  (the original probe)
#   B: does TARGETING the articulation at the question help, and help more than blind?
FAMILY_A = [("T1", "T0"), ("T1", "T2"), ("T2", "T0")]
FAMILY_B = [("T3", "T0"), ("T3", "T1")]
PERCEPTION = C.PERCEPTION_CATS
REASONING = C.REASONING_CATS


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def holm(pv):
    items = sorted(pv.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k], prev = adj, adj
    return out


def load(tag, metric="correct"):
    """metric: 'correct' (PRIMARY, boxed letter) | 'correct_tolerant' | 'correct_official'."""
    per = collections.defaultdict(dict)   # arm -> index -> [0/1 per draw]
    cat = {}
    rows, _ = C.read_jsonl(f"{C.OUT}/{tag}/scored.jsonl")
    for r in rows:
        per[r["arm"]].setdefault(r["index"], []).append(r[metric])
        cat[r["index"]] = r["category"]
    return per, cat


def phat(d):
    return {i: sum(v) / len(v) for i, v in d.items()}


def paired(a_draws, b_draws, idxs, boot, rng):
    """Delta of per-item p-hat + two-level bootstrap CI + exact McNemar on majority vote."""
    pa, pb = phat(a_draws), phat(b_draws)
    diffs = [pa[i] - pb[i] for i in idxs]
    delta = sum(diffs) / len(diffs)
    lo_hi = []
    for _ in range(boot):
        samp = [rng.choice(idxs) for _ in idxs]                    # level 1: items
        tot = 0.0
        for i in samp:                                             # level 2: draws within item
            da, db = a_draws[i], b_draws[i]
            ra = sum(rng.choice(da) for _ in da) / len(da)
            rb = sum(rng.choice(db) for _ in db) / len(db)
            tot += ra - rb
        lo_hi.append(tot / len(samp))
    lo_hi.sort()
    lo, hi = lo_hi[int(.025 * boot)], lo_hi[int(.975 * boot) - 1]
    maj = lambda v: 1 if sum(v) * 2 > len(v) else (0 if sum(v) * 2 < len(v) else None)
    b = c = 0
    for i in idxs:
        ma, mb = maj(a_draws[i]), maj(b_draws[i])
        if ma is None or mb is None:
            continue
        b += (ma == 1 and mb == 0)
        c += (ma == 0 and mb == 1)
    return dict(delta=delta, ci=(lo, hi), b=b, c=c, p=mcnemar_exact(b, c), n=len(idxs))


def run(per, idxs, boot, seed=0, family=None):
    rng = random.Random(seed)
    res, pv = {}, {}
    for a, b in (family or FAMILY_A):
        if a == "INTERACTION":
            need = ["T0", "T1", "I0", "I1"]
            if any(k not in per for k in need):
                continue
            m = {k: phat(per[k]) for k in need}
            # per-item interaction: (I1 - I0) - (T1 - T0)
            d = {i: (m["I1"][i] - m["I0"][i]) - (m["T1"][i] - m["T0"][i]) for i in idxs}
            delta = sum(d.values()) / len(d)
            # BUG 7: this bootstrap used to resample ITEMS ONLY, using fixed per-item point
            # estimates -- so the interaction CI ignored decode variance while every other
            # contrast included it. Inconsistent and anti-conservative. Now two-level, like the rest.
            bs = []
            for _ in range(boot):
                s = [rng.choice(idxs) for _ in idxs]
                tot = 0.0
                for i in s:
                    r = {}
                    for k in need:
                        dr = per[k][i]
                        r[k] = sum(rng.choice(dr) for _ in dr) / len(dr)
                    tot += (r["I1"] - r["I0"]) - (r["T1"] - r["T0"])
                bs.append(tot / len(s))
            bs.sort()
            lo, hi = bs[int(.025 * boot)], bs[int(.975 * boot) - 1]
            # bootstrap two-sided p for H0: interaction == 0 (so it can enter the Holm family)
            frac = sum(1 for v in bs if v <= 0) / len(bs)
            bp = min(1.0, 2 * min(frac, 1 - frac))
            res["INTERACTION"] = dict(delta=delta, ci=(lo, hi), b=None, c=None, p=bp,
                                      p_kind="bootstrap", n=len(idxs))
            pv["INTERACTION"] = bp
            continue
        if a not in per or b not in per:
            continue
        r = paired(per[a], per[b], idxs, boot, rng)
        res[f"{a}-{b}"] = r
        pv[f"{a}-{b}"] = r["p"]
    for k, v in holm(pv).items():
        res[k]["holm"] = v
    return res


def show(title, res, per, idxs):
    print(f"\n===== {title}  (n={len(idxs)}) =====")
    for arm in sorted(per):
        p = phat({i: per[arm][i] for i in idxs if i in per[arm]})
        if p:
            print(f"  {arm:12s} acc = {sum(p.values()) / len(p):.4f}")
    for k, r in res.items():
        base = (f"  {k:14s} d={r['delta']:+.4f}  95%CI[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]")
        if r.get("p_kind") == "bootstrap":
            print(f"{base}  boot_p={r['p']:.4g} holm={r.get('holm', float('nan')):.4g}")
        else:
            print(f"{base}  McNemar b={r['b']} c={r['c']} p={r['p']:.4g} "
                  f"holm={r.get('holm', float('nan')):.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--metric", default="correct",
                    choices=["correct", "correct_tolerant", "correct_official"],
                    help="PRIMARY is 'correct' (boxed letter). The other two are the "
                         "pre-registered sensitivity and the comparability metric; the "
                         "conclusion should be reported under all three.")
    a = ap.parse_args()

    print(f"[pass4] metric = {a.metric}"
          + ("  (PRIMARY)" if a.metric == "correct" else "  (robustness)"))
    per, cat = load(a.tag, a.metric)
    common = set.intersection(*[set(per[k]) for k in per if k in ("T0", "T1", "T2", "I0", "I1", "I2")]) \
        if len([k for k in per if k in ("T0", "T1", "T2", "I0", "I1", "I2")]) else set()
    idxs = sorted(common)
    print(f"[pass4] arms={sorted(per)}  paired items={len(idxs)}  boot={a.boot}")

    if "A5" in per:
        p = phat(per["A5"])
        # A5 HEADLINE = correct_tolerant (user decision, 2026-08-06, pre-outcome).
        # CapRL is a captioner: it was RL-trained to produce dense descriptions and never trained
        # to follow an MCQ format instruction, so its answer-extraction rate is expected to be
        # worse than the reasoners' even under D13's general extractor. Under the PRIMARY that
        # would read as "cannot do VQA" when it actually means "doesn't conclude in a recognisable
        # form" -- a capability claim contaminated by a format artifact. The tolerant metric falls
        # back to can_infer over the whole text, so it measures what this arm is for. Descriptive
        # only either way: A5 is NOT in the pre-registered contrast family.
        tag = "  <-- A5 HEADLINE" if a.metric == "correct_tolerant" else \
              "  (not A5's headline; see unextract_rate in score_meta)"
        print(f"[pass4] A5 (captioner VQA, DESCRIPTIVE ONLY) acc = "
              f"{sum(p.values()) / len(p):.4f} on n={len(p)}{tag}")

    ridx = [i for i in idxs if cat.get(i) in REASONING]
    pidx = [i for i in idxs if cat.get(i) in PERCEPTION]
    out_res = {}
    for fname, fam in (("A_blind", FAMILY_A), ("B_targeted", FAMILY_B)):
        if not all(x in per for pair in fam for x in pair if x and x != "INTERACTION"):
            print(f"\n[pass4] family {fname}: arms missing, skipped")
            continue
        out_res[f"{fname}_all"] = run(per, idxs, a.boot, family=fam)
        show(f"FAMILY {fname} - ALL MMStar", out_res[f"{fname}_all"], per, idxs)
        if pidx:
            out_res[f"{fname}_perception"] = run(per, pidx, a.boot, family=fam)
            show(f"FAMILY {fname} - PERCEPTION (coarse+fine)", out_res[f"{fname}_perception"], per, pidx)
        if ridx and fname == "B_targeted":
            # T3-T1 is near-tautological on perception axes (a targeted description of a directly
            # observable fact IS close to the answer) and genuinely informative on reasoning axes.
            out_res[f"{fname}_reasoning"] = run(per, ridx, a.boot, family=fam)
            show(f"FAMILY {fname} - REASONING axes (the interpretable half for T3-T1)",
                 out_res[f"{fname}_reasoning"], per, ridx)
    res_all, res_p = out_res.get("A_blind_all", {}), out_res.get("A_blind_perception", {})

    out = f"{C.OUT}/{a.tag}/analysis_{a.metric}.json"
    json.dump(C.provenance(pass_="4_analyze", metric=a.metric, n_paired=len(idxs),
                           n_perception=len(pidx), n_reasoning=len(ridx), families=out_res),
              open(out, "w"), indent=1, default=str)
    print(f"\n[pass4] -> {out}")


if __name__ == "__main__":
    main()
