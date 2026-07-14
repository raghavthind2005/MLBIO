#!/usr/bin/env python3
"""Track T analysis — primary Δ(priv-base), placebo gate, recovery, per-subgroup, variance decomp.

Reads mv_gen_{MODE}.json (per-draw `ok`). Dry-runnable on the smoke to VET the pipeline before
the real run (numbers are noisy at n=20; the point is that the estimators compute correctly).

Estimators:
  * continuous per-item p̂ = mean(draws correct); Δ = mean_i(p̂_a - p̂_b).
    TWO-LEVEL bootstrap CI: resample items with replacement, then resample the K draws WITHIN
    each resampled item — so the CI includes decode variance, not just between-item variance.
  * majority-vote exact McNemar (directive-faithful; ties excluded).
  * recovery = (self-base)/(priv-base), joint bootstrap (num+den recomputed together).
  * Holm across the reported McNemar contrasts.
  * mc (primary) / ff (secondary) split.
  * variance decomposition of the per-item Δ: between-item vs decode(within/K) -> was K enough?

Runs in-container (needs numpy; scipy not required). Usage: python3 mv_analyze.py [smoke|full] [nboot]
"""
import os, sys, json, re
from math import factorial
import numpy as np

MODE  = sys.argv[1] if len(sys.argv) > 1 else "smoke"
NBOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
DSS   = os.environ.get("DSS", ".")
np.random.seed(0)
ARMS  = ("base", "privileged", "self", "placebo")


def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


def load():
    light = json.load(open(f"{DSS}/mv_gen_{MODE}.json"))
    data, box, qtype = {}, {}, {}
    for r in light:
        data.setdefault(r["pid"], {})[r["arm"]] = np.array([int(bool(d["ok"])) for d in r["draws"]], float)
        box.setdefault(r["pid"], {})[r["arm"]] = [d.get("box") for d in r["draws"]]
        qtype[r["pid"]] = r["qtype"]
    return data, box, qtype


def phat(data, arm, pids):
    return np.array([data[p][arm].mean() for p in pids])


def two_level_boot(data, a, b, pids, nb):
    pids = list(pids); n = len(pids); out = np.empty(nb)
    for t in range(nb):
        idx = np.random.randint(0, n, n)
        da = np.empty(n); db = np.empty(n)
        for j, ii in enumerate(idx):
            A = data[pids[ii]][a]; B = data[pids[ii]][b]
            da[j] = A[np.random.randint(0, len(A), len(A))].mean()
            db[j] = B[np.random.randint(0, len(B), len(B))].mean()
        out[t] = da.mean() - db.mean()
    return out


def mcnemar_majority(data, a, b, pids):
    bcnt = ccnt = ties = 0
    for p in pids:
        ka, kb = data[p][a].mean(), data[p][b].mean()
        ma = 1 if ka > 0.5 else (0 if ka < 0.5 else None)
        mb = 1 if kb > 0.5 else (0 if kb < 0.5 else None)
        if ma is None or mb is None:
            ties += 1; continue
        if ma == 1 and mb == 0: bcnt += 1
        elif ma == 0 and mb == 1: ccnt += 1
    nd = bcnt + ccnt
    if nd == 0:
        p = 1.0
    else:
        k = min(bcnt, ccnt)
        p = min(1.0, 2 * sum(comb(nd, i) for i in range(0, k + 1)) / (2 ** nd))
    return bcnt, ccnt, ties, p


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def holm(named):
    items = sorted(named.items(), key=lambda kv: kv[1]); m = len(items); out = {}
    prev = 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        prev = max(prev, adj); out[name] = prev
    return out


def analyze(data, pids, label):
    print(f"\n===== {label}   n={len(pids)} =====")
    for a in ARMS:
        print(f"  {a:11} avg@K acc = {phat(data, a, pids).mean():.3f}")
    contrasts = {}
    for a, b in [("privileged", "base"), ("self", "base"), ("placebo", "base"), ("privileged", "placebo")]:
        d = phat(data, a, pids).mean() - phat(data, b, pids).mean()
        boot = two_level_boot(data, a, b, pids, NBOOT); lo, hi = ci(boot)
        p2 = 2 * min((boot > 0).mean(), (boot < 0).mean())
        bcnt, ccnt, ties, pmc = mcnemar_majority(data, a, b, pids)
        contrasts[f"{a}-{b}"] = pmc
        print(f"  Δ({a}-{b}) = {d:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  boot_p={p2:.4f}"
              f"  | McNemar-maj b={bcnt} c={ccnt} ties={ties} exact_p={pmc:.4f}")

    pb, pp, ps = (phat(data, x, pids) for x in ("base", "privileged", "self"))
    den = pp.mean() - pb.mean(); num = ps.mean() - pb.mean()
    rec = num / den if abs(den) > 1e-9 else float("nan")
    pl = list(pids); n = len(pl); recs = []
    for _ in range(NBOOT):
        idx = np.random.randint(0, n, n)
        b_ = np.mean([data[pl[i]]["base"].mean() for i in idx])
        p_ = np.mean([data[pl[i]]["privileged"].mean() for i in idx])
        s_ = np.mean([data[pl[i]]["self"].mean() for i in idx])
        if abs(p_ - b_) > 1e-9:
            recs.append((s_ - b_) / (p_ - b_))
    rlo, rhi = ci(np.array(recs)) if recs else (float("nan"), float("nan"))
    print(f"  recovery (self-base)/(priv-base) = {rec:.3f}  95%CI[{rlo:.3f},{rhi:.3f}]  (den={den:+.3f})")

    hp = holm(contrasts)
    print("  Holm-adjusted McNemar p:", {k: round(v, 4) for k, v in hp.items()})

    di = phat(data, "privileged", pids) - phat(data, "base", pids)
    between = float(di.var(ddof=1))
    within = float(np.mean([data[p]["privileged"].var() / len(data[p]["privileged"])
                            + data[p]["base"].var() / len(data[p]["base"]) for p in pids]))
    frac = within / (between + within) if (between + within) > 0 else 0.0
    print(f"  var-decomp of per-item Δ: between-item={between:.4f} decode(within/K)={within:.4f} "
          f"decode_frac={frac:.1%}  [<~20% => K was sufficient]")


def main():
    data, box, qtype = load()
    allp = sorted(data.keys(), key=int)
    mc = [p for p in allp if qtype[p] == "multi-choice"]
    ff = [p for p in allp if qtype[p] == "free-form"]

    # answer-format confound: on MC a value-boxed answer (not a letter) is scored wrong; if seeding
    # shifts arms toward value-boxing it biases them DOWN (conservative, but differential). Watch it.
    print("\n  MC non-letter-box rate per arm (higher => more value-boxed answers scored wrong):")
    for a in ARMS:
        bs = [b for p in mc for b in box[p][a]]
        nonl = sum(1 for b in bs if not (b and re.fullmatch(r"[A-Fa-f]", b.strip())))
        print(f"    {a:11} = {nonl / len(bs):.3f}  (n={len(bs)})" if bs else f"    {a:11} n/a")

    analyze(data, allp, "ALL scored")
    if mc: analyze(data, mc, "MC (primary)")
    if ff: analyze(data, ff, "FF (secondary)")


if __name__ == "__main__":
    main()
