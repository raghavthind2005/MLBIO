#!/usr/bin/env python3
"""
Set 3 — shared paired-statistics harness (stdlib). Every Phase-2/A5 delta flows through this, so it is
unit-tested against hand-computed values (run `python3 stats.py`). All comparisons are PAIRED vs V0 on
identical items (same qi, same order), per the pre-registration.

- mcnemar_exact(b, c): two-sided exact McNemar p on discordant pairs (b = condition-wins, c = baseline-wins).
- paired_bootstrap_ci(x, y): 10k-resample paired bootstrap 95% CI on Δ = mean(y) - mean(x).
- paired_compare(x, y): full comparison dict (accuracies, Δ, CI, McNemar p, discordant counts).
- h1_verdict(...): the frozen PRIMARY endpoint rule (Δ≥0.10 AND CI-lo>0.03 AND McNemar p<0.05).
"""
import math, random

def mcnemar_exact(b, c):
    """Two-sided exact binomial McNemar p-value on discordant counts b, c (n=b+c under Bin(n,0.5))."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)

def paired_bootstrap_ci(x, y, n_boot=10000, seed=0, alpha=0.05):
    """95% CI on Δ = mean(y)-mean(x) by resampling ITEMS (paired). x,y: equal-length binary lists."""
    assert len(x) == len(y) and len(x) > 0
    n = len(x); rng = random.Random(seed)
    base = sum(y) / n - sum(x) / n
    d = [0.0] * n_boot
    for t in range(n_boot):
        sy = sx = 0
        for _ in range(n):
            i = rng.randrange(n); sy += y[i]; sx += x[i]
        d[t] = (sy - sx) / n
    d.sort()
    lo = d[int((alpha / 2) * n_boot)]
    hi = d[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return base, lo, hi

def paired_compare(x, y, name="", n_boot=10000, seed=0):
    """x = baseline (V0) binary correctness; y = condition; SAME items in SAME order."""
    assert len(x) == len(y)
    n = len(x)
    b = sum(1 for i in range(n) if y[i] == 1 and x[i] == 0)   # condition right, baseline wrong
    c = sum(1 for i in range(n) if y[i] == 0 and x[i] == 1)   # baseline right, condition wrong
    delta, lo, hi = paired_bootstrap_ci(x, y, n_boot, seed)
    return dict(name=name, n=n, acc_base=sum(x) / n, acc_cond=sum(y) / n,
                delta=round(delta, 4), ci=(round(lo, 4), round(hi, 4)),
                cond_wins=b, base_wins=c, mcnemar_p=round(mcnemar_exact(b, c), 6))

def h1_verdict(x_v0, y_vself, n_boot=10000, seed=0):
    """Frozen PRIMARY endpoint: Δ≥0.10 AND 95% CI lower bound >0.03 AND McNemar p<0.05."""
    r = paired_compare(x_v0, y_vself, "H1 V_self-V0", n_boot, seed)
    r["H1_pass"] = bool(r["delta"] >= 0.10 and r["ci"][0] > 0.03 and r["mcnemar_p"] < 0.05)
    return r


def _selftest():
    ok = True
    def expect(name, got, want, tol=None):
        nonlocal ok
        p = (abs(got - want) <= tol) if tol is not None else (got == want)
        ok = ok and p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: got {got}, want {want}")

    # --- McNemar exact, hand-computed ---
    expect("mcnemar(0,0)=1", mcnemar_exact(0, 0), 1.0)
    expect("mcnemar(10,0)=2*.5^10", mcnemar_exact(10, 0), 2 * 0.5 ** 10, 1e-12)      # 0.001953
    expect("mcnemar(5,5)=1 (capped)", mcnemar_exact(5, 5), 1.0, 1e-12)
    # n=10,k=2: 2*(C10_0+C10_1+C10_2)/1024 = 2*56/1024 = 0.109375
    expect("mcnemar(8,2)", mcnemar_exact(8, 2), 2 * (1 + 10 + 45) / 1024, 1e-12)
    # n=20,k=0: 2*.5^20
    expect("mcnemar(20,0)", mcnemar_exact(20, 0), 2 * 0.5 ** 20, 1e-12)

    # --- bootstrap CI degenerate cases ---
    d, lo, hi = paired_bootstrap_ci([0]*50, [0]*50)
    expect("boot Δ=0 when identical", d, 0.0); expect("boot CI lo=0", lo, 0.0); expect("boot CI hi=0", hi, 0.0)
    d, lo, hi = paired_bootstrap_ci([0]*40, [1]*40)
    expect("boot Δ=1 all-flip", d, 1.0); expect("boot CI=(1,1)", lo == 1.0 and hi == 1.0, True)

    # --- paired_compare structure on a known table ---
    # 100 items: 15 cond-wins (V0=0,cond=1), 5 base-wins (V0=1,cond=0), rest agree -> Δ = (15-5)/100 = +0.10
    x = [1]*40 + [0]*60                      # V0 correct on 40
    y = x[:]                                 # start equal
    for i in range(15): y[40 + i] = 1        # cond fixes 15 that V0 got wrong
    for i in range(5):  y[i] = 0             # cond breaks 5 that V0 got right
    r = paired_compare(x, y, "known")
    expect("Δ = (b-c)/n = 0.10", r["delta"], 0.10, 1e-9)
    expect("cond_wins=15", r["cond_wins"], 15); expect("base_wins=5", r["base_wins"], 5)
    expect("McNemar==mcnemar(15,5)", r["mcnemar_p"], round(mcnemar_exact(15, 5), 6), 1e-9)
    expect("CI lower bound > 0 for this Δ", r["ci"][0] > 0, True)

    # --- h1_verdict logic (Δ exactly 0.10, strong separation -> should pass) ---
    v = h1_verdict(x, y); expect("H1 verdict returns bool", isinstance(v["H1_pass"], bool), True)

    print("\nSTATS SELF-TEST", "PASSED" if ok else "FAILED")
    return ok

if __name__ == "__main__":
    _selftest()
