#!/usr/bin/env python3
"""Pre-launch DESIGN + CODE audit for Track T generation outputs.

Runs on the generation dump (light json + full jsonl + meta) and the frozen manifests,
and ASSERTS that what was generated matches what the design specifies — i.e. "are we
actually implementing what we mean to implement". No GPU; login-node safe (stdlib only,
imports render_delta/WRAP from mv_gen for a single source of truth).

Usage: python3 mv_gen_audit.py [smoke|full]
Exit code 0 = all pass, 1 = at least one failure.
"""
import os, sys, json, hashlib
from collections import Counter
import mv_score
from mv_gen import render_delta, WRAP

MODE = sys.argv[1] if len(sys.argv) > 1 else "smoke"
DSS  = os.environ.get("DSS", ".")
sha  = lambda s: hashlib.sha256(s.encode()).hexdigest()


def main():
    meta     = json.load(open(f"{DSS}/mv_gen_{MODE}_meta.json"))
    K        = meta["K"]
    manifest = {r["pid"]: r for r in json.load(open(f"{DSS}/pool_manifest.json"))}
    placebo  = json.load(open(f"{DSS}/placebo_assignment.json"))
    full     = [json.loads(l) for l in open(f"{DSS}/mv_gen_{MODE}_full.jsonl")]
    D        = {r["pid"]: r["text"] for r in full if r.get("arm") == "selfdesc_D"}
    rows     = [r for r in full if r.get("arm") in ("base", "privileged", "self", "placebo")]

    pids = sorted({r["pid"] for r in rows}, key=int)
    arms = ["base", "privileged", "self", "placebo"]
    by   = {(r["pid"], r["arm"]): r for r in rows}
    fails = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   -> {detail}" if (detail and not cond) else ""))
        if not cond:
            fails.append(name)

    print(f"=== Track T generation audit  MODE={MODE}  n_items={len(pids)}  K={K} "
          f"git={meta.get('git_sha') or 'NA'} ===")

    # C1 — exactly K draws per (item, arm)
    bad = [(r["pid"], r["arm"], len(r["draws"])) for r in rows if len(r["draws"]) != K]
    check(f"C1 exactly K={K} draws per (item,arm)", not bad, str(bad[:5]))

    # C2 — all 4 arms present exactly once per item
    cnt = Counter((r["pid"], r["arm"]) for r in rows)
    missing = [(p, a, cnt[(p, a)]) for p in pids for a in arms if cnt[(p, a)] != 1]
    check("C2 4 arms x1 per item", not missing, str(missing[:5]))

    # C3 — injected payload SHA == intended content (the core arm-integrity check)
    mism = []
    for p in pids:
        exp = {"base": "",
               "privileged": render_delta(manifest[p]["delta"]),
               "self": D[p],
               "placebo": render_delta(manifest[placebo[p]]["delta"])}
        for a in arms:
            got  = by[(p, a)].get("payload_sha", "<missing>")
            want = "" if exp[a] == "" else sha(exp[a])
            if got != want:
                mism.append((p, a))
    check("C3 payload sha == intended (priv=own delta, placebo=donor delta, self=D, base=empty)",
          not mism, str(mism[:5]))

    # C4 — placebo donor correct, != self, payload distinct from privileged
    dbad = []
    for p in pids:
        r = by[(p, "placebo")]
        if r.get("donor") != placebo[p] or placebo[p] == p:
            dbad.append((p, "donor", r.get("donor")))
        if by[(p, "placebo")].get("payload_sha") == by[(p, "privileged")].get("payload_sha"):
            dbad.append((p, "payload==priv"))
    check("C4 placebo donor==assignment, donor!=self, payload!=privileged", not dbad, str(dbad[:5]))

    # C5 — base has no seed
    check("C5 base payload empty", all(by[(p, "base")].get("payload_sha", "") == "" for p in pids))

    # C6 — no prefill leakage: wrapper never appears in generated text (it lives in the prompt)
    leak = [(r["pid"], r["arm"]) for r in rows for d in r["draws"] if WRAP.strip() in d["text"]]
    check("C6 no wrapper string in generated text (prefill stayed in prompt)", not leak, str(leak[:5]))

    # C7 — GT/qtype join valid + stored answer matches manifest
    badj = [p for p in pids if p not in manifest or manifest[p]["qtype"] not in ("multi-choice", "free-form")]
    check("C7 manifest qtype valid for all pids", not badj, str(badj[:5]))
    amis = [r["pid"] for r in rows if r.get("answer") != manifest[r["pid"]]["answer"]]
    check("C7b stored answer == manifest answer", not amis, str(amis[:5]))

    # C8 — recompute metrics independently + assert ok is binary
    print("  --- recomputed metrics (cross-check by eye vs the job's MECHANICS block) ---")
    binbad = 0
    for a in arms:
        rs = [by[(p, a)] for p in pids]; n = len(rs)
        alld = [d for r in rs for d in r["draws"]]
        avgk = sum(sum(d["ok"] for d in r["draws"]) / K for r in rs) / n
        maj  = sum(int(2 * sum(d["ok"] for d in r["draws"]) > K) for r in rs) / n
        trunc = sum(d["trunc"] for d in alld) / len(alld)
        unst = sum(1 for r in rs if 0 < sum(d["ok"] for d in r["draws"]) < K) / n
        binbad += sum(1 for d in alld if d["ok"] not in (0, 1, True, False))
        print(f"    {a:11} avg@{K}={avgk:.3f} maj={maj:.3f} trunc={trunc:.3f} decode_unstable={unst:.3f}")
    check("C8 ok values are binary", binbad == 0, f"{binbad} non-binary")

    # C9 — self-desc has no answer leak and is non-empty
    dleak  = [p for p in pids if mv_score.extract_boxed(D[p]) is not None]
    dempty = [p for p in pids if len(D[p].strip()) == 0]
    check("C9 self-desc D: no boxed answer + non-empty", not dleak and not dempty,
          f"leak={dleak[:5]} empty={dempty[:5]}")

    # C10 — provenance completeness
    check("C10 meta provenance (code+artifact SHAs, params, seed, K)",
          all(k in meta for k in ("code_sha", "artifact_sha", "sampling", "seed", "K")))

    print(f"\n{'=' * 60}")
    print("ALL AUDIT CHECKS PASS" if not fails else "FAILURES: " + str(fails))
    print('=' * 60)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
