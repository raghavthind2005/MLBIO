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

MODE   = sys.argv[1] if len(sys.argv) > 1 else "smoke"
DSS    = os.environ.get("DSS", ".")
MVDIR  = os.environ.get("MVDIR", "/iopsstor/scratch/cscs/raghavthind/set2_pilot/mv")
IMGDIR = f"{MVDIR}/images"
sha    = lambda s: hashlib.sha256(s.encode()).hexdigest()
fsha   = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    meta     = json.load(open(f"{DSS}/mv_gen_{MODE}_meta.json"))
    K        = meta["K"]
    manifest = {r["pid"]: r for r in json.load(open(f"{DSS}/pool_manifest.json"))}
    placebo  = json.load(open(f"{DSS}/placebo_assignment.json"))
    full     = [json.loads(l) for l in open(f"{DSS}/mv_gen_{MODE}_full.jsonl")]
    D        = {r["pid"]: r["texts"] for r in full if r.get("arm") == "selfdesc_D"}   # K descriptions/item
    rows     = [r for r in full if r.get("arm") in ("base", "privileged", "self", "placebo")]
    viq      = {json.loads(l)["pid"]: json.loads(l) for l in open(f"{MVDIR}/mv_vi.jsonl")}

    def score(pid, text):
        r = manifest[pid]
        return mv_score.score_mc(text, r["answer"]) if r["qtype"] == "multi-choice" else mv_score.score_ff(text, r["answer"])

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

    # C3 — per-draw injected payload SHA == intended (core arm-integrity check). base: every draw empty.
    # priv: own delta (all K). placebo: donor delta (all K). self: the K draws' shas == the multiset of
    # the item's K independent self-descriptions.
    dshas = lambda p, a: sorted(d.get("payload_sha", "<missing>") for d in by[(p, a)]["draws"])
    mism = []
    for p in pids:
        want = {"base":       sorted([""] * K),
                "privileged": sorted([sha(render_delta(manifest[p]["delta"]))] * K),
                "placebo":    sorted([sha(render_delta(manifest[placebo[p]]["delta"]))] * K),
                "self":       sorted(sha(x) for x in D[p])}
        for a in arms:
            if dshas(p, a) != want[a]:
                mism.append((p, a))
    check("C3 per-draw payload sha == intended (priv=own delta, placebo=donor, self=K indep D's, base=empty)",
          not mism, str(mism[:5]))

    # C4 — placebo donor correct, != self, payload distinct from privileged
    dbad = []
    for p in pids:
        r = by[(p, "placebo")]
        if r.get("donor") != placebo[p] or placebo[p] == p:
            dbad.append((p, "donor", r.get("donor")))
        if r["draws"][0]["payload_sha"] == by[(p, "privileged")]["draws"][0]["payload_sha"]:
            dbad.append((p, "payload==priv"))
    check("C4 placebo donor==assignment, donor!=self, payload!=privileged", not dbad, str(dbad[:5]))

    # C5 — base has no seed (every draw)
    check("C5 base draws all unseeded (empty payload)",
          all(all(d["payload_sha"] == "" for d in by[(p, "base")]["draws"]) for p in pids))

    # C6 — the prompt/prefill must not be COPIED into the scored generation. vLLM output.text is
    # generation-only by construction; the real bug signature is the wrapper at the very START of
    # output (prompt echo), which would also hit ~ALL seeded cells. A model re-stating the wrapper
    # phrase mid-reasoning is benign — scoring is on \boxed{} and the seed is perception-only.
    W = WRAP.strip()
    echoes = [(r["pid"], r["arm"], d["text"].find(W)) for r in rows for d in r["draws"] if W in d["text"]]
    start_echo = [(p, a, off) for (p, a, off) in echoes if off < 40]
    check("C6 prompt not copied into generation (no wrapper at output start)", not start_echo, str(start_echo[:5]))
    if echoes:
        print(f"    [info] wrapper phrase re-emitted mid-reasoning in {len(echoes)} draw(s), "
              f"min char-offset={min(o for *_, o in echoes)} — benign restatement, not prompt echo")

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

    # C9 — self-desc: K/item, non-empty, answer-free. The model occasionally slips a boxed answer despite
    # the instruction; such self-draws are EXCLUDED in analysis. Pass if the leak rate is small (<1%);
    # fail only on systemic leakage (a broken prompt).
    dleak  = [(p, i) for p in pids for i, x in enumerate(D[p]) if mv_score.extract_boxed(x) is not None]
    dempty = [(p, i) for p in pids for i, x in enumerate(D[p]) if len(x.strip()) == 0]
    kbad   = [p for p in pids if len(D[p]) != K]
    ntot   = sum(len(D[p]) for p in pids)
    leakrate = len(dleak) / max(1, ntot)
    check("C9 self-desc: K/item, non-empty; answer-leak rate <1% (leaked draws excluded in analysis)",
          not (dempty or kbad) and leakrate < 0.01,
          f"rate={leakrate:.3%} leak={dleak[:5]} empty={dempty[:3]} kbad={kbad[:3]}")
    if dleak:
        print(f"    [info] {len(dleak)}/{ntot} self-descriptions leaked a boxed answer ({leakrate:.2%}) "
              f"-> those self-draws are dropped in mv_analyze: {dleak[:8]}")

    # C10 — provenance completeness
    check("C10 meta provenance (code+artifact+image SHAs, mv_vi, params, seed, K)",
          all(k in meta for k in ("code_sha", "artifact_sha", "image_sha", "mv_vi_sha", "sampling", "seed", "K")))

    # C11 — image provenance: on-disk image re-hash matches meta, present for every item
    ish = meta.get("image_sha", {})
    imgbad = []
    for p in pids:
        if p not in ish:
            imgbad.append((p, "no-meta-sha")); continue
        try:
            if fsha(f"{IMGDIR}/{p}.png") != ish[p]:
                imgbad.append((p, "disk!=meta"))
        except FileNotFoundError:
            imgbad.append((p, "missing-file"))
    check("C11 image sha present + on-disk re-hash == meta (right image bytes per item)",
          not imgbad, str(imgbad[:5]))

    # C12 — question provenance: mv_vi.jsonl hash matches meta, per-row question_sha == source
    check("C12 mv_vi.jsonl hash == meta", meta.get("mv_vi_sha") == fsha(f"{MVDIR}/mv_vi.jsonl"),
          f"meta={meta.get('mv_vi_sha')}")
    qbad = [(r["pid"], r["arm"]) for r in rows if r.get("question_sha") != sha(viq[r["pid"]]["question"])]
    check("C12b per-row question_sha == source question", not qbad, str(qbad[:5]))

    # C13 — independent re-score from stored text with the frozen scorer (catches storage/version drift)
    rsbad = boxbad = 0
    for r in rows:
        for d in r["draws"]:
            if bool(score(r["pid"], d["text"])) != bool(d["ok"]):
                rsbad += 1
            if mv_score.extract_boxed(d["text"]) != d.get("box"):
                boxbad += 1
    check("C13 re-score(stored text) == stored ok", rsbad == 0, f"{rsbad} mismatches")
    check("C13b re-extract box == stored box", boxbad == 0, f"{boxbad} mismatches")

    # ---- HUMAN scorer eyeball: the one thing no assert can settle (is the scorer's JUDGMENT right?) ----
    print("\n  ---- up to 10 cells where the K draws DISAGREE (inspect GT vs extracted boxes) ----")
    shown = 0
    for r in rows:
        oks = [d["ok"] for d in r["draws"]]
        if 0 < sum(oks) < len(oks):
            print(f"    pid={r['pid']} arm={r['arm']:10} qtype={r['qtype']:12} "
                  f"GT={manifest[r['pid']]['answer']!r} ok={oks} boxes={[d.get('box') for d in r['draws']]}")
            shown += 1
            if shown >= 10:
                break
    if shown == 0:
        print("    (no within-cell disagreement at this n)")

    print(f"\n{'=' * 60}")
    print("ALL AUDIT CHECKS PASS" if not fails else "FAILURES: " + str(fails))
    print('=' * 60)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
