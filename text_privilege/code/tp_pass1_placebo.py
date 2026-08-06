"""
Pass 1 — placebo donor assignment (CPU only, deterministic, no model).

Placebo = another item's CapRL caption. Per Q3 the donor is drawn from the SAME capability
category and matched on caption length: that is the harder, more conservative control, because
topic and style are held constant so only CONTENT differs. A cross-category donor would be
trivially detectable and would inflate the apparent content-specific effect.

Matching is done per caption_idx, so caption slot i of the target pairs with caption slot i of the
donor — keeping the "caption i pairs with draw i" structure intact in the placebo arm too.

  python tp_pass1_placebo.py [--tag smoke]
"""
import argparse, json, os
import tp_common as C


def assign(cap_rows, cat_of):
    """Deterministic same-category, length-matched, non-self donor assignment."""
    out, report = {}, []
    by_ci = {}
    for r in cap_rows:
        by_ci.setdefault(r["caption_idx"], []).append(r)

    for ci, rows in sorted(by_ci.items()):
        # bucket by category, sort by caption length -> nearest-length neighbour is adjacent
        buckets = {}
        for r in rows:
            buckets.setdefault(cat_of[r["index"]], []).append(r)
        for cat, items in sorted(buckets.items()):
            items = sorted(items, key=lambda r: (len(r["caption"]), r["index"]))
            n = len(items)
            if n < 2:
                # BUG 4 (found in pre-smoke re-check): a singleton category used to leave the item
                # with NO assignment, which then tripped `assert payload is not None` in Pass 2 and
                # crashed the run. Now: fall back to a cross-category donor from the same caption
                # slot, and LOG it loudly so the fallback rate is visible (gate G8 requires 0 on the
                # full run; the smoke may legitimately hit a few).
                pool = sorted([x for x in rows if x["index"] != items[0]["index"]],
                              key=lambda x: (abs(len(x["caption"]) - len(items[0]["caption"])),
                                             x["index"]))
                if pool:
                    r, donor = items[0], pool[0]
                    out[(r["index"], ci)] = dict(
                        donor_index=donor["index"], donor_caption=donor["caption"],
                        donor_caption_sha=donor["caption_sha"], category=cat,
                        len_target=len(r["caption"]), len_donor=len(donor["caption"]),
                        cross_category=True)
                    report.append(dict(index=r["index"], caption_idx=ci,
                                       fallback="singleton_category",
                                       donor_index=donor["index"],
                                       donor_category=cat_of[donor["index"]]))
                continue
            for pos, r in enumerate(items):
                # neighbour in length order; last element wraps to its predecessor
                donor = items[pos + 1] if pos + 1 < n else items[pos - 1]
                assert donor["index"] != r["index"], "self-assignment"
                out[(r["index"], ci)] = dict(
                    donor_index=donor["index"], donor_caption=donor["caption"],
                    donor_caption_sha=donor["caption_sha"], category=cat,
                    len_target=len(r["caption"]), len_donor=len(donor["caption"]))
    return out, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()

    cap_path = f"{C.OUT}/{a.tag}/captions.jsonl"
    cap_rows = [json.loads(l) for l in open(cap_path)]
    df = C.load_mmstar()
    cat_of = {int(r["index"]): r["category"] for _, r in df.iterrows()}

    amap, report = assign(cap_rows, cat_of)

    # --- asserts that make the control trustworthy -------------------------------------------
    for (idx, ci), d in amap.items():
        assert d["donor_index"] != idx, f"self donor at {idx}"
        if not d.get("cross_category"):
            assert cat_of[d["donor_index"]] == cat_of[idx], f"cross-category donor at {idx}"
    # every (item, caption slot) that exists must have a donor -- otherwise Pass 2 asserts out
    want = {(r["index"], r["caption_idx"]) for r in cap_rows}
    missing = want - set(amap)
    assert not missing, f"{len(missing)} (item,slot) pairs have no placebo donor: {sorted(missing)[:5]}"
    tgt = {(i, c): r for r in cap_rows for i, c in [(r["index"], r["caption_idx"])]}
    ident = sum(1 for (i, c), d in amap.items() if d["donor_caption"] == tgt[(i, c)]["caption"])
    assert ident == 0, f"{ident} donor captions are byte-identical to their target"

    lens = [abs(d["len_target"] - d["len_donor"]) for d in amap.values()]
    rel = [abs(d["len_target"] - d["len_donor"]) / max(1, d["len_target"]) for d in amap.values()]
    donors = [d["donor_index"] for d in amap.values()]
    reuse = max([donors.count(x) for x in set(donors)]) if donors else 0

    ser = {f"{i}|{c}": d for (i, c), d in amap.items()}
    out_path = f"{C.OUT}/{a.tag}/placebo_assignment.json"
    json.dump(ser, open(out_path, "w"), indent=0)

    meta = C.provenance(pass_="1_placebo", n_pairs=len(amap), fallbacks=report,
                        len_absdiff_mean=sum(lens) / max(1, len(lens)),
                        len_reldiff_mean=sum(rel) / max(1, len(rel)),
                        len_reldiff_max=max(rel) if rel else 0.0,
                        max_donor_reuse=reuse,
                        captions_sha=C.sha_file(cap_path),
                        assignment_sha=C.sha_file(out_path))
    json.dump(meta, open(f"{C.OUT}/{a.tag}/placebo_meta.json", "w"), indent=1)
    print(f"[pass1] pairs={len(amap)} mean|Δlen|={meta['len_absdiff_mean']:.0f} "
          f"mean rel={meta['len_reldiff_mean']:.3f} max rel={meta['len_reldiff_max']:.3f} "
          f"max_donor_reuse={reuse} fallbacks={len(report)}", flush=True)
    print("[pass1] assignment_sha", meta["assignment_sha"], flush=True)


if __name__ == "__main__":
    main()
