"""
Pass 3 — score every generation with the frozen judge-free scorer, and emit the per-arm
diagnostics that D7/D11c flagged as mandatory (they are confounds, not nice-to-haves):

  * un-inferable rate      -- the official rule's LLM fallback is DISABLED; un-inferable == wrong.
                              Both official rules penalise verbosity (letter must be in the last 4
                              words; text match rejected beyond 2x choice length), so this rate can
                              differ by arm for reasons unrelated to perception. Watch it.
  * predicted-letter distribution -- MMStar's gold labels are NOT uniform (B447 A429 D315 C309), so a
                              caption that shifts choice bias could move accuracy without moving
                              perception.
  * truncation rate        -- caption arms carry ~2k extra prompt tokens.
  * unclosed-<think> rate  -- thinking arms only.

Recomputable from Pass 2 at any time; nothing here is destructive.

  python tp_pass3_score.py [--tag smoke]
"""
import argparse, collections, glob, json, os
import tp_common as C


def _mean(v):
    return (sum(v) / len(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()

    df = C.load_mmstar()
    gt = {int(r["index"]): r["answer"] for _, r in df.iterrows()}
    cat = {int(r["index"]): r["category"] for _, r in df.iterrows()}
    # l2_category (18 axes) is carried through as well: MMStar's own headline metric is averaged
    # per l2 axis, and the finer breakdown is the only way to look inside "perception" later.
    l2 = {int(r["index"]): r["l2_category"] for _, r in df.iterrows()}
    ch = {int(r["index"]): C.parse_choices(r["question"]) for _, r in df.iterrows()}

    bad = [i for i, c in ch.items() if len(c) < 2]
    print(f"[pass3] choice-parse failures: {len(bad)}/{len(ch)}"
          + (f"  e.g. {bad[:5]}" if bad else ""), flush=True)

    out_path = f"{C.OUT}/{a.tag}/scored.jsonl"
    tmp = out_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)          # BUG 6: Appender opens "a"; a stale .tmp would duplicate every row
    app = C.Appender(tmp)
    summary = {}

    for gp in sorted(glob.glob(f"{C.OUT}/{a.tag}/gen_*.jsonl")):
        arm = os.path.basename(gp)[4:-6]
        if arm not in C.ARMS:
            continue
        is_think = (C.ARMS[arm][0] == "thinking")
        rows, stats = [], collections.Counter()
        preds, kinds = collections.Counter(), collections.Counter()
        # BUG 3: skip torn lines instead of crashing. BUG 2: a resumed run can append a duplicate
        # for a key whose original line was torn -- keep the last and report the count.
        raw, torn = C.read_jsonl(gp)
        raw, dupes = C.dedup_rows(raw, lambda r: (r["index"], r["draw"]))
        if torn or dupes:
            print(f"[pass3] {arm}: torn_lines={torn} duplicate_keys_dropped={dupes}", flush=True)
        for r in raw:
            idx = r["index"]
            s = C.score_item(r["text"], dict(ch[idx]), gt[idx], is_think)
            rows.append(dict(index=idx, draw=r["draw"], arm=arm,
                             correct=s["correct"],                    # boxed letter (PRIMARY)
                             correct_tolerant=s["correct_tolerant"],  # sensitivity
                             correct_official=s["correct_official"],  # MMStar's own scorer
                             box_kind=s["box_kind"], pred=s["box_pred"],
                             fc_artifact=s["fc_artifact"],
                             trunc=r["trunc"], ntok=r["ntok"], ptok=r["ptok"],
                             # carried so downstream analysis never has to re-join to gen_*.jsonl
                             img_tok=r.get("img_tok"), cumlogprob=r.get("cumlogprob"),
                             think_tok=r.get("think_tok"), answer_tok=r.get("answer_tok"),
                             caption_idx=r.get("caption_idx"), donor_index=r.get("donor_index"),
                             seed=r.get("seed"),
                             closed_think=r.get("closed_think", 1),
                             category=cat[idx], l2_category=l2[idx]))
            stats["n"] += 1
            stats["correct"] += s["correct"]
            stats["correct_tolerant"] += s["correct_tolerant"]
            stats["correct_official"] += s["correct_official"]
            stats["fc_artifact"] += s["fc_artifact"]
            stats["trunc"] += r["trunc"]
            stats["unclosed"] += (0 if r.get("closed_think", 1) else 1)
            kinds[s["box_kind"]] += 1
            preds[s["box_pred"] or "NONE"] += 1
        app.write(rows)
        n = max(stats["n"], 1)
        summary[arm] = dict(
            n=stats["n"],
            acc=stats["correct"] / n,                                   # PRIMARY (boxed letter)
            acc_tolerant=stats["correct_tolerant"] / n,                 # SENSITIVITY
            acc_official=stats["correct_official"] / n,                 # COMPARABILITY
            format_gap=(stats["correct_tolerant"] - stats["correct"]) / n,
            box_kind_dist={k: v / n for k, v in sorted(kinds.items())},
            box_letter_rate=kinds["letter"] / n,
            box_missing_rate=kinds["none"] / n,
            box_value_rate=kinds["value"] / n,                          # the Track-T §11.8 confound
            firstchar_artifact_rate=stats["fc_artifact"] / n,
            trunc_rate=stats["trunc"] / n,
            unclosed_think_rate=stats["unclosed"] / n,
            # chain-length signal: the washout hypothesis predicts a caption SHORTENS the chain
            # (S2T: better perception -> 20.8% shorter traces). Reported per arm from the start.
            mean_ntok=_mean([r["ntok"] for r in rows]),
            mean_think_tok=_mean([r["think_tok"] for r in rows if r["think_tok"] is not None]),
            mean_answer_tok=_mean([r["answer_tok"] for r in rows if r["answer_tok"] is not None]),
            mean_img_tok=_mean([r["img_tok"] for r in rows if r["img_tok"] is not None]),
            mean_cumlogprob=_mean([r["cumlogprob"] for r in rows if r["cumlogprob"] is not None]),
            pred_dist={k: v / n for k, v in sorted(preds.items())})
        s = summary[arm]
        print(f"[pass3] {arm:3s} n={s['n']:6d} acc={s['acc']:.4f} "
              f"tol={s['acc_tolerant']:.4f} official={s['acc_official']:.4f} "
              f"fmt_gap={s['format_gap']:+.4f}", flush=True)
        print(f"          box: letter={s['box_letter_rate']:.4f} value={s['box_value_rate']:.4f} "
              f"missing={s['box_missing_rate']:.4f} | trunc={s['trunc_rate']:.4f} "
              f"unclosed={s['unclosed_think_rate']:.4f} "
              f"fc_artifact={s['firstchar_artifact_rate']:.4f}", flush=True)
        _f = lambda x: "n/a" if x is None else f"{x:.1f}"
        print(f"          len: ntok={_f(s['mean_ntok'])} think={_f(s['mean_think_tok'])} "
              f"answer={_f(s['mean_answer_tok'])} img_tok={_f(s['mean_img_tok'])} "
              f"cumlogprob={_f(s['mean_cumlogprob'])}", flush=True)
    app.close()
    os.replace(out_path + ".tmp", out_path)

    json.dump(C.provenance(pass_="3_score", summary=summary,
                           choice_parse_failures=len(bad),
                           scored_sha=C.sha_file(out_path)),
              open(f"{C.OUT}/{a.tag}/score_meta.json", "w"), indent=1)
    print("[pass3] ->", out_path, flush=True)


if __name__ == "__main__":
    main()
