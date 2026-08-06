"""
Peek reporter: per-candidate caption statistics + full text for eyeball inspection.

Split out of peek_captions.sbatch because the inline heredoc version had a quoting bug --
escaped quotes inside an f-string expression (`{v[0][\"caption\"]}`) are a SyntaxError once the
nested bash quoting mangles them. Generation was unaffected; only the printing failed. Lesson:
never put Python inside a doubly-quoted shell heredoc; put it in a file.

  python peek_report.py [--show 2]
"""
import argparse, collections, json, os, statistics
import tp_common as C

CANDIDATES = ["A_genconfig", "B_readme3b", "C_base_pp"]


def repetition(text):
    """Fraction of the caption occupied by its most-repeated 6-gram. High => degenerate looping."""
    w = text.split()
    if len(w) < 24:
        return 0.0
    g = collections.Counter(tuple(w[i:i + 6]) for i in range(len(w) - 6))
    return g.most_common(1)[0][1] * 6 / max(len(w), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=2, help="how many full captions to print")
    a = ap.parse_args()

    df = C.load_mmstar()
    q = {int(r["index"]): str(r["question"]) for _, r in df.iterrows()}
    ans = {int(r["index"]): str(r["answer"]) for _, r in df.iterrows()}

    for dec in CANDIDATES:
        p = f"{C.OUT}/peek_{dec}/captions.jsonl"
        if not os.path.exists(p):
            print(f"[skip] {dec}: no captions")
            continue
        rows, _ = C.read_jsonl(p)
        ln = sorted(r["ntok"] for r in rows)
        r6 = [repetition(r["caption"]) for r in rows]
        by = collections.defaultdict(list)
        for r in rows:
            by[r["index"]].append(r)
        dis = [len({x["caption_sha"] for x in v}) / len(v) for v in by.values() if len(v) > 1]
        print("=" * 78)
        print(f"CANDIDATE {dec}   n={len(rows)}")
        print(f"  ntok min/med/max = {ln[0]} / {ln[len(ln)//2]} / {ln[-1]}"
              f"   (CapRL++ regularises at tau1=2048, tau2=3072)")
        print(f"  truncated={sum(r['trunc'] for r in rows)}  "
              f"empty={sum(1 for r in rows if not r['caption'].strip())}")
        print(f"  repetition mean={statistics.mean(r6):.3f} max={max(r6):.3f}")
        if dis:
            print(f"  distinct-caption fraction across draws mean={statistics.mean(dis):.3f}")

    ref = f"{C.OUT}/peek_{CANDIDATES[0]}/captions.jsonl"
    if os.path.exists(ref) and a.show:
        rows, _ = C.read_jsonl(ref)
        rows.sort(key=lambda r: (r["index"], r["caption_idx"]))
        for r in rows[:a.show]:
            i = r["index"]
            print("#" * 78)
            print(f"ITEM {i}  draw {r['caption_idx']}  ntok={r['ntok']}  gt={ans.get(i)}")
            print(f"QUESTION: {q.get(i, '?')[:400]}")
            print("-" * 78)
            print(r["caption"])
            print()


if __name__ == "__main__":
    main()
