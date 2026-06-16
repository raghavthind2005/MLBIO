#!/usr/bin/env python3
"""
Build a self-contained, browsable HTML of every FORCED sample's turn-0 vs turn-1
reasoning and answers, grouped by category. Each sample is a collapsible card;
flips are colour-coded. Images embedded as base64 (cached + downscaled).

  python build_forced_traces_html.py   ->  forced_traces.html
"""
import base64
import html
import io
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

SD = Path(__file__).parent
forced = [json.loads(l) for l in open(SD/"results_forced"/"forced_results.jsonl")
          if l.strip() and "error" not in json.loads(l)]

# ── image cache ───────────────────────────────────────────────────────────────
_cache = {}
def img_uri(r, maxw=260, q=70):
    p = SD/"data"/"hallusionbench"/"data"/r["category"]/r["subcategory"]/f"{r['set_id']}_{r['figure_id']}.png"
    key = str(p)
    if key in _cache:
        return _cache[key]
    if not p.exists():
        _cache[key] = None
        return None
    im = Image.open(p).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, round(im.height*maxw/im.width)))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=q)
    uri = "data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()
    _cache[key] = uri
    return uri

YN = {"1": "Yes", "0": "No", None: "—"}
def yn(v): return YN.get(v, "—")

def cls_and_tag(r):
    ct = r.get("change_type")
    c0, c1 = r.get("is_correct_turn0"), r.get("is_correct")
    if ct == "wrong_right": return "flip-good", "✗ → ✓  corrected"
    if ct == "right_wrong": return "flip-bad",  "✓ → ✗  broke"
    if c0 == 1 and c1 == 1:  return "both-ok",   "✓ → ✓"
    if c0 == 0 and c1 == 0:  return "both-bad",  "✗ → ✗"
    return "unparsed", "unparsed"

# ── group + order (flips first within each category) ──────────────────────────
groups = defaultdict(list)
for r in forced:
    groups[r["subcategory"]].append(r)
rank = {"wrong_right":0, "right_wrong":1}
for k in groups:
    groups[k].sort(key=lambda r: rank.get(r.get("change_type"), 2))

# per-category summary
def acc(rs, f):
    v=[r[f] for r in rs if r.get(f) is not None]
    return sum(v)/len(v)*100 if v else float("nan")
cat_order = sorted(groups, key=lambda k: -(acc(groups[k],"is_correct")-acc(groups[k],"is_correct_turn0")))

# ── build HTML ────────────────────────────────────────────────────────────────
parts = []
parts.append("""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forced Re-examination — all turn-0 vs turn-1 traces</title><style>
body{max-width:1000px;margin:30px auto;padding:0 20px;font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1a1a1a}
h1{border-bottom:2px solid #eee;padding-bottom:.3em}
h2{border-bottom:1px solid #eee;padding-bottom:.2em;margin-top:1.8em}
table{border-collapse:collapse;margin:1em 0}th,td{border:1px solid #ddd;padding:5px 10px;text-align:center}th{background:#f7f7f7}
details.sample{border:1px solid #e2e2e2;border-radius:7px;margin:8px 0;padding:4px 10px}
details.sample summary{cursor:pointer;font-weight:600;padding:5px 2px}
details.sample[open]{background:#fcfcfc}
.flip-good summary{border-left:6px solid #31a354;padding-left:8px}
.flip-bad  summary{border-left:6px solid #de2d26;padding-left:8px}
.both-ok   summary{border-left:6px solid #c6dbef;padding-left:8px}
.both-bad  summary{border-left:6px solid #969696;padding-left:8px}
.unparsed  summary{border-left:6px solid #fdae6b;padding-left:8px}
.tag{font-size:.8em;color:#666;font-weight:400}
.body img{border:1px solid #eee;border-radius:6px;margin:8px 0;display:block}
.turn{margin:10px 0}.turn b{display:block;margin-bottom:3px}
pre{background:#f6f8fa;border:1px solid #e3e3e3;border-radius:6px;padding:10px 12px;white-space:pre-wrap;font-size:12.5px;line-height:1.45;overflow:auto;max-height:420px}
.idx a{margin-right:14px;white-space:nowrap}
.meta{color:#444;margin:.2em 0}
</style></head><body>""")
parts.append("<h1>Forced re-examination — every turn-0 vs turn-1 trace, by category</h1>")
parts.append('<p class="meta">Each sample: the model answers (turn&nbsp;0), the <b>same image is '
             're-injected</b>, and it answers again (turn&nbsp;1). Cards are collapsed — click to expand. '
             'Colour: <span style="color:#31a354">green = corrected (✗→✓)</span>, '
             '<span style="color:#de2d26">red = broke (✓→✗)</span>, grey/blue = unchanged.</p>')

# summary table
parts.append("<h2>Summary</h2><table><tr><th>category</th><th>n</th><th>turn-0 acc</th>"
             "<th>turn-1 acc</th><th>gain</th><th>✗→✓</th><th>✓→✗</th></tr>")
for k in cat_order:
    rs = groups[k]
    a0, a1 = acc(rs,"is_correct_turn0"), acc(rs,"is_correct")
    wr = sum(1 for r in rs if r.get("change_type")=="wrong_right")
    rw = sum(1 for r in rs if r.get("change_type")=="right_wrong")
    parts.append(f'<tr><td><a href="#{k}">{k}</a></td><td>{len(rs)}</td><td>{a0:.1f}%</td>'
                 f'<td>{a1:.1f}%</td><td>{a1-a0:+.1f}pp</td><td>{wr}</td><td>{rw}</td></tr>')
parts.append("</table>")
parts.append('<p class="idx"><b>Jump to:</b> ' +
             " ".join(f'<a href="#{k}">{k}</a>' for k in cat_order) + "</p>")

# per-category sections
for k in cat_order:
    parts.append(f'<h2 id="{k}">{k} <span class="tag">({len(groups[k])} samples)</span></h2>')
    for r in groups[k]:
        cls, tag = cls_and_tag(r)
        q = html.escape(r["question"])
        summ = (f'{tag} &nbsp; <span class="tag">[{r["set_id"]}_{r["figure_id"]}, gt={yn(r["gt_answer"])}]</span> '
                f'&nbsp; {q}')
        parts.append(f'<details class="sample {cls}"><summary>{summ}</summary><div class="body">')
        uri = img_uri(r)
        if uri:
            parts.append(f'<img src="{uri}" alt="image">')
        else:
            parts.append('<p class="meta"><i>(image unavailable)</i></p>')
        parts.append(f'<p class="meta"><b>Q:</b> {q}<br><b>Ground truth:</b> {yn(r["gt_answer"])}</p>')
        t0, t1 = r.get("thinking_per_stage", ["", ""])[:2] + ["", ""][:0] if r.get("thinking_per_stage") else ("","")
        ts = r.get("thinking_per_stage") or ["",""]
        t0 = ts[0] if len(ts)>0 else ""
        t1 = ts[1] if len(ts)>1 else ""
        c0mark = "correct" if r.get("is_correct_turn0")==1 else ("wrong" if r.get("is_correct_turn0")==0 else "unparsed")
        c1mark = "correct" if r.get("is_correct")==1 else ("wrong" if r.get("is_correct")==0 else "unparsed")
        parts.append(f'<div class="turn"><b>Turn 0 — answer: {yn(r.get("pred_turn0"))} ({c0mark})</b>'
                     f'<pre>{html.escape(t0.strip())}</pre></div>')
        parts.append(f'<div class="turn"><b>Turn 1 — answer: {yn(r.get("pred_turn1"))} ({c1mark}) '
                     f'&nbsp;[image re-injected]</b><pre>{html.escape(t1.strip())}</pre></div>')
        parts.append("</div></details>")

parts.append("</body></html>")
out = SD/"forced_traces.html"
out.write_text("\n".join(parts))
print(f"wrote {out}  ({out.stat().st_size//1024} KB, {len(forced)} samples, "
      f"{sum(1 for v in _cache.values() if v)} images)")
