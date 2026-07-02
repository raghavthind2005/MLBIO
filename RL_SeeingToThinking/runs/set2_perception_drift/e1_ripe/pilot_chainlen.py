#!/usr/bin/env python3
"""
Set-2 Step-0.5 PILOT — does off-the-shelf Qwen3-VL-4B-Thinking produce LONG,
IMAGE-GROUNDED, NON-TRUNCATED reasoning chains on CLEVR compositional questions,
and does chain length scale with question difficulty?

GATE only (PLAN.md R1) — verifies the substrate + harness; makes NO scientific claim.

Fluke-risk checks (search "CHECK"):
  CHECK-ENV/MODEL/TMPL/IMG : right libs, right weights (36 layers), thinking template,
                             image actually fed.
  CHECK-TRUNC : generation did NOT hit the token cap (else "short chain" is a truncation fake).
  CHECK-CLOSE : chain closed with '</think>' (a real, complete chain).
  CHECK-GROUND: reasoning references real scene vocabulary (looking, not guessing).
  CHECK-BLANK : gray-image swap changes the answer (image is USED, not a language prior).

Reads a LOCAL parquet + LOCAL model → fully offline. Saves results.json.
"""
import os, sys, io, re, json, time, random
import numpy as np

MODEL = "/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking"
# compositional CLEVR (attribute queries / comparisons / multi-hop). NOT counting-only.
DATA  = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/data/clevr_cogent_valA.parquet"
OUT   = "/iopsstor/scratch/cscs/raghavthind/set2_pilot/out"
os.makedirs(OUT, exist_ok=True)

IMAGE_PAD_ID = 151655
N_ITEMS, N_LEVELS, REPEAT_ITEMS, BLANK_ITEMS = 20, 5, 3, 2
MAX_NEW = 8192   # generous; README out_seq_length=32768. Any hit_cap is flagged as truncation.
GEN = dict(max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0, top_p=0.95, top_k=20, repetition_penalty=1.0)

CLEVR_VOCAB = set("""gray grey red blue green brown purple cyan yellow
cube cubes block blocks sphere spheres ball balls cylinder cylinders
large small big tiny metal metallic shiny rubber matte behind front left right""".split())

def log(*a): print(*a, flush=True)

# ------------------------------------------------------------------ ENV
log("="*70, "\nCHECK-ENV")
import torch, transformers
log(f"python {sys.version.split()[0]}  torch {torch.__version__}  transformers {transformers.__version__}")
assert torch.cuda.is_available(), "no CUDA"
log(f"gpu {torch.cuda.get_device_name(0)}  bf16={torch.cuda.is_bf16_supported()}")
assert tuple(int(x) for x in transformers.__version__.split('.')[:2]) >= (4, 56)

# ------------------------------------------------------------------ DATA (fail-fast)
log("="*70, "\nDATA")
import pyarrow.parquet as pq
from PIL import Image
tbl = pq.read_table(DATA)
cols = tbl.column_names
log(f"parquet cols={cols}  rows={tbl.num_rows}")
QCOL = next(c for c in ("problem", "question", "query") if c in cols)
ACOL = next(c for c in ("solution", "answer", "label") if c in cols)
ICOL = next(c for c in ("images", "image", "img") if c in cols)
log(f"using  question='{QCOL}'  answer='{ACOL}'  image='{ICOL}'")
df = tbl.to_pandas()

def load_img(cell):
    x = cell
    if isinstance(x, (list, np.ndarray)) and len(x) and not isinstance(x, (bytes, dict, str)):
        x = x[0]
    if isinstance(x, dict):
        if x.get("bytes") is not None: return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
        if x.get("path"):              return Image.open(x["path"]).convert("RGB")
    raise ValueError(f"cannot decode image cell type {type(cell)} -> {str(x)[:80]}")

def clean_q(p):
    p = str(p).replace("<image>", " ").strip()
    p = re.sub(r"(?is)\b(output|put|provide|answer)\b.{0,40}?(<answer>|\\boxed|inside|tag).*$", "", p).strip()
    return p

def norm(s):
    s = re.sub(r"<[^>]+>", " ", str(s).lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()

def proxy(q):
    toks = re.findall(r"[a-z]+", q.lower())
    return sum(t in CLEVR_VOCAB for t in toks) + q.lower().count("same") + q.lower().count("other")

# verify format on row 0
log(f"raw  problem[0] : {repr(df[QCOL].iloc[0])[:200]}")
log(f"clean         : {repr(clean_q(df[QCOL].iloc[0]))[:200]}")
log(f"raw  solution[0]: {repr(df[ACOL].iloc[0])[:120]}   norm='{norm(df[ACOL].iloc[0])}'")
_im0 = load_img(df[ICOL].iloc[0]); log(f"image[0]        : {_im0.size} {_im0.mode}")

pool = [dict(idx=int(i), q=clean_q(df[QCOL].iloc[i]), ans=norm(df[ACOL].iloc[i]),
             prox=proxy(clean_q(df[QCOL].iloc[i]))) for i in range(len(df))]
pv = sorted(p["prox"] for p in pool)
log(f"proxy distribution: min={pv[0]} p25={pv[len(pv)//4]} median={pv[len(pv)//2]} p75={pv[3*len(pv)//4]} max={pv[-1]}")

# selection: try quantile bins; if degenerate, fall back to rank-even spacing over sorted-by-proxy
rng = random.Random(0)
chosen, per = [], max(1, N_ITEMS // N_LEVELS)
edges = [pv[int(k*(len(pv)-1)/N_LEVELS)] for k in range(N_LEVELS+1)]
if len(set(edges)) > 2:
    for lv in range(N_LEVELS):
        lo, hi = edges[lv], edges[lv+1]
        b = [p for p in pool if (lo <= p["prox"] < hi) or (lv == N_LEVELS-1 and p["prox"] >= hi)]
        rng.shuffle(b)
        for p in b[:per]: p["level"] = lv; chosen.append(p)
if len(chosen) < N_ITEMS:                      # rank-even fallback guarantees a spread
    srt = sorted(pool, key=lambda p: p["prox"])
    picks = [srt[int(round(j*(len(srt)-1)/(N_ITEMS-1)))] for j in range(N_ITEMS)]
    chosen = []
    for j, p in enumerate(picks): p["level"] = j*N_LEVELS//N_ITEMS; chosen.append(p)
chosen = chosen[:N_ITEMS]
log(f"selected {len(chosen)} items; proxies={[p['prox'] for p in chosen]}")

# ------------------------------------------------------------------ MODEL
log("="*70, "\nCHECK-MODEL")
from transformers import AutoProcessor, AutoModelForImageTextToText
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
tok = processor.tokenizer
n_layers = model.config.text_config.num_hidden_layers
log(f"class {type(model).__name__}  dtype {next(model.parameters()).dtype}  "
    f"layers {n_layers}  params {sum(p.numel() for p in model.parameters())/1e9:.2f}B  load {time.time()-t0:.0f}s")
assert n_layers == 36, "wrong model?"

def build_inputs(q, image):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text, processor(text=[text], images=[image], return_tensors="pt").to("cuda")

# CHECK-TMPL / CHECK-IMG on first item
log("="*70, "\nCHECK-TMPL / CHECK-IMG")
_im = load_img(df[ICOL].iloc[chosen[0]["idx"]])
_txt, _inp = build_inputs(chosen[0]["q"], _im)
log(f"templated tail : {repr(_txt[-70:])}")
assert _txt.rstrip().endswith("<think>"), "Thinking template did not auto-open <think>!"
n_imgpad = int((_inp["input_ids"] == IMAGE_PAD_ID).sum())
log(f"image_pad toks={n_imgpad}  grid_thw={_inp.get('image_grid_thw').tolist() if 'image_grid_thw' in _inp else 'NA'}  "
    f"prompt_len={_inp['input_ids'].shape[1]}")
assert n_imgpad > 0, "image not fed!"

# ------------------------------------------------------------------ generation
def generate(q, image, seed):
    transformers.set_seed(seed)
    _, inputs = build_inputs(q, image)
    in_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, **GEN)
    gen_ids = out[0][in_len:]
    text = tok.decode(gen_ids, skip_special_tokens=False)
    has_close = "</think>" in text
    think = text.split("</think>")[0]
    after = text.split("</think>", 1)[1] if has_close else ""
    chain_tok = len(tok(think, add_special_tokens=False).input_ids)
    ground = sum(w in CLEVR_VOCAB for w in re.findall(r"[a-z]+", think.lower()))
    mm = re.findall(r"\\boxed\{([^}]*)\}", text)
    boxed = mm[-1].strip() if mm else None
    pred = norm(boxed) if boxed else norm(after)
    return dict(seed=seed, n_gen=int(gen_ids.shape[0]), hit_cap=int(gen_ids.shape[0]) >= MAX_NEW,
                has_close=has_close, chain_tok=chain_tok, ground_hits=ground, boxed=boxed,
                pred=pred, think_head=think[:200], think_tail=think[-200:], after=after[:120])

# ------------------------------------------------------------------ main sweep
log("="*70, "\nGENERATION (graded items)")
results = []
for k, it in enumerate(chosen):
    try:
        im = load_img(df[ICOL].iloc[it["idx"]]); t0 = time.time()
        r = generate(it["q"], im, seed=1000+k)
        r.update(k=k, idx=it["idx"], level=it["level"], prox=it["prox"], q=it["q"], gt=it["ans"],
                 img=im.size, secs=round(time.time()-t0, 1))
        gt = it["ans"]; r["correct"] = bool(gt) and (gt in r["pred"].split() or gt == r["pred"] or r["pred"] in gt.split())
        results.append(r)
        log(f"[{k:02d}] px{it['prox']:>2} chain_tok={r['chain_tok']:>5} close={int(r['has_close'])} "
            f"cap={int(r['hit_cap'])} grnd={r['ground_hits']:>2} pred='{r['pred'][:14]}' gt='{gt[:14]}' "
            f"ok={int(r['correct'])} {r['secs']}s | {it['q'][:46]}")
    except Exception as e:
        log(f"[{k:02d}] ERROR {type(e).__name__}: {e}")

# ------------------------------------------------------------------ CHECK-BLANK
log("="*70, "\nCHECK-BLANK (real vs gray image, same seed → image is the only variable)")
blank = []
for k in range(min(BLANK_ITEMS, len(chosen))):
    it = chosen[k]; im = load_img(df[ICOL].iloc[it["idx"]])
    rr = generate(it["q"], im, 1); rb = generate(it["q"], Image.new("RGB", im.size, (128,128,128)), 1)
    d = (rr["pred"] != rb["pred"]) or (rr["after"] != rb["after"])
    blank.append(dict(k=k, q=it["q"], gt=it["ans"], real=rr["pred"], blank=rb["pred"], differs=bool(d)))
    log(f"[blank {k}] gt='{it['ans'][:14]}' real='{rr['pred'][:14]}' blank='{rb['pred'][:14]}' differs={bool(d)} | {it['q'][:46]}")

# ------------------------------------------------------------------ variance
log("="*70, "\nSAMPLING VARIANCE (3 items x 3 seeds)")
variance = []
for k in range(min(REPEAT_ITEMS, len(chosen))):
    it = chosen[k]; im = load_img(df[ICOL].iloc[it["idx"]])
    lens = [generate(it["q"], im, s)["chain_tok"] for s in (11, 22, 33)]
    variance.append(dict(k=k, q=it["q"], chain_toks=lens))
    log(f"[var {k}] chain_toks={lens} spread={max(lens)-min(lens)} | {it['q'][:46]}")

# ------------------------------------------------------------------ summary
log("="*70, "\nSUMMARY")
if results:
    ct = [r["chain_tok"] for r in results]
    trunc = sum(r["hit_cap"] for r in results); noclose = sum(not r["has_close"] for r in results)
    log(f"chain_tok  min={min(ct)} median={int(np.median(ct))} mean={int(np.mean(ct))} max={max(ct)}")
    log(f"CHECK-TRUNC hit_cap={trunc}/{len(results)}   CHECK-CLOSE no_close={noclose}/{len(results)}")
    log(f"grounded(>=3 vocab): {sum(r['ground_hits']>=3 for r in results)}/{len(results)}   "
        f"correct: {sum(r['correct'] for r in results)}/{len(results)}")
    bl = {}
    for r in results: bl.setdefault(r["level"], []).append(r["chain_tok"])
    for lv in sorted(bl): log(f"  level {lv}: n={len(bl[lv])} median_chain_tok={int(np.median(bl[lv]))} "
                              f"median_prox={int(np.median([r['prox'] for r in results if r['level']==lv]))}")
    log(f"blank-control differs: {sum(b['differs'] for b in blank)}/{len(blank)} (want ALL → image USED)")
    json.dump(dict(gen=GEN, model=MODEL, data=DATA, n=len(results), results=results, blank=blank,
                   variance=variance, summary=dict(min=min(ct), median=int(np.median(ct)), max=max(ct),
                   hit_cap=trunc, no_close=noclose)), open(f"{OUT}/pilot_results.json","w"), indent=2, default=str)
    log(f"saved -> {OUT}/pilot_results.json")
else:
    log("NO RESULTS — all items errored")
