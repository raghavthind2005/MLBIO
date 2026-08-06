"""
Probe A — shared machinery: config, data, scorer, prompts, arms, provenance, resume.

Every constant here is traceable to an authoritative source; see PROBE_A_DECISIONS_FOR_APPROVAL.md.
NOTHING in this file may be changed after the smoke is signed off without a logged amendment.
"""
import os, io, re, json, glob, time, hashlib, subprocess

USER = os.environ.get("USER", "raghavthind")
SCRATCH = f"/iopsstor/scratch/cscs/{USER}"
TP = f"{SCRATCH}/text_privilege"
DATA = f"{TP}/data/mmstar"
OUT = f"{TP}/out"

# ---------------------------------------------------------------- models
SHARED = "/capstor/store/cscs/swissai/a0174/models"
MODELS = {
    "thinking": f"{SHARED}/Qwen3-VL-4B-Thinking",
    "instruct": f"{SHARED}/Qwen3-VL-4B-Instruct",
    "caprl":    f"{SCRATCH}/models/CapRL-Qwen3VL-4B",
}

# ---------------------------------------------------------------- decode
# Source: each model card's **Multimodal** best-practice block (all arms carry an image).
# `presence_penalty` and `max_tokens` exist ONLY in the cards, not in generation_config.json
# (D11d). Thinking's block is byte-identical to Track-T's frozen recipe.
DECODE = {
    "thinking": dict(temperature=1.0, top_p=0.95, top_k=20,
                     repetition_penalty=1.0, presence_penalty=0.0, max_tokens=40960),
    "instruct": dict(temperature=0.7, top_p=0.80, top_k=20,
                     repetition_penalty=1.0, presence_penalty=1.5, max_tokens=16384),
    # Q6 OPEN: CapRL-Qwen3VL-4B has NO model-specific recommendation. This is candidate A =
    # its shipped generation_config (== the Instruct base), with presence_penalty deliberately 0.0
    # because a presence penalty punishes token reuse, which is harmful for dense captioning and
    # was absent during CapRL's GRPO training. Candidate B is the README's CapRL-3B example
    # (temp 1.0 / top_p 1.0). RESOLVED BY MEASUREMENT AT THE SMOKE, not by this default.
    "caprl":    dict(temperature=0.7, top_p=0.80, top_k=20,
                     repetition_penalty=1.0, presence_penalty=0.0, max_tokens=4096),
}
CAPRL_DECODE_CANDIDATES = {
    "A_genconfig": dict(temperature=0.7, top_p=0.80, top_k=20,
                        repetition_penalty=1.0, presence_penalty=0.0, max_tokens=4096),
    "B_readme3b":  dict(temperature=1.0, top_p=1.00, top_k=-1,
                        repetition_penalty=1.0, presence_penalty=0.0, max_tokens=4096),
    "C_base_pp":   dict(temperature=0.7, top_p=0.80, top_k=20,
                        repetition_penalty=1.0, presence_penalty=1.5, max_tokens=4096),
}
# BUG 1 (found in pre-smoke re-check): Track-T's 40960/49152 assumed a SHORT payload. Here the
# payload is a CapRL caption of up to `caprl.max_tokens` = 4096 tokens. Worst case for T1:
#   question <=500 + hint ~15 + wrapper ~15 + caption <=4096  = ~4626 text tokens
#   + image up to 6591 (measured MMStar max)                  = ~11217 prompt tokens
#   + max_tokens 40960                                        = ~52177 > 49152  -> assert fires.
# Raised with headroom (Set-3 precedent: 40960 -> 61440 for exactly this reason).
MAX_MODEL_LEN = {"thinking": 61440, "instruct": 32768, "caprl": 16384}
IMG_TOK_MAX = 6591          # measured over all 1500 MMStar images (D11c)
SEED = 0

# ---------------------------------------------------------------- prompts
# CapRL's own RL training instruction, verbatim from the `prompt` column of CapRL-QA-75K.
# Using anything else puts the captioner off-distribution from its own training (D11d).
CAPTION_PROMPT = "Please describe this image in detail."

# ---- QUESTION-CONDITIONED caption (arms T3/I3), added 2026-08-06 pre-outcome ------------------
# Motivation: the peek showed the question-BLIND caption is high quality yet can miss the queried
# fact entirely -- MMStar item 0 asks about "the book", the caption calls it "a magazine or
# brochure" and places it "in front of" rather than on/beneath. A null on T1 would therefore only
# license "question-blind dense captioning isn't the lever", not "articulation isn't the lever".
#
# STRUCTURAL anti-leak measure (stronger than any instruction): the captioner is given the
# question STEM ONLY -- the options are stripped. It cannot name a choice it has never seen.
# Wording of the prohibition follows Track-T's self-description prompt, which achieved a measured
# leak rate of 2/2485 (0.08%).
CAPTION_PROMPT_Q = (
    "Look carefully at the image and describe what is actually visible in it, paying closest "
    "attention to everything that bears on the question below. Report the concrete visual "
    "details: objects, their attributes, their positions, the spatial relations between them, "
    "any text, numbers or labels that appear, and how the elements are arranged relative to one "
    "another. Be thorough: include every detail that could bear on the question, and also "
    "anything else you notice.\n\n"
    "Describe only what you observe. Do not answer the question, do not perform any calculation "
    "or inference, and do not state, imply, or guess a conclusion.\n\n"
    "Question: {stem}\n\n"
    "Give your description as a plain list of short factual statements, one per line."
)

def question_stem(question):
    """
    Question text with the options block REMOVED -- the structural anti-leak measure for T3/I3.
    Handles BOTH formats (Options:/Choices:) and also strips format-2's "Hint:" preamble, which
    itself tells the model to output an option letter and would defeat the point.
    """
    q = str(question)
    m = _MARKER.search(q)
    cut = m.start() if m else _options_span(q)[1]   # format 3 has no marker; cut at the first option
    if cut is not None:
        q = q[:cut]
    q = re.sub(r"^\s*Hint\s*:.*?\n", "", q, flags=re.S | re.I)
    q = re.sub(r"^\s*Question\s*:\s*", "", q, flags=re.I)
    return q.strip()

# Frozen payload wrapper. Byte-identical across T1/T2/I1/I2 so those arms differ ONLY in content.
# Adapted from Track-T's frozen "From the figure…" (MMStar includes natural images, not only figures).
WRAPPER = "From the image, I can see the following:\n{payload}\n"

# S1 RESOLVED 2026-08-06 from the cloned source: MMStar/eval/vlmeval/utils/dataset.py:97-99.
# Both strings below are VERBATIM from that builder (note the trailing spaces before \n).
#
# Deviation, declared: upstream appends these only when the dataframe has separate A/B/C/D
# columns, and it gates the second string behind `for_llm`. Our MMStar release (parquet AND the
# official TSV) carries the options INLINE in `question` and has no A/B/C/D columns, so the
# upstream builder would emit no instruction at all for a VLM. We append both, verbatim,
# IDENTICALLY IN EVERY ARM. It therefore cannot confound any contrast (it only sets the absolute
# level), and it materially reduces false negatives under the position-0-anchored official scorer.
#
# 2026-08-06 AMENDMENT (pre-outcome, no generation has run): we ALSO require a \boxed{} answer.
# Rationale in tp_common's scorer section. Legitimacy note: changing the instrument is fine now
# precisely because no outcome data exists; doing it after seeing results would be the violation.
ANSWER_HINT = ("Please select the correct answer from the options above. \n"
               "Answer with the option's letter from the given choices directly, "
               "such as answer letter 'A' only. \n"
               "Put your final answer letter in \\boxed{}. \n")

ARMS = {
    #  arm : (model_key, payload_kind)
    "T0": ("thinking", None),      "T1": ("thinking", "caption"),  "T2": ("thinking", "placebo"),
    "I0": ("instruct", None),      "I1": ("instruct", "caption"),  "I2": ("instruct", "placebo"),
    # question-conditioned caption. Its control is T1/I1 (both are true captions of the true
    # image; only TARGETING differs), so it needs no placebo of its own -- T2/I2 already control
    # for "any long authoritative prefill".
    "T3": ("thinking", "caption_q"), "I3": ("instruct", "caption_q"),
    "A5": ("caprl",    None),      # captioner answering MMStar itself, identical official prompt (Q4)
}

# Interpretation split that must be pre-stated, because it is a property of the task not the data:
# for a PERCEPTION question, a correctly targeted description IS very close to the answer -- there
# is no clean line between "described the relevant fact" and "leaked it". For a REASONING question
# (logical/math/science), targeted description is genuinely upstream of the answer. So T3-T1 is
# near-tautological on the perception axes and genuinely informative on the reasoning axes, and
# the two must be reported separately rather than pooled.
REASONING_CATS = {"logical reasoning", "math", "science & technology", "instance reasoning"}
PERCEPTION_CATS = {"coarse perception", "fine-grained perception"}

# ---------------------------------------------------------------- data
def load_mmstar(limit=0):
    """
    1500 rows: index, question, answer, category, l2_category, image, meta_info.

    `limit` takes a CATEGORY-STRATIFIED, seeded subset -- NOT df.head(). MMStar is ordered by
    category (250 each), so a head-slice would put every smoke item in one category, which would
    (a) leave the placebo's same-category matching untested and (b) be unrepresentative of the
    perception subset. Stratified keeps all 6 categories present at any limit >= 6.
    """
    import pandas as pd
    df = pd.read_parquet(f"{DATA}/mmstar.parquet")
    assert len(df) == 1500, f"expected 1500 MMStar rows, got {len(df)}"
    need = {"index", "question", "answer", "category", "l2_category", "image"}
    assert need <= set(df.columns), f"missing columns: {need - set(df.columns)}"
    if limit and limit < len(df):
        per = max(2, limit // df["category"].nunique())   # >=2 so same-category donors exist
        parts = [g.sort_values("index").head(per) for _, g in df.groupby("category", sort=True)]
        df = pd.concat(parts).sort_values("index").reset_index(drop=True)
    return df

def image_bytes(cell):
    return cell["bytes"] if isinstance(cell, dict) else cell

def pil_image(cell):
    from PIL import Image
    return Image.open(io.BytesIO(image_bytes(cell))).convert("RGB")

# MMStar ships TWO question formats (found by smoke gate G2b, which failed 346/1500 on a
# parser built for format 1 only):
#
#   format 1 (1154 items):  "...question...\nOptions: A: xxx, B: yyy, C: zzz, D: www"
#   format 2 ( 346 items):  "Hint: ...\nQuestion: ...\nChoices:\n(A) xxx\n(B) yyy\n(C) ...\n(D) ..."
#
# Format 2 is concentrated in math (207) and logical reasoning (113) -- MathVista-derived items --
# and uses "Choices:" with "(A)" rather than "Options:" with "A:". Had this slipped through,
# can_infer_text and the value-box mapping would have had no choices for 23% of the data, and
# disproportionately on the REASONING axes, which is precisely where T3-T1 is interpretable.
_MARKER = re.compile(r"\b(?:Options|Choices)\s*:", re.I)
# accepts "A: ", "(A) ", "A. " -- all three appear
_OPT_SPLIT = re.compile(r"(?:^|[\n,;\s])\(?([A-D])(?:\)|:|\.)\s")

def _options_span(question):
    """
    -> (choices dict, char offset in `question` where the options block begins) or ({}, None).

    Format 3 (found by G2b after the format-2 fix) has NO marker at all -- the options simply
    follow the question on their own lines. So: search after the marker when there is one, else
    search the whole string. Among candidate A,B,C,D runs keep the LONGEST, breaking ties by the
    LATEST start, since the real block is the last thing in the question and a stray "(A)" in the
    question text would otherwise capture the run.
    """
    q = str(question)
    m = _MARKER.search(q)
    base = m.end() if m else 0
    body = q[base:]
    hits = list(_OPT_SPLIT.finditer(body))
    best = []
    for i, h0 in enumerate(hits):
        if h0.group(1) != "A":
            continue
        run, want = [h0], "B"
        for h in hits[i + 1:]:
            if h.group(1) == want:
                run.append(h)
                want = chr(ord(want) + 1)
                if want > "D":
                    break
        if len(run) > len(best) or (len(run) == len(best) and run[0].start() > best[0].start()):
            best = run
    if len(best) < 2:
        return {}, None
    out = {}
    for i, h in enumerate(best):
        end = best[i + 1].start() if i + 1 < len(best) else len(body)
        out[h.group(1)] = body[h.end():end].strip().rstrip(",;").strip()
    return out, base + best[0].start()

def parse_choices(question):
    """-> dict {'A': text, ...}. Empty dict if no options block is present/parseable."""
    return _options_span(question)[0]

# ---------------------------------------------------------------- scorer
# S2 RESOLVED 2026-08-06 by reading the VERBATIM source (cloned to $TP/vendor/MMStar).
#
# ⚠ MAJOR CORRECTION: MMStar does NOT use VLMEvalKit's generic `can_infer`. It ships its OWN
# scorer, `MMStar_eval` in eval/vlmeval/evaluate/mmstar.py, which is far stricter and
# ANCHORED AT POSITION 0 of the prediction. An earlier reconstruction of `can_infer` (written from
# web summaries) invented three rules that do not exist -- a "last 4 words" window, an
# "answer is ([ABCD])" regex, and a 2x-length rejection -- and would have produced
# systematically different, ARM-DEPENDENT accuracy. Both scorers below are now transcribed from
# the cloned source.

# ---- PRIMARY: MMStar's own official scorer, semantics transcribed verbatim -------------------
#   answer  = gt.lower().strip().replace('\n',' ')
#   predict = pred.lower().strip().replace('\n',' ')
#   credit iff  predict[0]==answer  or  '('+answer  or  'option '+answer  or 'the answer is '+answer
# Note how unforgiving this is: "Based on the image, B" scores WRONG. This is why the official
# answer-format instruction (ANSWER_HINT) is load-bearing, and why the per-arm format rate is a
# first-class reported number -- verbosity differs by arm, so format failures are a confound.
def score_mmstar_official(prediction, gt):
    answer = str(gt).lower().strip().replace("\n", " ")
    predict = str(prediction).lower().strip().replace("\n", " ")
    try:
        if answer == predict[0]:
            return 1
        if predict[0] == "(" and answer == predict[1]:
            return 1
        if predict[0:7] == "option " and answer == predict[7]:
            return 1
        if predict[0:14] == "the answer is " and answer == predict[14]:
            return 1
    except Exception:
        pass
    return 0

def official_firstchar_artifact(prediction, gt):
    """
    ⚠ A REAL DEFECT IN THE OFFICIAL SCORER, discovered by unit-testing it (2026-08-06).

    `answer == predict[0]` compares only the FIRST CHARACTER, so any prose opener whose first
    letter happens to equal the gold letter is credited:
        gold B + "Based on the image..."   -> credited
        gold B + "Between B and C..."      -> credited
        gold C + "Considering the..."      -> credited
        gold A + "According to the chart"  -> credited
    With MMStar's non-uniform gold labels (B447 A429 D315 C309) and common English openers, this
    is not a rare edge case.

    Why it matters HERE specifically: our arms differ in what precedes the answer, so response
    STYLE may differ by arm -- meaning the false-positive rate of the primary metric can differ by
    arm for reasons unrelated to perception. That is a confound in the metric itself, so we detect
    and report it per arm rather than letting it ride.

    Returns 1 when official credited but the prediction is NOT a clean letter answer.
    """
    answer = str(gt).lower().strip().replace("\n", " ")
    predict = str(prediction).lower().strip().replace("\n", " ")
    if not score_mmstar_official(predict, answer):
        return 0
    if predict[:1] == answer and (len(predict) == 1 or not predict[1].isalnum()):
        return 0                                   # bare letter, optionally punctuated
    if predict[:1] == "(" or predict[:7] == "option " or predict[:14] == "the answer is ":
        return 0                                   # explicit, unambiguous forms
    return 1

# ---- SECONDARY (pre-registered sensitivity): VLMEvalKit can_infer, now verbatim --------------
# Deliberately NOT the primary: it is a different function from the one MMStar scores with. It is
# carried so we can report how much of any arm difference is a pure answer-FORMAT artifact of the
# strict position-0 rule (the Track-T §11.8 pattern: strict primary + tolerant sensitivity).
_REJECT = ["Sorry, I can't help with images of people yet.",
           "I can't process this file.",
           "I'm sorry, but without the image provided",
           "Cannot determine the answer"]
_PUNCT = ".()[],:;!*#{}"

def _count_choice(splits, choices):
    return sum(1 for c in choices if c in splits)

def can_infer_option(answer, choices):
    if "Failed to obtain answer via API" in answer:
        return False
    for err in _REJECT:
        if err in answer:
            return "Z"
    mod = answer
    for c in _PUNCT:
        mod = mod.replace(c, " ")
    splits = [x.strip() for x in mod.split()]
    count = _count_choice(splits, choices)
    # NOTE: upstream has a `verbose`-gated branch here that returns False when 'A' may be an
    # article. It is a no-op unless the VERBOSE env var is set, so it is omitted; the consequence
    # is that a stray article "A" CAN be credited as choice A. That is upstream behaviour, not our
    # deviation -- and it is measured per arm by the format diagnostics in Pass 3.
    if count == 1:
        for ch in choices:
            if ch in splits:
                return ch
    elif count == 0 and _count_choice(splits, {"Z", ""}) == 1:
        return "Z"
    return False

def can_infer_text(answer, choices):
    # Upstream MUTATES the caller's dict (lowercases values in place); we copy instead. Semantics
    # are identical; the copy just prevents a cross-call landmine since we reuse one choices dict.
    a = answer.lower()
    cands = [k for k in choices if str(choices[k]).lower() in a]
    return cands[0] if len(cands) == 1 else False

def can_infer(answer, choices):
    copt = can_infer_option(str(answer), choices)
    return copt if copt else can_infer_text(str(answer), choices)

# ---- BOXED EXTRACTION: the project's own audited instrument -----------------------------------
# Same balanced-brace walk as Set-2/3 `common.py:extract_boxed` (self-tested 599/599) and Track-T's
# `mv_score.py` (0-FP asserted). Balanced-brace, not a regex, so nested braces (\boxed{\text{B}})
# survive. Takes the LAST \boxed{ in the text.
def extract_boxed(text):
    key = "\\boxed{"
    i = text.rfind(key)
    if i < 0:
        return None
    j, depth = i + len(key), 1
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + len(key):j]
        j += 1
    return None            # unbalanced (usually a truncated generation)

_LATEX = ("\\text", "\\mathrm", "\\mathbf", "\\rm", "\\bf")

def _canon_box(s):
    if s is None:
        return None
    t = s.strip()
    for w in _LATEX:
        t = t.replace(w, "")
    return t.strip().strip("{}").strip().strip("().,:;$ ").strip()

def score_boxed(text, choices, gt):
    """
    PRIMARY. -> (correct, kind, payload)
      kind = 'letter'  a bare A-D was boxed                       -> scored strictly
             'value'   the boxed content matches one option's TEXT -> counted, credited only in
                       the pre-registered TOLERANT metric (Track-T §11.8 pattern)
             'other'   boxed something uninterpretable
             'none'    no \\boxed{} at all  (or truncated mid-box)
    """
    raw = extract_boxed(text)
    if raw is None:
        return 0, "none", None
    c = _canon_box(raw)
    if c and len(c) == 1 and c.upper() in "ABCD":
        return int(c.upper() == gt), "letter", c.upper()
    if choices:
        cl = (c or "").lower()
        hits = [k for k, v in choices.items() if str(v).strip() and str(v).lower() == cl]
        if not hits:
            hits = [k for k, v in choices.items() if str(v).strip() and str(v).lower() in cl]
        if len(hits) == 1:
            return 0, "value", hits[0]     # NOT credited by the primary; the tolerant metric does
    return 0, "other", c

def post_think(text):
    """Thinking models emit <think>…</think> then the answer. Score only what follows."""
    return text.split("</think>", 1)[1] if "</think>" in text else text

def score_item(raw_text, choices, gt, is_thinking):
    """
    THREE metrics on the SAME generation. Nothing is re-generated to produce them, so reporting
    all three costs nothing and lets the conclusion be checked under each.

      correct           PRIMARY     — boxed letter, strict. No silent false-positive mode: a
                                      missing/!letter box is WRONG and is counted.
      correct_tolerant  SENSITIVITY — additionally credits a boxed option-VALUE that maps to the
                                      gold choice, else falls back to can_infer. This is exactly
                                      the Track-T §11.8 remedy, and it exists because boxing does
                                      NOT remove the arm-dependent format effect (there,
                                      non-letter-box rate was base 0.045 vs privileged 0.159).
      correct_official  COMPARABILITY — MMStar's own scorer, so we can state whether the finding
                                      survives under the benchmark's published definition.
    """
    body = post_think(raw_text) if is_thinking else raw_text
    boxed_ok, kind, payload = score_boxed(body, choices, gt)
    tol = boxed_ok
    if not tol:
        if kind == "value":
            tol = int(payload == gt)
        else:
            p = can_infer(body, choices) if choices else False
            tol = int(bool(p) and p != "Z" and p == gt)
    return dict(correct=boxed_ok,
                box_kind=kind, box_pred=payload,
                correct_tolerant=int(tol),
                correct_official=score_mmstar_official(body, gt),
                fc_artifact=official_firstchar_artifact(body, gt))

# ---------------------------------------------------------------- provenance / io
def sha_str(s):
    return hashlib.sha256(s.encode()).hexdigest()

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "UNKNOWN"

def _versions():
    """Library versions exist only in the run environment. Set-2/3's record had to log
    'exact vLLM/transformers version strings' as [NOT IN RECORD]; do not repeat that."""
    out = {}
    for name in ("vllm", "transformers", "torch", "pandas", "numpy"):
        try:
            out[name] = __import__(name).__version__
        except Exception as e:
            out[name] = f"UNAVAILABLE({type(e).__name__})"
    return out

def _model_fingerprints():
    """Cheap, stable identity for each checkpoint: hash the small config files, not the 9 GB of
    weights. Enough to prove two runs used the same checkpoint."""
    fp = {}
    for k, p in MODELS.items():
        d = {}
        for f in ("config.json", "generation_config.json", "preprocessor_config.json",
                  "model.safetensors.index.json"):
            fpath = os.path.join(p, f)
            if os.path.exists(fpath):
                d[f] = sha_file(fpath)[:16]
        fp[k] = dict(path=p, sha=d)
    return fp

def provenance(**extra):
    d = dict(git=git_sha(), user=USER, time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             seed=SEED, models=_model_fingerprints(), versions=_versions(),
             decode=DECODE, max_model_len=MAX_MODEL_LEN, img_tok_max=IMG_TOK_MAX,
             wrapper=WRAPPER, wrapper_sha=sha_str(WRAPPER),
             caption_prompt=CAPTION_PROMPT, caption_prompt_sha=sha_str(CAPTION_PROMPT),
             answer_hint=ANSWER_HINT, answer_hint_sha=sha_str(ANSWER_HINT))
    d.update(extra)
    return d

def read_jsonl(path):
    """
    Robust reader for append-only files that may have been torn by a crash.
    Skips unparseable lines (BUG 3: a torn line used to crash Pass 3) and DEDUPES nothing here —
    callers that need dedup use dedup_rows(). Returns (rows, n_torn).
    """
    rows, torn = [], 0
    if not os.path.exists(path):
        return rows, torn
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                torn += 1
    return rows, torn

def dedup_rows(rows, keyfn):
    """
    BUG 2: a torn line mid-file means its key is regenerated and appended, so the same key can
    appear twice. Keep the LAST occurrence (the completed one) and report how many were dropped.
    """
    seen = {}
    for r in rows:
        seen[keyfn(r)] = r
    return list(seen.values()), len(rows) - len(seen)

def load_done(path, keyfn):
    """Resume: set of already-completed keys in an append-only jsonl."""
    rows, _ = read_jsonl(path)
    return {keyfn(r) for r in rows}

class Appender:
    """Append-only writer that flushes+fsyncs per batch so a crash never corrupts prior rows."""
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a")
    def write(self, rows):
        for r in rows:
            self.f.write(json.dumps(r) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())
    def close(self):
        self.f.close()

def build_llm(model_key, max_images=1):
    """One vLLM engine. Captioner and reasoners never co-reside (PAPO sleep/wake OOM lesson)."""
    from vllm import LLM
    return LLM(model=MODELS[model_key], dtype="bfloat16",
               max_model_len=MAX_MODEL_LEN[model_key], gpu_memory_utilization=0.90,
               limit_mm_per_prompt={"image": max_images}, seed=SEED, trust_remote_code=False)

def sampling(model_key, n=1, override=None):
    from vllm import SamplingParams
    d = dict(override or DECODE[model_key])
    return SamplingParams(n=n, seed=SEED, **d)

def chat_prefix(model_key, question_text, processor=None):
    """
    Render the chat template up to the assistant turn. The payload (if any) is string-appended
    by the caller, so it lands as an ASSISTANT PREFILL in both rows:
      - thinking : template auto-opens <think>, so the payload sits inside the reasoning block
      - instruct : no <think>; the payload opens the assistant turn
    Track-T commit 1dfb5cb fixed a double-<think> bug here, which is why gate S3 asserts this.
    """
    from transformers import AutoProcessor
    proc = processor or AutoProcessor.from_pretrained(MODELS[model_key])
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question_text}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def question_text(row):
    # "Question: {q}\n" mirrors the official builder (dataset.py:95); options are already inline
    # in `question` for this release. Format-2 items ALREADY begin with their own "Hint:/Question:"
    # preamble, so do not prepend a second one (found alongside the G2b format discovery).
    q = str(row["question"])
    head = "" if re.match(r"\s*(Hint|Question)\s*:", q) else "Question: "
    return f"{head}{q}\n{ANSWER_HINT}"
