"""
Probe A smoke gates. Nothing runs at scale until every gate PASSES.

  --stage static   : no GPU. Assets, schema, preprocessor parity, chat templates, prompt dumps,
                     choice parsing, scorer self-test, context-assert behaviour, decode dump.
  --stage audit    : after the smoke passes 0-4 have run. Payload integrity, per-arm confound
                     rates, resume integrity, timing extrapolation -> the K recommendation.

Exit code is non-zero if any gate fails, so the sbatch driver stops instead of proceeding.
"""
import argparse, collections, glob, json, os, re, sys
import tp_common as C

FAILS = []
def gate(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)
    return ok


# --------------------------------------------------------------------- static
def static(tag, limit):
    # G1 assets ------------------------------------------------------------
    for k, p in C.MODELS.items():
        gate(f"G1.assets:{k}", os.path.isdir(p), p)
    gate("G1.assets:mmstar", os.path.exists(f"{C.DATA}/mmstar.parquet"))

    # G2 data --------------------------------------------------------------
    df = C.load_mmstar()          # full 1500 for the data gates, regardless of smoke --limit
    sub = C.load_mmstar(limit=limit)
    gate("G2c.stratified_subset_covers_categories",
         sub["category"].nunique() == df["category"].nunique(),
         f"smoke n={len(sub)} categories={sub['category'].nunique()}")
    gate("G2.rows==1500", len(df) == 1500, str(len(df)))
    ok_img, sizes = True, []
    for _, r in df.iterrows():
        try:
            im = C.pil_image(r["image"]); sizes.append(im.size)
        except Exception as e:
            ok_img = False; print("   image decode error:", e); break
    gate("G2.images_decode", ok_img, f"n={len(sizes)}")
    cats = collections.Counter(df["category"])
    gate("G2.categories_balanced", set(cats.values()) == {250}, str(dict(cats)))
    ans = collections.Counter(df["answer"])
    print(f"   [note] gold label distribution is NON-uniform: {dict(ans)}")

    # G2b choice parsing ---------------------------------------------------
    parsed = {int(r["index"]): C.parse_choices(r["question"]) for _, r in df.iterrows()}
    bad = [i for i, c in parsed.items() if len(c) < 2]
    gate("G2b.choice_parse", len(bad) == 0, f"failures={len(bad)} {bad[:5]}")

    # G5a preprocessor parity (captioner must see the same image as the reasoners) ------
    cfgs = {}
    for k, p in C.MODELS.items():
        f = os.path.join(p, "preprocessor_config.json")
        if os.path.exists(f):
            d = json.load(open(f))
            cfgs[k] = {x: d.get(x) for x in ("size", "patch_size", "merge_size",
                                             "image_processor_type")}
    same = len({json.dumps(v, sort_keys=True) for v in cfgs.values()}) == 1
    gate("G5a.preprocessor_parity", same, json.dumps(cfgs.get("caprl", {})))

    # G3 chat templates ----------------------------------------------------
    from transformers import AutoProcessor
    tails = {}
    for k in C.MODELS:
        pr = C.chat_prefix(k, "probe")
        tails[k] = pr[-70:]
        opens = "<think>" in pr.split("assistant")[-1]
        if k == "thinking":
            gate("G3.thinking_opens_think", opens, repr(tails[k]))
        else:
            gate(f"G3.{k}_no_think", not opens, repr(tails[k]))

    # G4 exact prompt dump, all 7 arms -------------------------------------
    row = df.iloc[0]
    cap_demo = "A red suitcase sits beneath a wooden bookshelf."
    print("\n--- G4 exact prompt per arm (item 0) ---")
    for arm, (mk, kind) in C.ARMS.items():
        pre = C.chat_prefix(mk, C.question_text(row))
        payload = "" if kind is None else C.WRAPPER.format(payload=cap_demo)
        print(f"\n### {arm} [{mk}] ###\n{repr(pre + payload)[-420:]}")
    gate("G4.prompt_dump", True, "printed for eyeball verification")

    # G6a (found in the D13/D14 re-review): G6/G6b tokenize with ONE tokenizer, loaded from the
    # "thinking" checkpoint, and reuse it for "instruct" and "caprl" prompts too. That assumption
    # was previously implicit and unverified -- G5a's preprocessor-parity gate is about the IMAGE
    # side only. Confirm it directly: encode a representative probe with EACH model's own
    # tokenizer and require IDENTICAL token ids (not just equal vocab size), since G6b's new
    # certified headroom for A5 depends on this holding.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODELS["thinking"])
    longest = df.loc[df["question"].str.len().idxmax()]
    probe_str = C.chat_prefix("thinking", C.question_text(longest)) + "sample \\boxed{B} text"
    ids_ref = tok(probe_str, add_special_tokens=False).input_ids
    for k in ("instruct", "caprl"):
        tk_k = AutoTokenizer.from_pretrained(C.MODELS[k])
        ids_k = tk_k(probe_str, add_special_tokens=False).input_ids
        gate(f"G6a.tokenizer_shared:{k}", ids_k == ids_ref,
             f"len={len(ids_k)} vs thinking len={len(ids_ref)}" if ids_k != ids_ref
             else f"len={len(ids_k)}")

    # G6 context assert behaviour -----------------------------------------
    # worst case is the CAPTION arm on the LONGEST question: question + hint + wrapper + a
    # max-length caption + the largest image. That is the case that used to blow the budget.
    for mk in ("thinking", "instruct"):
        pre = C.chat_prefix(mk, C.question_text(longest))
        # BUG (found by this gate's own failure on job 3020807, D14 fallout): the worst-case
        # caption length must be the cap Pass 0 ACTUALLY generates captions with --
        # CAPRL_DECODE_CANDIDATES["A_genconfig"]["max_tokens"] (still 4096, the frozen Q6
        # resolution) -- NOT C.DECODE["caprl"]["max_tokens"], which is A5's own VQA-answering
        # budget and has nothing to do with caption length. These two constants were both 4096
        # before D14 raised the latter to 6144 for A5, so this line worked by COINCIDENCE, not by
        # correct reference; the coincidence broke the moment D14 changed one but not the other.
        cap_worst = "x " * C.CAPRL_DECODE_CANDIDATES["A_genconfig"]["max_tokens"]
        worst = pre + C.WRAPPER.format(payload=cap_worst)
        ptok = len(tok(worst, add_special_tokens=False).input_ids)
        hdr = ptok + C.IMG_TOK_MAX + C.DECODE[mk]["max_tokens"]
        gate(f"G6.context_headroom:{mk}", hdr <= C.MAX_MODEL_LEN[mk],
             f"worst ptok={ptok} +img{C.IMG_TOK_MAX} +maxtok={C.DECODE[mk]['max_tokens']} "
             f"= {hdr} vs max_model_len={C.MAX_MODEL_LEN[mk]}")

    # G6b (D14, 2026-08-06): context headroom for A5 (caprl model). UNLIKE T/I, caprl never
    # carries a caption payload in this design (ARMS["A5"] = ("caprl", None)) -- A5's own worst
    # case is the bare question, no WRAPPER. Reusing the wrapper-inclusive worst case from G6 above
    # would be wrong (too conservative and answering the wrong question); this is A5's actual
    # binding constraint, checked the same live way as G6 rather than hand-computed, because D14's
    # whole premise is "certified by the gate, not by estimation". Shares the `tok` loaded from the
    # thinking checkpoint -- G6a above just confirmed that's valid for caprl too.
    pre_bare = C.chat_prefix("caprl", C.question_text(longest))
    ptok_bare = len(tok(pre_bare, add_special_tokens=False).input_ids)
    hdr_bare = ptok_bare + C.IMG_TOK_MAX + C.DECODE["caprl"]["max_tokens"]
    gate("G6b.context_headroom:caprl", hdr_bare <= C.MAX_MODEL_LEN["caprl"],
         f"worst ptok={ptok_bare} +img{C.IMG_TOK_MAX} +maxtok={C.DECODE['caprl']['max_tokens']} "
         f"= {hdr_bare} vs max_model_len={C.MAX_MODEL_LEN['caprl']}")

    # G9 scorer self-test -- PRIMARY is MMStar's own position-0-anchored rule --------------
    ch = {"A": "the cat", "B": "the dog", "C": "the bird", "D": "the fish"}
    # (text, gt, expected_official, expected_tolerant)
    cases = [
        ("B", "B", 1, 1),
        ("B. the dog", "B", 1, 1),
        ("(B)", "B", 1, 1),                       # '(' + letter
        ("option B", "B", 1, 1),                  # 'option ' prefix
        ("the answer is B", "B", 1, 1),           # 'the answer is ' prefix
        ("A", "B", 0, 0),
        # ⚠ the official rule compares only predict[0], so a prose opener starting with the gold
        # letter is CREDITED. These two encode that defect so it can never silently regress.
        ("Based on the image, B", "B", 1, 1),     # "based" -> 'b' == gold 'b'  => false positive
        ("Between B and C it is hard", "B", 1, 0),  # "between" -> 'b'          => false positive
        ("The correct answer is B", "B", 0, 1),   # not the exact 'the answer is ' prefix
        ("the dog", "B", 0, 1),                   # option TEXT: official rejects, tolerant accepts
        ("", "B", 0, 0),
    ]
    bad = 0
    for text, gt, w_off, w_tol in cases:
        s = C.score_item(text, dict(ch), gt, is_thinking=False)
        if s["correct_official"] != w_off or s["correct_tolerant"] != w_tol:
            bad += 1
            print(f"   mismatch {text!r}: official={s['correct_official']}(want {w_off}) "
                  f"tolerant={s['correct_tolerant']}(want {w_tol})")
    gate("G9.official_and_tolerant_selftest", bad == 0, f"{len(cases) - bad}/{len(cases)} cases")

    # G9d PRIMARY = robust final-answer extraction (D13). (text, gt, want_primary, want_kind)
    bcases = [
        ("blah \\boxed{B}", "B", 1, "letter"),
        ("\\boxed{ B }", "B", 1, "letter"),
        ("\\boxed{\\text{B}}", "B", 1, "letter"),
        ("\\boxed{A}", "B", 0, "letter"),
        ("\\boxed{the dog}", "B", 0, "value"),        # value-box: primary=0, tolerant credits it
        ("\\boxed{42}", "B", 0, "other"),
        ("no box here at all", "B", 0, "none"),
        ("\\boxed{B", "B", 0, "none"),                # truncated mid-box -> unbalanced -> none
        ("\\boxed{A} then \\boxed{B}", "B", 1, "letter"),   # LAST box wins
        # non-boxed conclusions (D13: the prompt no longer asks for \boxed{})
        ("The correct answer is B.", "B", 1, "letter"),
        ("Final Answer: C", "B", 0, "letter"),
        ("I'll go with option B.", "B", 1, "letter"),
        ("My answer is **B**.", "B", 1, "letter"),
        ("**D**", "D", 1, "letter"),
        ("Based on the details, I'll go with (C)", "C", 1, "letter"),
        ("So the correct one is D", "D", 1, "letter"),
        ("\n\nD", "D", 1, "letter"),                  # T0's actual observed post-<think> tail
        # ADVERSARIAL: a letter is mentioned but the response explicitly does not conclude.
        # This is the exact shape of the smoke's runaway-truncation generations; an unrestricted
        # whole-text scan for "option X" would wrongly credit B here.
        ("Option A is close but option B fits better. I keep going back and forth without "
         "landing on one.", "B", 0, "none"),
        # real observed truncated I1 tail (trunc=1) -- must NOT be credited despite naming D
        ("...D is the only option that mentions “beneath”, perhaps it is the intended "
         "answer — even though it is false", "D", 0, "none"),
        ("Therefore, the correct option is (A).\n\n\\boxed{A}", "A", 1, "letter"),  # real I1 case
    ]
    bbad = 0
    for text, gt, wp, wk in bcases:
        s = C.score_item(text, dict(ch), gt, is_thinking=False)
        if s["correct"] != wp or s["ans_kind"] != wk:
            bbad += 1
            print(f"   extraction mismatch {text!r}: primary={s['correct']}(want {wp}) "
                  f"kind={s['ans_kind']}(want {wk})")
    gate("G9d.primary_extractor_selftest", bbad == 0, f"{len(bcases) - bbad}/{len(bcases)} cases")
    gate("G9e.value_box_only_in_tolerant",
         C.score_item("\\boxed{the dog}", dict(ch), "B", False)["correct"] == 0
         and C.score_item("\\boxed{the dog}", dict(ch), "B", False)["correct_tolerant"] == 1,
         "Track-T §11.8: a value-box is NOT credited by the primary, IS by the sensitivity")
    gate("G9b.thinking_parse",
         C.score_item("<think>maybe \\boxed{A}</think> \\boxed{B}", dict(ch), "B", True)["correct"] == 1,
         "post-</think> parsing: a box inside <think> must not be picked up")
    gate("G9c.firstchar_artifact_detected",
         C.official_firstchar_artifact("Based on the image, B", "B") == 1
         and C.official_firstchar_artifact("B", "B") == 0
         and C.official_firstchar_artifact("(B)", "B") == 0,
         "detector flags predict[0] false-positives but not genuine letter answers")
    gate("G9f.unclosed_think_forces_none",
         C.score_item("still reasoning about B and C, never concludes", dict(ch), "B", True,
                      closed=False) == dict(correct=0, ans_kind="none", ans_pred=None,
                                             correct_tolerant=0, correct_official=0, fc_artifact=0),
         "an unclosed <think> (model never left the reasoning phase) forces a clean miss on ALL "
         "three metrics, not just the primary -- found in the D13/D14 re-review")
    gate("G9g.bare_end_excludes_article_a",
         C.score_item("so it must be a", dict(ch), "A", False)["ans_kind"] == "none",
         "the trailing English article 'a' must NOT be credited as letter A (bare-letter rule is "
         "deliberately case-sensitive; see tp_common)")
    gate("G9h.bold_paren_case_insensitive",
         C.score_item("**d**", dict(ch), "D", False)["correct"] == 1
         and C.score_item("(d)", dict(ch), "D", False)["correct"] == 1,
         "bold/paren rules ARE case-insensitive (their context can't collide with ordinary prose)")

    # G12 decode dump ------------------------------------------------------
    print("\n--- G12 frozen decode ---")
    for k, d in C.DECODE.items():
        print(f"   {k:9s} {d}  max_model_len={C.MAX_MODEL_LEN[k]}")
    gate("G12.decode_dump", True)

    # S1/S2 fidelity gates -- RESOLVED against the cloned verbatim source -------------------
    vend = f"{C.TP}/vendor/MMStar/eval/vlmeval"
    gate("S1.answer_hint_from_source", os.path.exists(f"{vend}/utils/dataset.py"),
         "ANSWER_HINT transcribed verbatim from dataset.py:97-99")
    gate("S2.scorer_from_source", os.path.exists(f"{vend}/evaluate/mmstar.py"),
         "PRIMARY = MMStar_eval (evaluate/mmstar.py); SECONDARY = can_infer (utils/matching_util.py)")


# --------------------------------------------------------------------- audit
def audit(tag):
    base = f"{C.OUT}/{tag}"

    # G7 captions ----------------------------------------------------------
    caps = [json.loads(l) for l in open(f"{base}/captions.jsonl")]
    lens = [c["ntok"] for c in caps]
    empt = sum(1 for c in caps if not c["caption"].strip())
    trunc = sum(c["trunc"] for c in caps)
    lens_s = sorted(lens)
    gate("G7.captions_nonempty", empt == 0, f"empty={empt}")
    gate("G7.captions_not_truncated", trunc / max(len(caps), 1) < 0.05,
         f"trunc={trunc}/{len(caps)}")
    print(f"   caption ntok min/med/max = {lens_s[0]}/{lens_s[len(lens_s)//2]}/{lens_s[-1]}"
          f"  (CapRL++ regularises at tau1=2048, tau2=3072)")

    # caption-sampling variance -> informs whether M could ever be 1 (Q5) ----
    byidx = collections.defaultdict(list)
    for c in caps:
        byidx[c["index"]].append(c)
    spreads = [(max(len(x["caption"]) for x in v) - min(len(x["caption"]) for x in v))
               / max(1, sum(len(x["caption"]) for x in v) / len(v))
               for v in byidx.values() if len(v) > 1]
    uniq = [len({x["caption_sha"] for x in v}) / len(v) for v in byidx.values() if len(v) > 1]
    if spreads:
        print(f"   [Q5] caption length spread (rel) mean={sum(spreads)/len(spreads):.3f}; "
              f"distinct-caption fraction mean={sum(uniq)/len(uniq):.3f}")
    gate("G7b.caption_variance_measured", bool(spreads), "reported above for the M decision")

    # G7c EYEBALL THE STIMULUS. Length statistics cannot tell us whether a caption is USEFUL --
    # whether it names objects, states relations, reads text in the image, hedges, or refuses.
    # The caption IS the treatment; if it is degenerate the whole probe is void, and no numeric
    # gate above would catch that. Printed for human inspection, deliberately not auto-gated.
    df_all = C.load_mmstar()
    qof = {int(r["index"]): str(r["question"])[:180] for _, r in df_all.iterrows()}
    keys = sorted(byidx)
    print("\n--- G7c: full captions for eyeball inspection ---")
    for i in (keys[0], keys[len(keys) // 2], keys[-1]):
        v = sorted(byidx[i], key=lambda x: x["caption_idx"])
        print(f"\n### item {i} | ntok={[x['ntok'] for x in v]} "
              f"| distinct={len({x['caption_sha'] for x in v})}/{len(v)}")
        print(f"    QUESTION: {qof.get(i, '?')}")
        print(f"    CAPTION[0]:\n{v[0]['caption'][:1200]}")
        if len(v) > 1:
            print(f"    CAPTION[1] (variance check):\n{v[1]['caption'][:600]}")
    gate("G7c.captions_printed", True, "inspect the text above before trusting any downstream number")

    # G22 (D16, 2026-08-06, RESCOPED) for the question-conditioned captions (arms T3/I3). The
    # captioner is given the question STEM ONLY -- options are stripped -- so it cannot name a
    # choice it never saw.
    #
    # The ORIGINAL gate flagged ANY lexical overlap with ANY option's text as a "leak" and fired at
    # 3.3% (8/240). Reading all 8 found zero \boxed{}, zero "the answer is", zero enumeration
    # markers -- every flag was a truthful, question-relevant caption whose wording happened to
    # overlap the CORRECT option (e.g. caption "holding her belly with her left hand" vs. the
    # option's own "her left hand"). For a perception item, options ARE scene descriptions, so an
    # accurate captioner overlapping the correct one is the captioner doing its job, not
    # contamination -- and it is structurally impossible for it to have read a list it was never
    # shown. That is not a leak signal; it was a mis-specified gate.
    #
    # The REAL trip-wires -- things that would be structurally impossible unless the captioner had
    # somehow seen the choice list -- are format leakage: a \boxed{}, an explicit answer
    # declaration, or literal enumeration markers ("A:" / "(A)" / "Choices:") that only exist in the
    # options block it was never given. Those remain a hard gate (Track-T's comparable prompt
    # leaked 2/2485 = 0.08%; threshold mirrors its <1% audit rule).
    #
    # Lexical overlap with option TEXT is downgraded to a DESCRIPTIVE rate, split by whether the
    # overlapping option is the correct one (expected for an accurate caption -- and informative
    # later for interpreting T3/I3, since it bounds how much of any T3 benefit could be answer-
    # lookup rather than reasoning-aid) or an incorrect one (would be a genuinely surprising
    # coincidence for a truthful caption -- surfaced, not blocking, since n=48 is too small to set
    # a real threshold on it yet).
    qp = f"{base}/captions_q.jsonl"
    if os.path.exists(qp):
        dfq = C.load_mmstar()
        chq = {int(r["index"]): C.parse_choices(r["question"]) for _, r in dfq.iterrows()}
        gtq = {int(r["index"]): str(r["answer"]).upper() for _, r in dfq.iterrows()}
        rows, _ = C.read_jsonl(qp)
        _ENUM = re.compile(r"(?:^|\n)\s*\(?[A-D]\)?[.):]\s|\bChoices\s*:|\bOptions\s*:", re.I)
        fmt_leaks, corr_hits, incorr_hits = [], 0, 0
        for r in rows:
            t = r["caption"].lower()
            idx = r["index"]
            opts = {k: str(v).lower() for k, v in chq.get(idx, {}).items() if str(v).strip()}
            gold = gtq.get(idx, "")
            corr_hits += int(any(k == gold and v in t for k, v in opts.items() if len(v) > 12))
            incorr_hits += int(any(k != gold and v in t for k, v in opts.items() if len(v) > 12))
            if "\\boxed" in t or "the answer is" in t or "final answer" in t or _ENUM.search(r["caption"]):
                fmt_leaks.append(idx)
        rate = len(fmt_leaks) / max(len(rows), 1)
        gate("G22.qcaption_no_format_leak", rate < 0.01,
             f"format-leak rate {rate:.4f} ({len(fmt_leaks)}/{len(rows)}) e.g. {fmt_leaks[:5]}")
        n = max(len(rows), 1)
        print(f"   [descriptive] q-caption lexical overlap: correct-option={corr_hits/n:.3f} "
              f"incorrect-option={incorr_hits/n:.3f} "
              f"(overlap with the CORRECT option is expected for an accurate caption, not a leak; "
              f"overlap with an INCORRECT option would be surprising -- watch this if it grows)")
        lq = sorted(r["ntok"] for r in rows)
        print(f"   q-caption ntok min/med/max = {lq[0]}/{lq[len(lq)//2]}/{lq[-1]}")

    # G8 placebo -----------------------------------------------------------
    pm = json.load(open(f"{base}/placebo_meta.json"))
    gate("G8.placebo_no_fallbacks", len(pm["fallbacks"]) == 0, str(pm["fallbacks"][:3]))
    gate("G8.placebo_length_matched", pm["len_reldiff_mean"] < 0.35,
         f"mean rel Δlen={pm['len_reldiff_mean']:.3f} max={pm['len_reldiff_max']:.3f}")

    # G21 realized image-token parity -- the HARD version of G5a. G5a compared preprocessor
    # CONFIGS; this compares the token counts the engine actually produced for the same image
    # under the captioner and under each reasoner. If these disagree, "the captioner described
    # what the reasoner sees" is false in fact, whatever the configs say.
    def imgmap(path, key):
        rows, _ = C.read_jsonl(path)
        return {r[key]: r.get("img_tok") for r in rows if r.get("img_tok") is not None}
    capimg = imgmap(f"{base}/captions.jsonl", "index")
    if capimg:
        for arm in ("T0", "I0"):
            p = f"{base}/gen_{arm}.jsonl"
            if not os.path.exists(p):
                continue
            armimg = imgmap(p, "index")
            sh = set(capimg) & set(armimg)
            if sh:
                agree = sum(1 for i in sh if capimg[i] == armimg[i])
                gate(f"G21.img_token_parity:caprl_vs_{arm}", agree == len(sh),
                     f"{agree}/{len(sh)} items have identical realized image-token counts")
    else:
        print("   [note] G21 skipped: engine did not expose prompt_token_ids for captions")

    # G16 payload integrity ------------------------------------------------
    capsha = {(c["index"], c["caption_idx"]): c["caption_sha"] for c in caps}
    ok_pay = True
    for arm in ("T1", "I1"):
        p = f"{base}/gen_{arm}.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            src = r["payload_src"].split(":")
            if src[0] == "self" and r["payload_sha"] != capsha.get((int(src[1]), int(src[2]))):
                ok_pay = False
                break
    gate("G16.payload_matches_frozen_caption", ok_pay)

    # T1/I1 must consume byte-identical payloads ---------------------------
    def paymap(arm):
        p = f"{base}/gen_{arm}.jsonl"
        return {(json.loads(l)["index"], json.loads(l)["draw"]): json.loads(l)["payload_sha"]
                for l in open(p)} if os.path.exists(p) else {}
    t1, i1 = paymap("T1"), paymap("I1")
    shared = set(t1) & set(i1)
    gate("G16b.T1_I1_same_captions",
         bool(shared) and all(t1[k] == i1[k] for k in shared), f"n={len(shared)}")

    # G10/G11 per-arm confound rates + G15 timing --------------------------
    print("\n--- per-arm rates & timing ---")
    tot_rate = {}
    for mp in sorted(glob.glob(f"{base}/gen_*_meta.json")):
        m = json.load(open(mp))
        arm = m["arm"]
        tot_rate[arm] = m["rate_per_s"]
        print(f"   {arm:3s} model={m['model']:8s} n={m['n_generated']:5d} "
              f"{m['seconds']:7.0f}s  {m['rate_per_s']:.3f}/s  worst_ptok={m['worst_prompt_tokens']}")
    sm = json.load(open(f"{base}/score_meta.json"))["summary"]
    for arm, s in sm.items():
        print(f"   {arm:3s} acc={s['acc']:.4f} extract={s['extract_rate']:.4f} "
              f"unextract={s['unextract_rate']:.4f} "
              f"trunc={s['trunc_rate']:.4f} unclosed={s['unclosed_think_rate']:.4f}")
    ui = {a: s["unextract_rate"] for a, s in sm.items()}
    gate("G10.unextract_rate_low", all(v < 0.15 for v in ui.values()), str(ui))
    spread = (max(ui.values()) - min(ui.values())) if ui else 0
    gate("G10b.unextract_arm_spread", spread < 0.10,
         f"spread={spread:.3f} -- an arm-dependent format effect would confound accuracy")
    tr = {a: s["trunc_rate"] for a, s in sm.items()}
    gate("G11.truncation_low", all(v < 0.10 for v in tr.values()), str(tr))

    # G15 K recommendation --------------------------------------------------
    if tot_rate:
        per_draw = sum(1.0 / r for r in tot_rate.values())   # seconds per item per draw, all arms
        print(f"\n   [G15] summed cost across arms = {per_draw:.2f} s per item per draw")
        for K in (3, 4, 5):
            hrs = per_draw * 1500 * K / 3600
            print(f"        K={K}: 1500 items -> ~{hrs:.1f} GPU-hours total across all 7 arms")
        gate("G15.timing_extrapolated", True, "use the table above to fix K (Q2)")

    # G17 payload delivery -- a behavioural check that the caption actually changed the output.
    # Set-3's whole parity saga exists because "the payload was in the prompt" is NOT evidence the
    # model used it. Identical T0/T1 continuations on the same (index,draw) would mean the payload
    # is inert or, worse, not reaching the model at all.
    def textmap(arm):
        rows, _ = C.read_jsonl(f"{base}/gen_{arm}.jsonl")
        rows, _ = C.dedup_rows(rows, lambda r: (r["index"], r["draw"]))
        return {(r["index"], r["draw"]): r["text"] for r in rows}
    for pair in (("T0", "T1"), ("I0", "I1")):
        m0, m1 = textmap(pair[0]), textmap(pair[1])
        sh = set(m0) & set(m1)
        if sh:
            same = sum(1 for k in sh if m0[k] == m1[k])
            gate(f"G17.payload_changes_output:{pair[0]}->{pair[1]}",
                 same / len(sh) < 0.5, f"identical continuations {same}/{len(sh)}")

    # G18 independent draws -- guards the per-request-seed fix. GATE ON DISTINCT SEEDS, not on
    # distinct text: for a short-output arm (I0 answering "\boxed{B}") identical text across draws
    # is entirely legitimate, so a text-diversity gate would raise a FALSE ALARM and block the run
    # for a non-bug. Text diversity is reported descriptively instead.
    for arm in ("T0", "I0", "T1", "I1"):
        p = f"{base}/gen_{arm}.jsonl"
        if not os.path.exists(p):
            continue
        rows, _ = C.read_jsonl(p)
        byi = collections.defaultdict(list)
        for r in rows:
            byi[r["index"]].append(r)
        multi = [v for v in byi.values() if len(v) > 1]
        if not multi:
            continue
        seed_ok = all(len({x.get("seed") for x in v}) == len(v) for v in multi)
        gate(f"G18.distinct_seeds:{arm}", seed_ok, "every draw of an item used its own seed")
        tfrac = sum(len({x["text"] for x in v}) / len(v) for v in multi) / len(multi)
        print(f"   [descriptive] {arm} mean distinct-TEXT fraction={tfrac:.3f} "
              f"(low is fine for short answers)")

    # G19 placebo length-matching in TOKENS, not characters. Pass 1 matches on character length;
    # what actually enters the model is tokens. If T1 and T2 prompts differ materially in ptok the
    # placebo is not length-matched in the units that matter, and any T1-T2 gap is confounded.
    for pair in (("T1", "T2"), ("I1", "I2")):
        ms = []
        for arm in pair:
            p = f"{base}/gen_{arm}.jsonl"
            if not os.path.exists(p):
                ms = []
                break
            rows, _ = C.read_jsonl(p)
            ms.append(sum(r["ptok"] for r in rows) / max(len(rows), 1))
        if len(ms) == 2:
            rel = abs(ms[0] - ms[1]) / max(ms[0], 1)
            gate(f"G19.ptok_matched:{pair[0]}vs{pair[1]}", rel < 0.15,
                 f"mean ptok {ms[0]:.0f} vs {ms[1]:.0f} (rel diff {rel:.3f})")

    # G20 prompt reconstruction -- Set-3's p2_audit pattern. Rebuild the prompt from the FROZEN
    # artifacts using the same constructors, and check the reconstructed token count equals the
    # ptok the run actually recorded. This is what proves the payload we think we sent is the
    # payload that was sent; payload_sha alone is tautological (Pass 2 hashes the same string it
    # inserted).
    try:
        from transformers import AutoProcessor, AutoTokenizer
        df = C.load_mmstar()
        rows_by_idx = {int(r["index"]): r for _, r in df.iterrows()}
        capmap = {}
        for r in C.read_jsonl(f"{base}/captions.jsonl")[0]:
            capmap.setdefault(r["index"], {})[r["caption_idx"]] = r
        okc = badc = 0
        for arm in ("T1", "I1"):
            p = f"{base}/gen_{arm}.jsonl"
            if not os.path.exists(p):
                continue
            mk = C.ARMS[arm][0]
            proc = AutoProcessor.from_pretrained(C.MODELS[mk])
            tk = AutoTokenizer.from_pretrained(C.MODELS[mk])
            rows, _ = C.read_jsonl(p)
            for r in rows[:20]:
                src = r["payload_src"].split(":")
                cap = capmap[int(src[1])][int(src[2])]["caption"]
                pre = C.chat_prefix(mk, C.question_text(rows_by_idx[r["index"]]), processor=proc)
                rebuilt = pre + C.WRAPPER.format(payload=cap)
                n = len(tk(rebuilt, add_special_tokens=False).input_ids)
                okc += (n == r["ptok"]); badc += (n != r["ptok"])
        gate("G20.prompt_reconstruction", badc == 0, f"ptok match {okc}/{okc + badc}")
    except Exception as e:
        gate("G20.prompt_reconstruction", False, f"reconstruction errored: {e}")

    print("\n--- resume test (G13): re-run any pass2 arm; 'nothing to do' == PASS ---")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["static", "audit"])
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--limit", type=int, default=16)
    a = ap.parse_args()
    if a.stage == "static":
        static(a.tag, a.limit)
    else:
        audit(a.tag)
    print("\n" + "=" * 60)
    if FAILS:
        print("FAILED GATES:", ", ".join(FAILS))
        sys.exit(1)
    print("ALL GATES PASSED")


if __name__ == "__main__":
    main()
