"""T0b -- does the objective RANK captions correctly, and is the forward correct?

T0a found D(vague) < D(mismatched) < D(matched) on 3 of 4 items: the contentless caption
scored best. Two questions follow, and they must be answered in this order, because the
second is meaningless if the first fails.

PHASE 0 -- IS THE FORWARD CORRECT?  T0a's C1 (packed vs padded log-probs) reached 0.219,
far above bf16's ~0.008 ulp at a sampled token's typical log-prob -- but one item was
EXACTLY 0.0000, which a systematic indexing bug would not produce. Decisive test: recompute
the log-softmax in fp32. If the discrepancy collapses, it was bf16 accumulation between the
varlen and masked flash kernels. If it survives, `forward_packed_logits` is wrong and every
distortion this project has computed is void. Reported as a DISTRIBUTION, not a max --
T0a reporting only the max is why this was ambiguous.

PHASE 1 -- DOES D RANK CAPTIONS CORRECTLY?  The point that makes this testable without
good captions: the captions are SUPPOSED to be bad at step 0, and training exists to fix
them. So do not ask "is this caption good". Ask which way `D` moves as a caption is
degraded. That is quality-independent and it is the actual gradient direction.

    THE ABLATION LADDER, per item, all scored against ONE shared sighted pass:
        no_evidence   no caption at all      -- q_0
        matched_100   the model's caption
        matched_60    first 60% of its words
        matched_30    first 30% of its words
        vague         contentful-sounding, empty
        mismatched    another image's caption

    If the instrument is sound:  D(matched_100) < D(matched_60) < D(matched_30) < D(vague)
    and every one of them below D(no_evidence) -- a caption that helps must beat having no
    caption. If D instead FALLS as content is removed, the objective rewards uninformative
    captions, training will move captions that way, and nothing opposes it: J_success scores
    the SIGHTED answer and exerts no force whatever on caption content.

WHY no_evidence IS THE KEY ARM. D(no_evidence) = KL(p || q_0) is exactly the per-item
vision-necessity, measured in the same units as everything else: how much the image changed
the model's trajectory distribution. It is both the ceiling a caption must beat AND the
variable that decides whether an item could ever have discriminated. THE DECISIVE STATISTIC
IS THE RELATIONSHIP between D(no_evidence) and (D(vague) - D(matched_100)): if matched wins
precisely on the items where the image mattered, the estimand is sound and the substrate is
the problem. If matched loses even there, the estimand is wrong.

A NOTE ON WHAT THIS CANNOT SHOW. One sampled caption per rung is not a group. `g_c` captions
per item are also drawn, so the within-group spread of D and the variance decomposition
(8.1) get a first measurement -- but at this n they are indicative, not settled.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

VAGUE_CAPTION = (
    "The image shows a scene with several elements arranged in it. There are various "
    "objects and features visible, and the overall composition includes different parts "
    "that relate to one another in the usual way."
)


def _truncate_words(text: str, frac: float) -> str:
    w = str(text).split()
    if not w:
        return text
    return " ".join(w[:max(1, int(len(w) * frac))])


def _load(model_path: str):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        device_map="cuda:0")
    model.eval()
    return model, processor, processor.tokenizer


def _rows(parquet: str, n: int):
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    t = pq.read_table(parquet, columns=["problem", "answer", "images", "problem_id"])
    d = t.to_pydict()
    out = []
    for i in range(min(n, t.num_rows)):
        cell = d["images"][i]
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        out.append({"problem_id": d["problem_id"][i], "problem": str(d["problem"][i]),
                    "answer": str(d["answer"][i]),
                    "image": Image.open(io.BytesIO(raw)).convert("RGB")})
    return out


def _generate(model, processor, tokenizer, messages, images, max_new_tokens, temperature):
    import torch

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    kw = {"images": images} if images else {}
    inputs = processor(text=[text], add_special_tokens=False, return_tensors="pt", **kw)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                             top_p=0.99, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:],
                            skip_special_tokens=True).strip()


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-items", type=int, default=16)
    ap.add_argument("--g-c", type=int, default=3, help="sampled captions per item")
    ap.add_argument("--m", type=int, default=2, help="trajectories per item")
    ap.add_argument("--max-prompt-length", type=int, default=12800)
    ap.add_argument("--max-caption-tokens", type=int, default=256)
    ap.add_argument("--max-answer-tokens", type=int, default=256)
    ap.add_argument("--min-pixels", type=int, default=262144)
    ap.add_argument("--max-pixels", type=int, default=4194304)
    args = ap.parse_args()

    import ca21_prompts as P
    from ca21_contexts import append_responses, build_prompt_row
    from ca21_estimator import distortion_from_logits
    from ca21_leak import leak_flags
    from ca21_logging import variance_decomposition
    from ca21_packing import gather_response_logits, gather_response_logits_reference
    from ca21_worker import forward_packed_logits

    print(f"[t0b] loading {args.model}", flush=True)
    model, processor, tokenizer = _load(args.model)
    from verl.utils.dataset import process_image

    rows = _rows(args.parquet, args.n_items)
    for r in rows:
        r["pimage"] = process_image(r["image"], args.min_pixels, args.max_pixels)
    print(f"[t0b] {len(rows)} items, g_c={args.g_c}, m={args.m}", flush=True)

    t = time.time()
    caps, trajs = [], []
    for r in rows:
        caps.append([_generate(model, processor, tokenizer,
                               P.build_captioner_messages(r["problem"]), [r["pimage"]],
                               args.max_caption_tokens, 1.0) for _ in range(args.g_c)])
        trajs.append([_generate(model, processor, tokenizer,
                                P.build_sighted_messages(r["problem"]), [r["pimage"]],
                                args.max_answer_tokens, 1.0) for _ in range(args.m)])
    t_gen = time.time() - t
    print(f"[t0b] generation {t_gen:.0f}s", flush=True)

    dev = model.device
    results, records, t_score = [], [], time.time()

    for i, r in enumerate(rows):
        c0 = caps[i][0]
        # The ladder plus the group. no_evidence FIRST: it is the ceiling everything else
        # must beat, and on its own it measures this item's vision-necessity.
        variants = {
            "no_evidence": None,
            "matched_100": c0,
            "matched_60": _truncate_words(c0, 0.60),
            "matched_30": _truncate_words(c0, 0.30),
            "vague": VAGUE_CAPTION,
            "mismatched": caps[(i + 1) % len(rows)][0],
        }
        for j, c in enumerate(caps[i]):
            variants[f"sample_{j}"] = c

        s_row = build_prompt_row(processor, tokenizer, P.build_sighted_messages(r["problem"]),
                                 [r["image"]], args.max_prompt_length,
                                 args.min_pixels, args.max_pixels)
        mm = {k: v.to(dev) for k, v in processor.image_processor(
            images=[r["pimage"]], return_tensors="pt").items()}

        blind_rows = {name: build_prompt_row(
            processor, tokenizer,
            P.build_no_evidence_messages(r["problem"]) if cap is None
            else P.build_answerer_messages(cap, r["problem"]),
            None, args.max_prompt_length, args.min_pixels, args.max_pixels)
            for name, cap in variants.items()}

        per_traj = []
        for ti, y in enumerate(trajs[i]):
            y_ids = tokenizer(y, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            T = int(y_ids.shape[0])
            if T < 8:
                continue
            resp = y_ids.unsqueeze(0)
            rmask = torch.ones_like(resp)

            s_ids, s_am, s_pos = append_responses(
                s_row["input_ids"].unsqueeze(0), s_row["attention_mask"].unsqueeze(0),
                s_row["position_ids"].unsqueeze(0), resp, rmask)
            s_ids, s_am, s_pos = s_ids.to(dev), s_am.to(dev), s_pos.to(dev)

            lg_packed, idx, B, S = forward_packed_logits(
                model, s_ids, s_am, s_pos, multi_modal_inputs=mm, padding_free=True)
            lg_s, m_s = gather_response_logits(lg_packed, idx, B, S, T)

            # ---- PHASE 0: bf16 vs fp32, as a distribution
            if ti == 0:
                with torch.no_grad():
                    padded = model(input_ids=s_ids, attention_mask=s_am,
                                   position_ids=s_pos.transpose(0, 1), **mm,
                                   use_cache=False).logits
                lg_ref, _ = gather_response_logits_reference(padded, s_am, T)
                lab = resp.to(dev).clamp_min(0).unsqueeze(-1)
                d16 = (lg_s.log_softmax(-1).gather(-1, lab).squeeze(-1)
                       - lg_ref.log_softmax(-1).gather(-1, lab).squeeze(-1)).abs()
                d32 = (lg_s.float().log_softmax(-1).gather(-1, lab).squeeze(-1)
                       - lg_ref.float().log_softmax(-1).gather(-1, lab).squeeze(-1)).abs()
                c1 = {
                    "bf16_max": float(d16.max()), "bf16_mean": float(d16.mean()),
                    "bf16_frac_over_1e-2": float((d16 > 1e-2).float().mean()),
                    "fp32_max": float(d32.max()), "fp32_mean": float(d32.mean()),
                    "fp32_frac_over_1e-2": float((d32 > 1e-2).float().mean()),
                }
                del padded, lg_ref

            row_d, row_hb = {}, {}
            for name, br in blind_rows.items():
                b_ids, b_am, b_pos = append_responses(
                    br["input_ids"].unsqueeze(0), br["attention_mask"].unsqueeze(0),
                    br["position_ids"].unsqueeze(0), resp, rmask)
                lgb, bidx, bB, bS = forward_packed_logits(
                    model, b_ids.to(dev), b_am.to(dev), b_pos.to(dev),
                    multi_modal_inputs=None, padding_free=True)
                lg_b, m_b = gather_response_logits(lgb, bidx, bB, bS, T)
                res = distortion_from_logits(lg_s.float(), lg_b.float(), m_s * m_b,
                                             labels=resp.to(dev), temperature=1.0)
                row_d[name] = float(res["kl"])
                row_hb[name] = float(res["entropy_q"])
                del lgb, lg_b
            per_traj.append({"traj_idx": ti, "T": T, "D": row_d, "H_blind": row_hb,
                             "H_sighted": float(res["entropy_p"])})
            for j in range(len(caps[i])):
                records.append({"uid": str(r["problem_id"]), "caption_idx": j,
                                "traj_idx": ti, "kl": row_d[f"sample_{j}"]})
            del lg_packed, lg_s

        if not per_traj:
            continue
        avg = {k: st.mean([p["D"][k] for p in per_traj]) for k in variants}
        ne = avg["no_evidence"]
        rec = {
            "problem_id": r["problem_id"], "c1": c1,
            "D": avg,
            "H_blind": {k: st.mean([p["H_blind"][k] for p in per_traj]) for k in variants},
            "H_sighted": st.mean([p["H_sighted"] for p in per_traj]),
            "vision_necessity_D_no_evidence": ne,
            "explanatory_fraction_matched": (1.0 - avg["matched_100"] / ne) if ne > 0 else None,
            "ladder_monotonic": (avg["matched_100"] <= avg["matched_60"]
                                 <= avg["matched_30"] <= avg["vague"]),
            "matched_beats_vague": avg["matched_100"] < avg["vague"],
            "matched_beats_no_evidence": avg["matched_100"] < ne,
            "leak": leak_flags(caps[i][0], r["answer"]),
            "caption_words": len(caps[i][0].split()),
        }
        results.append(rec)
        print(f"[t0b] {r['problem_id']}: ne={ne:.4f} m100={avg['matched_100']:.4f} "
              f"m60={avg['matched_60']:.4f} m30={avg['matched_30']:.4f} "
              f"vague={avg['vague']:.4f} mis={avg['mismatched']:.4f} "
              f"| E={rec['explanatory_fraction_matched']}", flush=True)

    t_score = time.time() - t_score
    n = len(results)
    ne_all = [r["vision_necessity_D_no_evidence"] for r in results]
    gap = [r["D"]["vague"] - r["D"]["matched_100"] for r in results]
    corr = None
    if n >= 3 and st.pstdev(ne_all) > 0 and st.pstdev(gap) > 0:
        mx, my = st.mean(ne_all), st.mean(gap)
        corr = (sum((a - mx) * (b - my) for a, b in zip(ne_all, gap))
                / (n * st.pstdev(ne_all) * st.pstdev(gap)))

    summary = {
        "n_items": n,
        "C1_bf16_max": max(r["c1"]["bf16_max"] for r in results) if n else None,
        "C1_fp32_max": max(r["c1"]["fp32_max"] for r in results) if n else None,
        "C1_fp32_frac_over_1e-2": max(r["c1"]["fp32_frac_over_1e-2"] for r in results) if n else None,
        "ladder_monotonic": sum(r["ladder_monotonic"] for r in results),
        "matched_beats_vague": sum(r["matched_beats_vague"] for r in results),
        "matched_beats_no_evidence": sum(r["matched_beats_no_evidence"] for r in results),
        "median_vision_necessity": st.median(ne_all) if n else None,
        "corr_visionnecessity_vs_matchedgap": corr,
        "leak_l1a": sum(r["leak"]["l1a_gold_in_caption"] for r in results),
        "leak_l1b": sum(r["leak"]["l1b_verdict_phrasing"] for r in results),
        "variance_decomposition": variance_decomposition(records),
        "timings": {"generation_s": t_gen, "scoring_s": t_score},
    }
    Path(args.out).write_text(json.dumps(
        {"summary": summary, "results": results, "captions": caps,
         "trajectories": trajs, "records": records}, indent=2))

    print("\n=== T0b SUMMARY ===")
    print(f"  PHASE 0  C1 bf16 max            : {summary['C1_bf16_max']:.3e}")
    print(f"           C1 fp32 max            : {summary['C1_fp32_max']:.3e}   <- decisive")
    print(f"           C1 fp32 frac > 1e-2    : {summary['C1_fp32_frac_over_1e-2']:.3f}")
    print(f"  PHASE 1  ladder monotonic       : {summary['ladder_monotonic']}/{n}")
    print(f"           matched < vague        : {summary['matched_beats_vague']}/{n}")
    print(f"           matched < no_evidence  : {summary['matched_beats_no_evidence']}/{n}")
    print(f"           median D(no_evidence)  : {summary['median_vision_necessity']:.4f}")
    print(f"           corr(necessity, gap)   : {corr}")
    print(f"  LEAK     L1a gold-in-caption    : {summary['leak_l1a']}/{n}")
    print(f"           L1b verdict phrasing   : {summary['leak_l1b']}/{n}")
    vd = summary["variance_decomposition"]
    print(f"  VAR      between={vd['sigma2_between']:.4g} within={vd['sigma2_within']:.4g} "
          f"snr={vd['signal_to_noise']:.3g}")
    print(f"\n[t0b] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
