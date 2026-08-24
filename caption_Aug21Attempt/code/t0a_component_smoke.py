"""T0a -- the first execution of the caption-distortion stack against a real model.

SCOPE, STATED HONESTLY. This is NOT the training smoke. There is no Ray, no FSDP, no vLLM
and no `fit()`. It loads the backbone with plain transformers and exercises the four
components that have never touched a GPU: `ca21_contexts` (tokenisation), `forward_packed_logits`
(our transcription of dp_actor._forward_micro_batch), `ca21_packing` (window extraction) and
`ca21_estimator` (the KL itself). Those are where SILENT wrongness lives -- a shape or mask
error there yields a finite, plausible distortion that nothing downstream can contradict.
Integration concerns (worker dispatch, vLLM caption sampling) belong to T0b.

FOUR CHECKS, IN INCREASING ORDER OF WHAT THEY WOULD CATCH.

  C1  PACKED == PADDED. `forward_packed_logits` duplicates verl's unpad/rope preamble
      (see ca21_worker's docstring). Here the same sequences go through both the packed path
      and an ordinary padded forward; the response log-probs must agree. This is the check
      that would catch a wrong rope transpose or a mis-indexed unpad -- neither of which
      raises, and both of which corrupt every distribution.

  C2  THE KL ORACLE. Per-position forward KL is >= 0 by construction, so a negative value is
      a bug and never noise. Enforced inside the estimator by exception.

  C3  ENTROPY SPREAD ~ 0. With `p` shared across a caption group, H(sighted) can differ
      between captions ONLY through the mask. Non-zero means some caption's blind sequence
      lost response positions and its D-hat is not comparable to its group-mates' -- which is
      exactly the comparison S12 normalises over.

  C4  DISCRIMINATION -- the one that tests whether the MEASUREMENT MEANS ANYTHING.
      C1-C3 can all pass on an instrument that returns confident numbers unrelated to caption
      quality. So each image is scored against three captions:
          matched     -- the model's own caption of THIS image
          vague       -- contentful-sounding but empty
          mismatched  -- the model's caption of a DIFFERENT image
      A working instrument must rank D(matched) < D(mismatched). If it does not, the caption
      term is measuring something other than how well the caption stands in for the image,
      and R1 would optimise that something for 40 steps and report it as a mechanism.
      **This is a pre-registered prediction, recorded before the run.**
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

VAGUE_CAPTION = (
    "The image shows a scene with several elements arranged in it. There are various "
    "objects and features visible, and the overall composition includes different parts "
    "that relate to one another in the usual way."
)


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
        out.append({
            "problem_id": d["problem_id"][i],
            "problem": str(d["problem"][i]),
            "answer": str(d["answer"][i]),
            "image": Image.open(io.BytesIO(raw)).convert("RGB"),
        })
    return out


def _generate(model, processor, tokenizer, messages, images, max_new_tokens, temperature):
    """One sample from the policy, through the ordinary HF path."""
    import torch

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=images, text=[text], add_special_tokens=False,
                       return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                             top_p=0.99, pad_token_id=tokenizer.pad_token_id)
    gen = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-items", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=12800)
    ap.add_argument("--max-caption-tokens", type=int, default=256)
    ap.add_argument("--max-answer-tokens", type=int, default=256)
    ap.add_argument("--min-pixels", type=int, default=262144)
    ap.add_argument("--max-pixels", type=int, default=4194304)
    args = ap.parse_args()

    import ca21_prompts as P
    from ca21_contexts import append_responses, build_prompt_row
    from ca21_estimator import distortion_from_logits
    from ca21_packing import gather_response_logits, gather_response_logits_reference
    from ca21_worker import forward_packed_logits

    print(f"[t0a] loading {args.model}", flush=True)
    t = time.time()
    model, processor, tokenizer = _load(args.model)
    print(f"[t0a] loaded in {time.time()-t:.1f}s", flush=True)

    from verl.utils.dataset import process_image

    rows = _rows(args.parquet, args.n_items)
    # Resize ONCE, up front, and use the same pixels for generation and for scoring. The
    # training path always sees process_image(min_pixels, max_pixels) output; a caption
    # written from a differently-sized image is not the caption training would produce.
    for r in rows:
        r["pimage"] = process_image(r["image"], args.min_pixels, args.max_pixels)
    print(f"[t0a] {len(rows)} items from {Path(args.parquet).name}", flush=True)

    # ---- captions: the model's own, per image (also exercises build_captioner_messages)
    t = time.time()
    captions = []
    for r in rows:
        c = _generate(model, processor, tokenizer,
                      P.build_captioner_messages(r["problem"]), [r["pimage"]],
                      args.max_caption_tokens, temperature=1.0)
        captions.append(c)
        print(f"[t0a] caption[{r['problem_id']}] {len(c)} chars: {c[:110]!r}", flush=True)
    t_caption = time.time() - t

    # ---- one sighted trajectory y per item (the shared y of S13)
    t = time.time()
    trajectories = []
    for r in rows:
        y = _generate(model, processor, tokenizer,
                      P.build_sighted_messages(r["problem"]), [r["pimage"]],
                      args.max_answer_tokens, temperature=1.0)
        trajectories.append(y)
    t_traj = time.time() - t

    results, timings = [], {"caption_gen_s": t_caption, "trajectory_gen_s": t_traj}
    t_score = time.time()

    for i, r in enumerate(rows):
        y_ids = tokenizer(trajectories[i], add_special_tokens=False,
                          return_tensors="pt")["input_ids"][0]
        T = int(y_ids.shape[0])
        if T < 8:
            print(f"[t0a] SKIP {r['problem_id']}: trajectory only {T} tokens", flush=True)
            continue

        # sighted context: image + question + instruction, then y appended
        s_row = build_prompt_row(processor, tokenizer, P.build_sighted_messages(r["problem"]),
                                 [r["image"]], args.max_prompt_length,
                                 args.min_pixels, args.max_pixels)
        resp = y_ids.unsqueeze(0)
        rmask = torch.ones_like(resp)
        s_ids, s_am, s_pos = append_responses(
            s_row["input_ids"].unsqueeze(0), s_row["attention_mask"].unsqueeze(0),
            s_row["position_ids"].unsqueeze(0), resp, rmask)

        # The image MUST go through process_image(min_pixels, max_pixels) first -- exactly
        # as the worker does (fsdp_workers.py:538-548) and as build_prompt_row did when it
        # sized the image-pad tokens now sitting in s_ids. Feeding pixel_values from the RAW
        # image would produce a grid that disagrees with those tokens.
        mm = {k: v.to(model.device) for k, v in
              processor.image_processor(images=[r["pimage"]], return_tensors="pt").items()}

        dev = model.device
        s_ids, s_am, s_pos = s_ids.to(dev), s_am.to(dev), s_pos.to(dev)

        lg_packed, idx, B, S = forward_packed_logits(
            model, s_ids, s_am, s_pos, multi_modal_inputs=mm, padding_free=True)
        lg_s, m_s = gather_response_logits(lg_packed, idx, B, S, T)

        # ---- C1: the same sequence through the ordinary padded forward
        with torch.no_grad():
            padded = model(input_ids=s_ids, attention_mask=s_am,
                           position_ids=s_pos.transpose(0, 1), **mm, use_cache=False).logits
        lg_ref, m_ref = gather_response_logits_reference(padded, s_am, T)
        lab = resp.to(dev).clamp_min(0).unsqueeze(-1)
        d_packed = lg_s.log_softmax(-1).gather(-1, lab).squeeze(-1)
        d_ref = lg_ref.log_softmax(-1).gather(-1, lab).squeeze(-1)
        c1 = float((d_packed - d_ref).abs().max().item())
        del padded, lg_ref

        # ---- three blind contexts, scored against that ONE sighted pass
        variants = {
            "matched": captions[i],
            "vague": VAGUE_CAPTION,
            "mismatched": captions[(i + 1) % len(captions)],
        }
        per_variant, ent_s = {}, {}
        for name, cap in variants.items():
            b_row = build_prompt_row(processor, tokenizer,
                                     P.build_answerer_messages(cap, r["problem"]),
                                     None, args.max_prompt_length,
                                     args.min_pixels, args.max_pixels)
            b_ids, b_am, b_pos = append_responses(
                b_row["input_ids"].unsqueeze(0), b_row["attention_mask"].unsqueeze(0),
                b_row["position_ids"].unsqueeze(0), resp, rmask)
            lgb_packed, bidx, bB, bS = forward_packed_logits(
                model, b_ids.to(dev), b_am.to(dev), b_pos.to(dev),
                multi_modal_inputs=None, padding_free=True)   # G-BLIND, structurally
            lg_b, m_b = gather_response_logits(lgb_packed, bidx, bB, bS, T)

            res = distortion_from_logits(lg_s, lg_b, m_s * m_b,
                                         labels=resp.to(dev), temperature=1.0)
            per_variant[name] = float(res["kl"].item())
            ent_s[name] = float(res["entropy_p"].item())
            del lgb_packed, lg_b

        del lg_packed, lg_s

        spread = max(ent_s.values()) - min(ent_s.values())
        rec = {
            "problem_id": r["problem_id"], "T": T,
            "c1_packed_vs_padded_max_logprob_diff": c1,
            "distortion": per_variant,
            "entropy_sighted": ent_s,
            "c3_entropy_spread": spread,
            "c4_matched_below_mismatched": per_variant["matched"] < per_variant["mismatched"],
            "caption_chars": len(captions[i]),
        }
        results.append(rec)
        print(f"[t0a] {r['problem_id']}: D matched={per_variant['matched']:.4f} "
              f"vague={per_variant['vague']:.4f} mismatched={per_variant['mismatched']:.4f} "
              f"| C1={c1:.2e} spread={spread:.2e}", flush=True)

    timings["scoring_s"] = time.time() - t_score

    n = len(results)
    summary = {
        "n_items": n,
        "C1_max_packed_vs_padded": max((r["c1_packed_vs_padded_max_logprob_diff"]
                                        for r in results), default=None),
        "C3_max_entropy_spread": max((r["c3_entropy_spread"] for r in results), default=None),
        "C4_matched_below_mismatched": sum(r["c4_matched_below_mismatched"] for r in results),
        "C4_n": n,
        "timings": timings,
    }
    Path(args.out).write_text(json.dumps(
        {"summary": summary, "results": results,
         "captions": captions, "trajectories": trajectories}, indent=2))

    print("\n=== T0a SUMMARY ===")
    print(f"  C1 packed vs padded, max |dlogp| : {summary['C1_max_packed_vs_padded']:.3e}")
    print(f"  C2 KL oracle                     : never fired (it raises)")
    print(f"  C3 max entropy spread            : {summary['C3_max_entropy_spread']:.3e}")
    print(f"  C4 matched < mismatched          : {summary['C4_matched_below_mismatched']}/{n}")
    for k, v in timings.items():
        print(f"  {k:<32}: {v:.1f}s")
    print(f"\n[t0a] -> {args.out}")

    # C4 is REPORTED, not asserted: with n this small a single inversion is not evidence of
    # a bug, and crashing here would discard the diagnostics that explain it.
    if n and summary["C4_matched_below_mismatched"] < n:
        print("\n[t0a] NOTE: matched did not beat mismatched on every item. With n small "
              "this may be sampling; if it persists at scale the instrument is not "
              "measuring caption quality and R1 must not run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
