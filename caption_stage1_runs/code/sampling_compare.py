"""Diagnostic: does off-card sampling explain the answer-format non-compliance?

NOT part of the frozen pipeline. A one-off measurement to settle a decision
with data instead of argument.

Background. The Pilot-0 smoke found Qwen3-VL-2B-Instruct ignoring
"Answer with only the final answer, in \\boxed{}": 75% of answers hit the cap
and only 18% emitted \\boxed{}. We were running deliberately off-card --
D23 chose temperature 1.0 / top_p 1.0 / top_k -1 for estimator unbiasedness,
overriding the vendor preset (0.7 / 0.8 / 20). Qwen3-VL is separately known to
follow output formats poorly (QwenLM/Qwen3-VL#1663) and to degenerate into
repetition under some settings (vllm-project/vllm#27157).

Four conditions on identical items, one model load:

    sampling in {untruncated (D23), model-card}  x  answer prefill in {off, on}

"Prefill" appends ``\\boxed{`` to the assistant turn so the model can only fill
in the answer and close the brace. If it works, the answer prefix becomes fixed
by construction, which both bounds T and makes D-hat insensitive to answer-side
sampling -- removing the tension between the two axes entirely.

Reported per condition: caption length (uncensored at the given cap), answer
length, truncation rate, \\boxed{} rate, EOS-reached rate, and a crude
repetition score to catch degeneration.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cs1_prompts as P  # noqa: E402
from pilot_generate import MAX_PIXELS, MIN_PIXELS, load_images, render  # noqa: E402

PRESETS = {
    # D23 as decided: untruncated, for estimator unbiasedness / on-policy GRPO.
    "untruncated": dict(temperature=1.0, top_p=1.0, top_k=-1),
    # Qwen3-VL vendor preset, from the model's own generation_config.json.
    "model_card": dict(temperature=0.7, top_p=0.8, top_k=20),
}

ANSWER_PREFILL = "\\boxed{"


def repetition_score(text: str) -> float:
    """Crude degeneration detector: 1 - (distinct 5-grams / total 5-grams).

    Near 0 is healthy; approaching 1 means the text loops. Catches the failure
    reported in vllm-project/vllm#27157.
    """
    toks = text.split()
    if len(toks) < 10:
        return 0.0
    grams = [" ".join(toks[i:i + 5]) for i in range(len(toks) - 4)]
    return 1.0 - len(set(grams)) / len(grams)


def summarise(name: str, caps: list[dict], answers: list[dict], eos_id: set[int]) -> dict:
    clen = sorted(c["n_tokens"] for c in caps)
    alen = sorted(len(a["token_ids"]) for a in answers)
    n_a = max(len(answers), 1)
    row = {
        "condition": name,
        "n_captions": len(caps),
        "caption_tok_median": clen[len(clen) // 2] if clen else 0,
        "caption_tok_p90": clen[int(len(clen) * 0.9)] if clen else 0,
        "caption_tok_max": clen[-1] if clen else 0,
        "caption_trunc_rate": sum(1 for c in caps if c["finish_reason"] == "length") / max(len(caps), 1),
        "caption_repetition": round(stats.mean([repetition_score(c["caption"]) for c in caps]), 3) if caps else 0,
        "n_answers": len(answers),
        "answer_tok_median": alen[len(alen) // 2] if alen else 0,
        "answer_tok_max": alen[-1] if alen else 0,
        "answer_trunc_rate": sum(1 for a in answers if a["finish_reason"] == "length") / n_a,
        "boxed_rate": sum(1 for a in answers if "\\boxed{" in a["answer"]) / n_a,
        "eos_reached_rate": sum(1 for a in answers if a["token_ids"] and a["token_ids"][-1] in eos_id) / n_a,
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--n-items", type=int, default=10)
    ap.add_argument("--g-captions", type=int, default=2)
    ap.add_argument("--caption-max-tokens", type=int, default=2048)
    ap.add_argument("--answer-max-tokens", type=int, default=512)
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((Path(args.pool) / "pool_manifest.json").read_text())
    items = manifest["items"][: args.n_items]

    processor = AutoProcessor.from_pretrained(
        args.model, size={"longest_edge": MAX_PIXELS, "shortest_edge": MIN_PIXELS}
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    eos_id = set(tok.all_special_ids) | {tok.eos_token_id}

    images = load_images(manifest, Path(args.snapshot))
    images = {it["index"]: images[it["index"]] for it in items}

    llm = LLM(
        model=args.model, trust_remote_code=True, max_model_len=16384,
        gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"size": {"longest_edge": MAX_PIXELS, "shortest_edge": MIN_PIXELS}},
        seed=0, enforce_eager=False, disable_log_stats=True,
    )

    rows = []
    for preset_name, preset in PRESETS.items():
        print(f"\n########## sampling preset: {preset_name} {preset} ##########", flush=True)

        cap_reqs = [{"prompt": render(processor, P.build_captioner_messages(it["stem"])),
                     "multi_modal_data": {"image": images[it["index"]]}} for it in items]
        cap_out = llm.generate(
            cap_reqs, SamplingParams(n=args.g_captions, max_tokens=args.caption_max_tokens, **preset)
        )
        caps = [{"index": it["index"], "cap_k": k, "caption": c.text,
                 "n_tokens": len(c.token_ids), "finish_reason": c.finish_reason}
                for it, o in zip(items, cap_out) for k, c in enumerate(o.outputs)]

        by_index = {it["index"]: it for it in items}
        for prefill in (False, True):
            reqs = []
            for c in caps:
                it = by_index[c["index"]]
                msgs = P.build_answerer_messages(c["caption"], it["full_text"])
                P.assert_blind(msgs)
                prompt = render(processor, msgs)
                if prefill:
                    prompt = prompt + ANSWER_PREFILL
                reqs.append({"prompt": prompt})
            a_out = llm.generate(
                reqs, SamplingParams(n=1, max_tokens=args.answer_max_tokens, **preset)
            )
            answers = [{"index": c["index"], "cap_k": c["cap_k"],
                        "answer": (ANSWER_PREFILL if prefill else "") + o.outputs[0].text,
                        "token_ids": list(o.outputs[0].token_ids),
                        "finish_reason": o.outputs[0].finish_reason}
                       for c, o in zip(caps, a_out)]

            name = f"{preset_name}/prefill={'on' if prefill else 'off'}"
            row = summarise(name, caps, answers, eos_id)
            rows.append(row)
            print(f"  {name}: {json.dumps(row, indent=None)}", flush=True)
            (out / f"answers_{preset_name}_prefill{int(prefill)}.jsonl").write_text(
                "\n".join(json.dumps(a) for a in answers)
            )
            print(f"    example answer: {answers[0]['answer'][:160]!r}", flush=True)

        (out / f"captions_{preset_name}.jsonl").write_text("\n".join(json.dumps(c) for c in caps))

    (out / "sampling_compare.json").write_text(json.dumps(rows, indent=2))

    print("\n" + "=" * 100)
    hdr = f"{'condition':34s} {'capMed':>7s} {'capP90':>7s} {'capTr':>7s} {'ansMed':>7s} {'ansTr':>7s} {'boxed':>7s} {'eos':>7s} {'rep':>6s}"
    print(hdr)
    print("-" * 100)
    for r in rows:
        print(f"{r['condition']:34s} {r['caption_tok_median']:7d} {r['caption_tok_p90']:7d} "
              f"{r['caption_trunc_rate']:7.1%} {r['answer_tok_median']:7d} {r['answer_trunc_rate']:7.1%} "
              f"{r['boxed_rate']:7.1%} {r['eos_reached_rate']:7.1%} {r['caption_repetition']:6.3f}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
