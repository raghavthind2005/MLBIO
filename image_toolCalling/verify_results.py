#!/usr/bin/env python3
"""
Verify data structure and field completeness for both result files.
Run locally after pulling results from the cluster.

Usage:
  python verify_results.py
  python verify_results.py --normal results_normal/raw_results.jsonl \
                           --tool   results_tool/tool_results.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Fields required in every record of each run type
NORMAL_REQUIRED = [
    "sample_id", "category", "subcategory", "visual_input",
    "question", "gt_answer", "image_path",
    "model_prediction", "answer_text", "thinking_content", "is_correct",
    "prompt_tokens", "completion_tokens",
    "thinking_chars", "n_image_tokens_approx", "visual_token_ratio_approx",
    "inference_time_s",
]

TOOL_REQUIRED = [
    "sample_id", "category", "subcategory", "visual_input",
    "question", "gt_answer", "image_path",
    "model_prediction", "answer_text", "is_correct",
    "thinking_chars", "all_thinking_chars",
    "n_tool_calls", "tool_calls", "total_image_tokens",
    "stages", "total_completion_tokens", "final_prompt_tokens",
    "visual_token_ratio_approx",
    "inference_time_s",
]

# Fields expected only in tool records with n_tool_calls > 0
TOOL_CALL_FIELDS = ["region", "thinking_chars_before", "image_tokens_added"]


def load(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [!] JSON parse error on line {i+1}: {e}")
    return records


def check_fields(records: list[dict], required: list[str], label: str) -> None:
    missing_counts = defaultdict(int)
    none_counts    = defaultdict(int)
    for r in records:
        for f in required:
            if f not in r:
                missing_counts[f] += 1
            elif r[f] is None:
                none_counts[f] += 1

    if missing_counts:
        print(f"  MISSING fields (field: n_records_missing):")
        for f, n in sorted(missing_counts.items(), key=lambda x: -x[1]):
            print(f"    {f:35s}: missing in {n}/{len(records)} records")
    else:
        print(f"  All required fields present in every record.")

    # Fields that are None in more than 5% of records
    noisy = {f: n for f, n in none_counts.items() if n > len(records) * 0.05}
    if noisy:
        print(f"  Fields with >5% None values:")
        for f, n in sorted(noisy.items(), key=lambda x: -x[1]):
            print(f"    {f:35s}: None in {n}/{len(records)} records")


def verify_normal(records: list[dict]) -> None:
    errors  = [r for r in records if "error" in r]
    records = [r for r in records if "error" not in r]
    scored  = [r for r in records if r.get("is_correct") is not None]
    failed  = [r for r in records if r.get("is_correct") is None]
    correct = sum(r["is_correct"] for r in scored)
    if errors:
        print(f"  Error records   : {len(errors)} (skipped)")
        for e in errors[:3]:
            print(f"    {e.get('sample_id','?')}: {str(e.get('error',''))[:80]}")

    print(f"  Total records   : {len(records)}")
    print(f"  Scored          : {len(scored)}  ({correct} correct = {correct/len(scored):.3f})")
    print(f"  Parse failures  : {len(failed)}")

    by_vi = defaultdict(lambda: [0, 0])
    for r in scored:
        by_vi[r["visual_input"]][0] += r["is_correct"]
        by_vi[r["visual_input"]][1] += 1
    print(f"  By visual_input : " +
          "  ".join(f"vi={k}: {c}/{t}={c/t:.3f}" for k,(c,t) in sorted(by_vi.items())))

    # Spot-check token counts
    has_pt = [r for r in records if r.get("prompt_tokens")]
    if has_pt:
        avg_pt = sum(r["prompt_tokens"] for r in has_pt) / len(has_pt)
        avg_ct = sum(r["completion_tokens"] for r in has_pt if r.get("completion_tokens")) / len(has_pt)
        avg_th = sum(r["thinking_chars"]   for r in records if r.get("thinking_chars")) / len(records)
        print(f"  Avg prompt_tok  : {avg_pt:.0f}")
        print(f"  Avg comp_tok    : {avg_ct:.0f}")
        print(f"  Avg think_chars : {avg_th:.0f}")

    check_fields(records, NORMAL_REQUIRED, "normal")


def verify_tool(records: list[dict]) -> None:
    errors  = [r for r in records if "error" in r]
    records = [r for r in records if "error" not in r]
    scored      = [r for r in records if r.get("is_correct") is not None]
    failed      = [r for r in records if r.get("is_correct") is None]
    correct     = sum(r["is_correct"] for r in scored)
    tool_users  = [r for r in records if r.get("n_tool_calls", 0) > 0]
    total_calls = sum(r.get("n_tool_calls", 0) for r in records)
    if errors:
        print(f"  Error records   : {len(errors)} (skipped)")
        for e in errors[:3]:
            print(f"    {e.get('sample_id','?')}: {str(e.get('error',''))[:80]}")

    print(f"  Total records   : {len(records)}")
    print(f"  Scored          : {len(scored)}  ({correct} correct = {correct/len(scored):.3f})")
    print(f"  Parse failures  : {len(failed)}")
    print(f"  Tool users      : {len(tool_users)} ({len(tool_users)/len(records)*100:.1f}%)")
    print(f"  Total tool calls: {total_calls}")

    by_vi = defaultdict(lambda: [0, 0, 0])
    for r in scored:
        by_vi[r.get("visual_input", "?")][0] += r["is_correct"]
        by_vi[r.get("visual_input", "?")][1] += 1
    for r in records:
        by_vi[r.get("visual_input", "?")][2] += r.get("n_tool_calls", 0)
    print(f"  By visual_input : " +
          "  ".join(f"vi={k}: {c}/{t}={c/t:.3f} (tc={tc})" for k,(c,t,tc) in sorted(by_vi.items())))

    # thinking_chars alias check
    mismatch = [r for r in records
                if r.get("thinking_chars") != r.get("all_thinking_chars")]
    if mismatch:
        print(f"  [!] thinking_chars != all_thinking_chars in {len(mismatch)} records")
    else:
        print(f"  thinking_chars alias: consistent in all records")

    # image token count verification
    wrong_tok = [r for r in records
                 if r.get("total_image_tokens") != 256 * (1 + r.get("n_tool_calls", 0))]
    if wrong_tok:
        print(f"  [!] total_image_tokens mismatch in {len(wrong_tok)} records")
    else:
        print(f"  total_image_tokens  : correct in all records (256 × (1+n_tool_calls))")

    # stages structure
    bad_stages = [r for r in records if not r.get("stages")]
    if bad_stages:
        print(f"  [!] missing/empty stages in {len(bad_stages)} records")
    else:
        # Check stages have required keys
        stage_fields = ["turn", "prompt_tokens", "completion_tokens", "finish_reason",
                        "thinking_chars", "answer_raw"]
        missing_stage_fields = defaultdict(int)
        for r in records:
            for stage in r.get("stages", []):
                for f in stage_fields:
                    if f not in stage:
                        missing_stage_fields[f] += 1
        if missing_stage_fields:
            print(f"  [!] Missing stage fields: {dict(missing_stage_fields)}")
        else:
            print(f"  stages structure    : all fields present")

    # tool_calls structure
    for r in tool_users:
        for tc in r.get("tool_calls", []):
            for f in TOOL_CALL_FIELDS:
                if f not in tc:
                    print(f"  [!] tool_call missing field '{f}' in sample {r.get('sample_id')}")

    # Avg token stats
    avg_th = sum(r.get("all_thinking_chars", 0) for r in records) / len(records)
    avg_it = sum(r.get("total_image_tokens", 0) for r in records) / len(records)
    avg_ct = sum(r.get("total_completion_tokens", 0) for r in records) / len(records)
    print(f"  Avg think_chars : {avg_th:.0f}")
    print(f"  Avg total_img_tok:{avg_it:.1f}")
    print(f"  Avg comp_tok    : {avg_ct:.0f}")

    check_fields(records, TOOL_REQUIRED, "tool")

    # Print tool-using samples detail
    if tool_users:
        print(f"\n  Tool-using samples:")
        for r in tool_users:
            tc_info = [(tc["region"], tc.get("thinking_chars_before", "?"))
                       for tc in r.get("tool_calls", [])]
            print(f"    {r['sample_id'][:50]:50s}  "
                  f"vi={r['visual_input']} sub={r.get('subcategory','?'):10s} "
                  f"correct={r.get('is_correct')} tools={r['n_tool_calls']} "
                  f"think={r.get('all_thinking_chars',0)}ch  calls={tc_info}")


FORCED_REQUIRED = TOOL_REQUIRED + [
    "answer_turn0", "pred_turn0", "pred_turn1", "answer_changed",
    "is_correct_turn0", "change_type",
    "thinking_turn0_chars", "thinking_turn1_chars",
    "comp_tokens_turn0", "comp_tokens_turn1",
    "truncated", "finish_turn0", "finish_turn1",
]


def verify_forced(records: list[dict]) -> None:
    errors  = [r for r in records if "error" in r]
    records = [r for r in records if "error" not in r]
    scored  = [r for r in records if r.get("is_correct") is not None]
    failed  = [r for r in records if r.get("is_correct") is None]
    correct = sum(r["is_correct"] for r in scored)
    changed = [r for r in scored if r.get("answer_changed") is True]

    if errors:
        print(f"  Error records   : {len(errors)} (skipped)")

    print(f"  Total records   : {len(records)}")
    print(f"  Scored          : {len(scored)}  ({correct} correct = {correct/len(scored):.3f})")
    print(f"  Parse failures  : {len(failed)}")

    by_vi = defaultdict(lambda: [0, 0])
    for r in scored:
        by_vi[r.get("visual_input","?")][0] += r["is_correct"]
        by_vi[r.get("visual_input","?")][1] += 1
    print(f"  By visual_input : " +
          "  ".join(f"vi={k}: {c}/{t}={c/t:.3f}" for k,(c,t) in sorted(by_vi.items())))

    # Turn-0 vs turn-1 accuracy (the core comparison)
    scored0 = [r for r in records if r.get("is_correct_turn0") is not None]
    correct0 = sum(r["is_correct_turn0"] for r in scored0)
    if scored0:
        acc0 = correct0 / len(scored0)
        acc1 = correct / len(scored)
        print(f"  qAcc turn0      : {correct0}/{len(scored0)} = {acc0:.3f}  (before re-exam)")
        print(f"  qAcc turn1      : {correct}/{len(scored)} = {acc1:.3f}  (after re-exam)")
        print(f"  Δ accuracy      : {acc1 - acc0:+.3f}")

    # Transition breakdown via change_type
    ct = defaultdict(int)
    for r in scored:
        ct[r.get("change_type", "?")] += 1
    print(f"  Transitions     : " +
          "  ".join(f"{k}={v}" for k, v in sorted(ct.items())))
    print(f"    wrong→right (↑, helped): {ct.get('wrong_right', 0)}")
    print(f"    right→wrong (↓, hurt)  : {ct.get('right_wrong', 0)}")

    # Truncation / budget check
    truncated = [r for r in records if r.get("truncated")]
    print(f"  Truncated       : {len(truncated)}/{len(records)} (hit token ceiling)")
    if truncated:
        for r in truncated[:5]:
            print(f"    {r['sample_id']}: t0_fin={r.get('finish_turn0')} t1_fin={r.get('finish_turn1')} "
                  f"t1_comp={r.get('comp_tokens_turn1')}")

    # Token verification
    wrong_tok = [r for r in records
                 if r.get("total_image_tokens") != 512 and r.get("image_path")]
    if wrong_tok:
        print(f"  [!] total_image_tokens != 512 in {len(wrong_tok)} records")
    else:
        print(f"  total_image_tokens  : all 512 ✓")

    wrong_stages = [r for r in records if len(r.get("stages", [])) != 2]
    if wrong_stages:
        print(f"  [!] stages length != 2 in {len(wrong_stages)} records")
    else:
        print(f"  stages length       : all 2 ✓")

    avg_t0 = sum(r.get("thinking_turn0_chars", 0) for r in records) / len(records)
    avg_t1 = sum(r.get("thinking_turn1_chars", 0) for r in records) / len(records)
    print(f"  Avg thinking t0 : {avg_t0:.0f} chars")
    print(f"  Avg thinking t1 : {avg_t1:.0f} chars")

    check_fields(records, FORCED_REQUIRED, "forced")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", default=None)
    ap.add_argument("--tool",   default=None)
    ap.add_argument("--forced", default=None)
    args = ap.parse_args()

    normal_path = Path(args.normal) if args.normal else SCRIPT_DIR / "results_normal" / "raw_results.jsonl"
    tool_path   = Path(args.tool)   if args.tool   else SCRIPT_DIR / "results_tool"   / "tool_results.jsonl"
    forced_path = Path(args.forced) if args.forced else SCRIPT_DIR / "results_forced" / "forced_results.jsonl"

    print("=" * 60)
    print(f"NORMAL RUN: {normal_path}")
    print("=" * 60)
    if normal_path.exists():
        verify_normal(load(normal_path))
    else:
        print(f"  [!] File not found: {normal_path}")

    print()
    print("=" * 60)
    print(f"TOOL RUN:   {tool_path}")
    print("=" * 60)
    if tool_path.exists():
        verify_tool(load(tool_path))
    else:
        print(f"  [!] File not found: {tool_path}")

    print()
    print("=" * 60)
    print(f"FORCED RUN: {forced_path}")
    print("=" * 60)
    if forced_path.exists():
        verify_forced(load(forced_path))
    else:
        print(f"  [!] File not found (not yet run): {forced_path}")


if __name__ == "__main__":
    main()
