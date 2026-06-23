#!/usr/bin/env python3
"""
Attention extraction for BabyVision two-turn B1'/B2' (b1cot / b2cot) ONLY.

THE QUESTION (Yulun, 2026-06-22): how does attention to the IMAGE behave across the
turn-2 reasoning, and how does that differ between re-injecting the image vs not?

  - b1cot  (--reinject):  the image is RE-INJECTED in turn 2 → the sequence has TWO
    visual blocks (visual_turn0 = original image, visual_turn1 = the re-shown image,
    i.e. ~2x as many image tokens). Does the turn-2 reasoning actually attend to the
    fresh re-injected image, or does it ignore it and lean on the folded text?

  - b2cot  (no reinject):  the image appears only in turn 0 → ONE visual block, now
    sitting far behind a long folded-reasoning user turn (same image-token count as
    the standard single-turn condition, but a much longer reasoning context). Does
    attention to that single image decay across the turn-2 reasoning ("see less")?

FAITHFULNESS (this is the whole point — see the strip bug Yulun caught):
Gemma-4's chat template DROPS the thought channel from assistant turns, so we must
NOT let apply_chat_template render the turn-2 model output — it would silently
vanish. Instead we:
  (A) build the EXACT turn-2 prompt the model saw — the same message structure as
      run_infer_b.build_turn2_prompt (turn-1 reasoning folded into the turn-2 USER
      message, image re-injected iff b1cot) — tokenized WITH the image; then
  (B) APPEND the model's actual turn-2 generation as RAW tokens
      (THINK_OPEN + thinking_trace + THINK_CLOSE + answer_text), bypassing the
      template entirely so the reasoning is guaranteed present.
Attention is scored FROM the appended turn-2 positions TO each visual block.
A self-check asserts our reconstructed prompt string matches build_turn2_prompt's,
so this can never silently drift from what inference actually fed the model.

No sglang server — loads Gemma-4 via HF (eager attn, device_map=auto) for
teacher-forcing. Run with sglang stopped so the GPUs are free.

Usage (see babyvision_attn_b_job.sh):
  python extract_attention_b.py --condition b1cot \
      --results /.../results_b1cot_reinject/results_run1.jsonl \
      --out     /.../results_b1cot_reinject/attention_b.jsonl
  python extract_attention_b.py --condition b2cot \
      --results /.../results_b2cot_noreinject/results_run1.jsonl \
      --out     /.../results_b2cot_noreinject/attention_b.jsonl
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Import the EXACT prompt constants + builder used at inference so this extractor
# can never drift from what the model actually saw. run_infer_b.py is colocated
# in this directory on the cluster (code/babyvision/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_infer_b import THINK_OPEN, THINK_CLOSE, build_turn2_prompt  # noqa: E402

MODEL_PATH = "/capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it"
DIAGNOSE = False

# A CUDA launch failure / IMA / device-side assert tears down the whole CUDA
# context — every subsequent forward then fails. Detect it so we can exit and let
# the job's retry loop relaunch a fresh process rather than limp through failures.
_FATAL_CUDA_MARKERS = (
    "unspecified launch failure", "illegal memory access", "device-side assert",
    "CUDA error", "CUBLAS_STATUS", "an illegal instruction",
)


def _is_fatal_cuda(exc: Exception) -> bool:
    s = str(exc)
    return any(m in s for m in _FATAL_CUDA_MARKERS)


def _is_oom(exc: Exception) -> bool:
    """Recoverable CUDA out-of-memory (the transient didn't fit) — free and skip,
    NOT a context-corrupting fatal error."""
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


# ─── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_processor(model_path: str, mem_cap_gib: int = 55):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    # CAP per-GPU memory so device_map can't pack GPU 0 full (the OOM cause): the
    # ~62 GB model then spreads ~15 GB/GPU, leaving each GPU ~40 GB free for the
    # O(seq^2) eager-attention transient. Without this, "auto" filled GPU 0 to ~88 GB
    # and every long-sequence forward OOM'd.
    n_gpus = torch.cuda.device_count()
    max_memory = {i: f"{mem_cap_gib}GiB" for i in range(n_gpus)}
    print(f"Loading model (bf16, eager attn, {n_gpus} GPUs capped at {mem_cap_gib}GiB)...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        attn_implementation="eager",   # required — flash-attn doesn't return weights
    )
    model.eval()
    print(f"Model loaded. Devices: {sorted(set(model.hf_device_map.values()))}")
    return model, processor


def get_image_token_id(processor) -> int:
    for attr in ("image_token_id", "image_token"):
        val = getattr(processor, attr, None)
        if val is not None:
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                tid = processor.tokenizer.convert_tokens_to_ids(val)
                if tid is not None and tid != processor.tokenizer.unk_token_id:
                    return tid
    for tok in ("<|image|>", "<image_soft_token>", "<image>", "<img>"):
        tid = processor.tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != processor.tokenizer.unk_token_id:
            return tid
    raise RuntimeError("Could not find image token ID.")


def find_visual_groups(ids: list, image_token_id: int) -> list:
    """Contiguous runs of image_token_id — one run per injected image, in order."""
    groups, cur = [], []
    for i, t in enumerate(ids):
        if t == image_token_id:
            if not cur or i == cur[-1] + 1:
                cur.append(i)
            else:
                groups.append(cur); cur = [i]
        else:
            if cur:
                groups.append(cur); cur = []
    if cur:
        groups.append(cur)
    return groups


# ─── Faithful prompt reconstruction (mirrors run_infer_b.build_turn2_prompt) ─────

def _turn2_user_text(t1_thinking: str, t1_answer: str) -> str:
    """MUST stay byte-identical to run_infer_b.build_turn2_prompt's fold_reasoning
    branch (lines ~177-198). The self-check in build_part_a() enforces this."""
    reasoning = (t1_thinking or "").strip()
    answer    = (t1_answer or "").strip()
    if reasoning:
        return (
            "On your first pass at this question you reasoned as follows:\n\n"
            f'"""\n{reasoning}\n"""\n\n'
            f"and you gave the answer: {answer}\n\n"
            "Look at the image again and re-examine that reasoning carefully — "
            "check each observation against what you actually see. Then give your "
            "final answer in \\boxed{Answer}."
        )
    return (
        "On your first pass you began working through this question:\n\n"
        f'"""\n{answer}\n"""\n\n'
        "You did not reach a final answer. Look at the image again, re-examine "
        "what you see, and give your final answer in \\boxed{Answer}."
    )


def build_part_a_messages(question, t1_thinking, t1_answer, img, reinject):
    """The turn-2 PROMPT messages, mirroring build_turn2_prompt's message list."""
    t1_model_content = f"{THINK_OPEN}{t1_thinking}{THINK_CLOSE}\n{t1_answer}"
    turn2_user = []
    if reinject:
        turn2_user.append({"type": "image", "image": img})
    turn2_user.append({"type": "text", "text": _turn2_user_text(t1_thinking, t1_answer)})
    return [
        {"role": "user", "content": [{"type": "image", "image": img},
                                     {"type": "text", "text": question}]},
        {"role": "assistant", "content": t1_model_content},
        {"role": "user", "content": turn2_user},
    ]


def reconstruct_t2_generation(t2_thinking: str, t2_answer: str) -> str:
    """The raw turn-2 tokens the model produced, after the prompt's <|turn>model\\n.
    Concluded: THINK_OPEN + thinking + THINK_CLOSE + \\n + answer.
    Runaway (no closed channel): THINK_OPEN + answer (the cut-off output)."""
    if (t2_thinking or "").strip():
        return f"{THINK_OPEN}{t2_thinking}{THINK_CLOSE}\n{t2_answer}"
    return f"{THINK_OPEN}{t2_answer}"


# ─── Attention extraction ────────────────────────────────────────────────────────

def thirds(arr: np.ndarray) -> dict:
    n = len(arr)
    if n < 3:
        v = float(arr.mean()) if n else 0.0
        return {"early": v, "mid": v, "late": v}
    t1, t2 = n // 3, 2 * n // 3
    return {"early": float(arr[:t1].mean()),
            "mid":   float(arr[t1:t2].mean()),
            "late":  float(arr[t2:].mean())}


@torch.no_grad()
def extract(model, processor, rec, img_path, image_token_id, reinject, max_seq_len):
    device = next(model.parameters()).device
    tok = processor.tokenizer

    question    = rec.get("question_sent", "")
    t1_thinking = rec.get("turn1_thinking") or ""
    t1_answer   = rec.get("turn1_answer_text") or ""
    t2_thinking = rec.get("thinking_trace") or ""
    t2_answer   = rec.get("answer_text") or ""

    img = Image.open(img_path).convert("RGB")

    # ── Part A: the exact turn-2 prompt, tokenized WITH the image(s) ────────────
    msgs = build_part_a_messages(question, t1_thinking, t1_answer, img, reinject)
    a = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )

    # Self-check: our reconstruction must match inference's build_turn2_prompt.
    ref_str, ref_nimg = build_turn2_prompt(processor, question, t1_thinking, t1_answer,
                                           reinject, enable_thinking=True, fold_reasoning=True)
    our_str = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True)
    if our_str != ref_str:
        # Non-fatal but loud — means this script drifted from run_infer_b.
        print(f"  [WARN id={rec.get('taskId')}] reconstructed prompt != build_turn2_prompt "
              f"(len {len(our_str)} vs {len(ref_str)})")

    a_ids  = a["input_ids"]
    len_a  = a_ids.shape[1]

    # ── Part B: append the model's ACTUAL turn-2 generation as raw tokens ───────
    t2_raw = reconstruct_t2_generation(t2_thinking, t2_answer)
    b_ids  = tok(t2_raw, add_special_tokens=False, return_tensors="pt")["input_ids"]
    len_b  = b_ids.shape[1]

    seq_len = len_a + len_b
    if seq_len > max_seq_len:
        print(f"  skip id={rec.get('taskId')} (seq_len={seq_len} > {max_seq_len})")
        return None
    if len_b == 0:
        print(f"  skip id={rec.get('taskId')} (no turn-2 output tokens)")
        return None

    # Assemble full inputs: input_ids = [A | B]; image-aligned aux tensors come from
    # A (all image tokens live in A). Seq-length-aligned tensors get extended for B
    # (text → attention 1, token_type 0).
    full_ids = torch.cat([a_ids, b_ids], dim=1)
    inputs = {"input_ids": full_ids.to(device)}
    for k, v in a.items():
        if k == "input_ids" or not torch.is_tensor(v):
            continue
        if v.dim() == 2 and v.shape[1] == len_a:          # seq-aligned (mask, type ids)
            pad = torch.ones((1, len_b), dtype=v.dtype) if k == "attention_mask" \
                  else torch.zeros((1, len_b), dtype=v.dtype)
            inputs[k] = torch.cat([v, pad], dim=1).to(device)
        else:                                              # image tensors (pixel_values…)
            inputs[k] = v.to(device)
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones((1, seq_len), dtype=torch.long, device=device)

    ids = full_ids[0].tolist()
    visual_groups = find_visual_groups(ids, image_token_id)
    if not visual_groups:
        print(f"  skip id={rec.get('taskId')} (no visual tokens found)")
        return None

    # Output positions = the appended turn-2 generation.
    out_pos = list(range(len_a, seq_len))
    out_idx = np.array(out_pos, dtype=int)

    # Named groups: each visual block + the turn-2 folded-reasoning user text.
    group_arrays = {}
    for gi, grp in enumerate(visual_groups):
        group_arrays[f"visual_turn{gi}"] = np.array(grp, dtype=int)
    all_visual = sorted(p for grp in visual_groups for p in grp)
    group_arrays["visual_all"] = np.array(all_visual, dtype=int)

    if DIAGNOSE:
        print(f"\n  [diag id={rec.get('taskId')}] seq_len={seq_len} len_a={len_a} "
              f"len_b={len_b} n_visual_groups={len(visual_groups)} "
              f"vis_sizes={[len(g) for g in visual_groups]}")
        print(f"  [diag] reinject={reinject} t2_raw_head={t2_raw[:80]!r}")

    # ── Forward with per-layer attention hooks (bounded memory) ─────────────────
    def _layer_idx(name):
        m = re.search(r"layers\.(\d+)\.", name)
        return int(m.group(1)) if m else -1

    attn_modules = [
        (n, m) for n, m in model.named_modules()
        if n.endswith(".self_attn") and "layers." in n
        and "vision" not in n.lower() and "audio" not in n.lower()
    ]
    attn_modules.sort(key=lambda x: _layer_idx(x[0]))
    n_layers = len(attn_modules)
    captured = {}

    def make_hook(li):
        def hook(module, inp, out):
            if not isinstance(out, tuple) or len(out) < 2 or out[1] is None:
                return out
            wm = out[1][0].mean(dim=0)                       # [q, kv] mean over heads
            idx = torch.as_tensor(out_idx, device=wm.device)
            rows = wm.index_select(0, idx).float().cpu().numpy()   # [n_out, kv]
            captured[li] = {g: (rows[:, cols].sum(axis=1) if cols.size
                                else np.zeros(len(out_idx), dtype=np.float32))
                            for g, cols in group_arrays.items()}
            return (out[0], None) + tuple(out[2:])
        return hook

    handles = [m.register_forward_hook(make_hook(i)) for i, (_, m) in enumerate(attn_modules)]
    try:
        # use_cache=False: no generation, so skip the KV cache (pure memory waste).
        # The hook nulls each layer's attention weights immediately, so only one
        # layer's transient [heads, seq, seq] is live at a time.
        out = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
        del out                       # drop logits/outputs before the next sample
    finally:
        for h in handles:
            h.remove()
    del inputs
    torch.cuda.empty_cache()

    # [n_layers, n_out] per group
    per = {g: np.zeros((n_layers, len(out_pos)), dtype=np.float32) for g in group_arrays}
    for li in range(n_layers):
        lr = captured.get(li)
        if lr is None:
            continue
        for g in group_arrays:
            per[g][li] = lr[g]

    result = {
        "taskId":   rec.get("taskId"),
        "condition": rec.get("condition"),
        "type":     rec.get("type"),
        "subtype":  rec.get("subtype"),
        "ansType":  rec.get("ansType"),
        "grade":    rec.get("grade"),               # present if graded file was used
        "reinject": reinject,
        "seq_len":  seq_len,
        "len_prompt": len_a,
        "n_output_tokens": len_b,                   # turn-2 reasoning length
        "n_layers": n_layers,
        "n_visual_groups": len(visual_groups),
        "n_visual_tokens_per_group": [len(g) for g in visual_groups],
        "t1_runaway": not bool((t1_thinking or "").strip()),
        "t2_runaway": not bool((t2_thinking or "").strip()),
    }
    for g, arr in per.items():
        per_pos = arr.mean(axis=0)                  # [n_out] avg over layers
        result[f"attn_{g}_mean"]      = float(arr.mean())
        result[f"attn_{g}_by_thirds"] = thirds(per_pos)          # decay across reasoning
        result[f"attn_{g}_per_layer"] = arr.mean(axis=1).tolist()  # [n_layers]
        result[f"attn_{g}_per_pos"]   = per_pos.tolist()         # [n_out] (full curve)
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["b1cot", "b2cot"],
                    help="b1cot = reinject (2 visual blocks); b2cot = no reinject")
    ap.add_argument("--results", required=True, help="results_run1.jsonl (or _graded)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=None,
                    help="dir holding the image files (image_file is relative to it). "
                         "Default: <repo>/repo/data/babyvision_data next to this script.")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=12288,
                    help="skip samples longer than this (eager attn is O(seq^2))")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    global DIAGNOSE
    DIAGNOSE = args.diagnose
    reinject = (args.condition == "b1cot")

    here = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else here / "repo" / "data" / "babyvision_data"

    # Prefer a graded file if it sits beside the raw results (carries `grade`).
    results_path = Path(args.results)
    graded = results_path.with_name(results_path.stem + "_graded" + results_path.suffix)
    if results_path.name == "results_run1.jsonl":
        cand = results_path.with_name("results_run1_graded.jsonl")
        if cand.exists():
            results_path = cand
            print(f"Using graded results: {results_path}")

    recs = []
    with open(results_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" not in r and r.get("answer_text") is not None:
                recs.append(r)

    out_path    = Path(args.out)
    poison_path = out_path.with_suffix(out_path.suffix + ".poison")   # crashed/errored
    skip_path   = out_path.with_suffix(out_path.suffix + ".skip")     # too long / no output

    def _load_ids(path):
        ids = set()
        if path.exists():
            for line in open(path):
                t = line.strip()
                if t:
                    try:
                        ids.add(int(t))
                    except ValueError:
                        ids.add(t)
        return ids

    done = set()
    if args.skip_existing:
        if out_path.exists():
            with open(out_path) as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["taskId"])
                    except Exception:
                        pass
        n_extracted = len(done)
        # Also skip taskIds permanently unprocessable (too long / no output → .skip)
        # or that crashed a prior attempt (.poison), so a relaunch only does genuinely
        # NEW work and the retry loop converges instead of re-tokenizing the long tail.
        poison = _load_ids(poison_path)
        toolong = _load_ids(skip_path)
        done |= poison | toolong
        recs = [r for r in recs if r.get("taskId") not in done]
        print(f"--skip-existing: {n_extracted} extracted + {len(toolong)} too-long + "
              f"{len(poison)} quarantined, {len(recs)} remaining")

    if args.max_samples:
        recs = recs[:args.max_samples]

    print(f"Condition={args.condition} (reinject={reinject}) — {len(recs)} samples")
    print(f"Data dir: {data_dir}")
    print(f"Output:   {out_path}\n")

    model, processor = load_model_and_processor(args.model)
    image_token_id = get_image_token_id(processor)
    print(f"Image token ID: {image_token_id}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_done = n_skip = 0
    with open(out_path, "a" if args.skip_existing else "w") as fout:
        for i, rec in enumerate(recs):
            img_path = data_dir / rec["image_file"]
            print(f"[{i+1:3d}/{len(recs)}] id={rec.get('taskId')}", end="  ", flush=True)
            try:
                res = extract(model, processor, rec, img_path, image_token_id,
                              reinject, args.max_seq_len)
                if res:
                    fout.write(json.dumps(res) + "\n"); fout.flush()
                    n_done += 1
                    extra = (f" v0={res.get('attn_visual_turn0_mean', 0):.4f}"
                             f" v1={res.get('attn_visual_turn1_mean', float('nan')):.4f}"
                             if reinject else
                             f" v0={res.get('attn_visual_turn0_mean', 0):.4f}")
                    print(f"seq={res['seq_len']} out={res['n_output_tokens']}"
                          f" vis_groups={res['n_visual_groups']}{extra}")
                else:
                    # Permanent skip (too long / no output / no visual) — record so a
                    # relaunch doesn't re-tokenize + re-preprocess it every attempt.
                    with open(skip_path, "a") as sf:
                        sf.write(f"{rec.get('taskId')}\n")
                    n_skip += 1
            except Exception as exc:
                # CUDA OOM is RECOVERABLE — free the failed allocation and skip this
                # (too-big) sample WITHOUT poisoning it or tearing down the run. This
                # is the common case (long eager-attention transient doesn't fit), not
                # a corruption, so the next (smaller) sample proceeds fine.
                if _is_oom(exc):
                    print(f"OOM (skip, too big for memory): tried "
                          f"{str(exc).split('Tried to allocate')[-1][:24].strip()}")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    with open(skip_path, "a") as sf:
                        sf.write(f"{rec.get('taskId')}\n")
                    n_skip += 1
                    continue
                print(f"ERROR: {exc}")
                # Non-OOM error: quarantine so a relaunch doesn't re-hit it forever.
                with open(poison_path, "a") as pf:
                    pf.write(f"{rec.get('taskId')}\n")
                if _is_fatal_cuda(exc):
                    # The CUDA context is now dead — every later sample would also
                    # fail. Exit immediately (rc=3) so the job's retry loop relaunches
                    # a FRESH process (new context) and resumes via --skip-existing.
                    print(f"\nFATAL CUDA error — context corrupted, exiting for fresh "
                          f"relaunch. Extracted this attempt: {n_done}.")
                    sys.stdout.flush()
                    os._exit(3)
                import traceback; traceback.print_exc()
                n_skip += 1

    print(f"\nDone. Extracted: {n_done}, skipped: {n_skip}")
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
