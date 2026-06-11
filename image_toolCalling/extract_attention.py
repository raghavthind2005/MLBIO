#!/usr/bin/env python3
"""
Extract attention weights from Gemma-4 via teacher-forcing.

Run AFTER run_eval.py. Requires sglang to be stopped first so the model
can be reloaded under HuggingFace with eager attention.

Strategy:
  For each saved (prompt + response), reconstruct the full token sequence
  and do a single forward pass with output_attentions=True (eager mode).
  For each layer, compute the mean attention each output token pays to:
    - visual tokens    (image soft tokens in the input)
    - system tokens    (system turn)
    - instruction tokens (user text, excluding image tokens)
  Store per-position and per-layer summaries — not the full O(n²) matrices.

Usage:
    python extract_attention.py [--model /path/to/model] [--max-samples N]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_PATH = "/capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it"

SYSTEM_PROMPT = (
    "You are a helpful visual question answering assistant. "
    "After your reasoning, answer ONLY with the single word 'Yes' or 'No'."
)


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model_and_processor(model_path: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)

    print(f"Loading model (bf16, eager attn)...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # flash-attn does not return attention weights
    )
    model.eval()
    print(f"Model loaded. Devices: {set(model.hf_device_map.values())}")
    return model, processor


def get_image_token_id(processor) -> int:
    """Find the ID of the image placeholder token used in input_ids."""
    # Try processor attributes first
    for attr in ("image_token_id", "image_token"):
        val = getattr(processor, attr, None)
        if val is not None:
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                tid = processor.tokenizer.convert_tokens_to_ids(val)
                if tid != processor.tokenizer.unk_token_id:
                    return tid
    # Try common token strings
    for tok in ("<image_soft_token>", "<image>", "<img>", "█"):
        tid = processor.tokenizer.convert_tokens_to_ids(tok)
        if tid != processor.tokenizer.unk_token_id:
            return tid
    raise RuntimeError(
        "Could not find image token ID. Check processor.tokenizer.special_tokens_map "
        "and set IMAGE_TOKEN_ID manually."
    )


# ─── Input construction ───────────────────────────────────────────────────────

def build_inputs(sample: dict, processor, device: torch.device):
    """
    Construct teacher-forcing inputs: full (prompt + saved response) as one sequence.
    Returns dict with input_ids, attention_mask, pixel_values (if image), and
    a token_groups dict mapping group name -> list of token positions.
    """
    img = None
    if sample.get("image_path"):
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            print(f"  [warn] could not open image: {e}")

    # Reconstruct the model's response text (thinking + answer)
    thinking = sample.get("thinking_content", "")
    answer   = sample.get("answer_text", "")
    if thinking:
        assistant_text = f"<think>\n{thinking}\n</think>\n{answer}"
    else:
        assistant_text = answer

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": img}, {"type": "text", "text": sample["question"]}]
                if img else sample["question"]
            ),
        },
        {"role": "assistant", "content": assistant_text},
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        # Older processor versions may not accept all kwargs; fall back
        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        inputs = processor(
            text=text,
            images=[img] if img else None,
            return_tensors="pt",
        )

    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
    return inputs, img


def identify_token_groups(
    input_ids: list[int],
    processor,
    image_token_id: int,
) -> dict[str, list[int]]:
    """
    Returns position lists for: visual, system, instruction, output.
    Uses <start_of_turn> / <end_of_turn> markers when available.
    """
    tok = processor.tokenizer
    try:
        sot = tok.convert_tokens_to_ids("<start_of_turn>")
        eot = tok.convert_tokens_to_ids("<end_of_turn>")
        sot_valid = (sot != tok.unk_token_id)
    except Exception:
        sot_valid = False

    visual_pos = [i for i, t in enumerate(input_ids) if t == image_token_id]

    if sot_valid:
        turn_starts = [i for i, t in enumerate(input_ids) if t == sot]
        # Expected: [system_turn, user_turn, model_turn]
        if len(turn_starts) >= 3:
            s_start, u_start, m_start = turn_starts[0], turn_starts[1], turn_starts[2]
            system_pos      = list(range(s_start, u_start))
            user_all        = set(range(u_start, m_start))
            instruction_pos = sorted(user_all - set(visual_pos))
            output_pos      = list(range(m_start, len(input_ids)))
        elif len(turn_starts) == 2:
            # No explicit system turn
            u_start, m_start = turn_starts[0], turn_starts[1]
            system_pos      = list(range(0, u_start))
            user_all        = set(range(u_start, m_start))
            instruction_pos = sorted(user_all - set(visual_pos))
            output_pos      = list(range(m_start, len(input_ids)))
        else:
            # Fallback: everything before first visual token is "system"
            first_vis = visual_pos[0] if visual_pos else 0
            system_pos      = list(range(0, first_vis))
            instruction_pos = []
            output_pos      = list(range(visual_pos[-1] + 1 if visual_pos else 0, len(input_ids)))
    else:
        # Heuristic fallback
        first_vis = visual_pos[0] if visual_pos else len(input_ids) // 4
        last_vis  = visual_pos[-1] if visual_pos else first_vis
        system_pos      = list(range(0, first_vis))
        instruction_pos = list(range(last_vis + 1, last_vis + 1 + 40))  # rough
        output_pos      = list(range(last_vis + 50, len(input_ids)))

    return {
        "visual":      visual_pos,
        "system":      system_pos,
        "instruction": instruction_pos,
        "output":      output_pos,
    }


# ─── Attention extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_attention(
    model,
    processor,
    sample: dict,
    image_token_id: int,
    max_seq_len: int = 4096,
) -> dict | None:
    device = next(model.parameters()).device
    inputs, img = build_inputs(sample, processor, device)

    input_ids = inputs["input_ids"]
    seq_len   = input_ids.shape[1]

    if seq_len > max_seq_len:
        print(f"  skip (seq_len={seq_len} > {max_seq_len})")
        return None

    ids    = input_ids[0].tolist()
    groups = identify_token_groups(ids, processor, image_token_id)

    n_output = len(groups["output"])
    if n_output == 0:
        print(f"  skip (no output positions found)")
        return None

    # Forward pass — returns attentions tuple: one tensor per layer [B, H, S, S]
    outputs = model(**inputs, output_attentions=True, return_dict=True)

    attentions = outputs.attentions  # tuple of [1, n_heads, seq, seq]
    n_layers   = len(attentions)

    # For each layer: compute mean attention from output positions → each token group
    # Immediately reduce to save memory (don't hold all full matrices at once).
    out_idx = groups["output"]
    group_sets = {
        "visual":      np.array(sorted(set(groups["visual"])     & set(range(seq_len))), dtype=int),
        "system":      np.array(sorted(set(groups["system"])     & set(range(seq_len))), dtype=int),
        "instruction": np.array(sorted(set(groups["instruction"])& set(range(seq_len))), dtype=int),
    }

    # shape: [n_layers, n_output] for each group
    per_layer_per_pos = {g: np.zeros((n_layers, n_output), dtype=np.float32) for g in group_sets}

    for layer_i, attn_layer in enumerate(attentions):
        # attn_layer: [1, n_heads, seq_len, seq_len]
        # Average over heads, take output rows
        attn_np = attn_layer[0].mean(dim=0).float().cpu().numpy()  # [seq, seq]
        out_rows = attn_np[out_idx, :]  # [n_output, seq_len]

        for gname, cols in group_sets.items():
            if cols.size > 0:
                per_layer_per_pos[gname][layer_i] = out_rows[:, cols].sum(axis=1)

        del attn_layer
        torch.cuda.empty_cache()

    # ── Summarise ─────────────────────────────────────────────────────────────
    def thirds(arr: np.ndarray) -> dict:
        n = len(arr)
        if n < 3:
            v = float(arr.mean())
            return {"early": v, "mid": v, "late": v}
        t1, t2 = n // 3, 2 * n // 3
        return {
            "early": float(arr[:t1].mean()),
            "mid":   float(arr[t1:t2].mean()),
            "late":  float(arr[t2:].mean()),
        }

    result = {
        "sample_id":         sample["sample_id"],
        "category":          sample.get("category"),
        "subcategory":       sample.get("subcategory"),
        "visual_input":      sample.get("visual_input"),
        "is_correct":        sample.get("is_correct"),
        "thinking_chars":    sample.get("thinking_chars"),
        "completion_tokens": sample.get("completion_tokens"),

        # Sequence structure
        "seq_len":              seq_len,
        "n_layers":             n_layers,
        "n_visual_tokens":      len(groups["visual"]),
        "n_system_tokens":      len(groups["system"]),
        "n_instruction_tokens": len(groups["instruction"]),
        "n_output_tokens":      n_output,

        # Global mean attention (scalar)
        "attn_visual_mean":      float(per_layer_per_pos["visual"].mean()),
        "attn_system_mean":      float(per_layer_per_pos["system"].mean()),
        "attn_instruction_mean": float(per_layer_per_pos["instruction"].mean()),

        # Mean over layers → per output position [list of n_output floats]
        "attn_visual_per_pos":      per_layer_per_pos["visual"].mean(axis=0).tolist(),
        "attn_system_per_pos":      per_layer_per_pos["system"].mean(axis=0).tolist(),
        "attn_instruction_per_pos": per_layer_per_pos["instruction"].mean(axis=0).tolist(),

        # Mean over output positions → per layer [list of n_layers floats]
        "attn_visual_per_layer":      per_layer_per_pos["visual"].mean(axis=1).tolist(),
        "attn_system_per_layer":      per_layer_per_pos["system"].mean(axis=1).tolist(),
        "attn_instruction_per_layer": per_layer_per_pos["instruction"].mean(axis=1).tolist(),

        # Attention at early / mid / late thirds of reasoning
        "attn_visual_by_thirds":      thirds(per_layer_per_pos["visual"].mean(axis=0)),
        "attn_system_by_thirds":      thirds(per_layer_per_pos["system"].mean(axis=0)),
        "attn_instruction_by_thirds": thirds(per_layer_per_pos["instruction"].mean(axis=0)),
    }

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",     default=None, help="Path to raw_results.jsonl")
    ap.add_argument("--model",       default=MODEL_PATH)
    ap.add_argument("--out",         default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    args = ap.parse_args()

    script_dir   = Path(__file__).parent
    results_path = Path(args.results) if args.results else script_dir / "results" / "raw_results.jsonl"
    out_path     = Path(args.out) if args.out else script_dir / "results" / "attention_results.jsonl"

    # Load eval results (skip errored lines)
    samples = []
    with open(results_path) as f:
        for line in f:
            rec = json.loads(line)
            if "error" not in rec and rec.get("answer_text") is not None:
                samples.append(rec)

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Extracting attention for {len(samples)} samples")
    print(f"Output: {out_path}\n")

    model, processor = load_model_and_processor(args.model)
    image_token_id   = get_image_token_id(processor)
    print(f"Image token ID: {image_token_id}\n")

    out_path.parent.mkdir(exist_ok=True)
    n_done = n_skip = 0

    with open(out_path, "w") as fout:
        for i, sample in enumerate(samples):
            print(f"[{i+1:3d}/{len(samples)}] {sample['sample_id']}", end="  ", flush=True)
            try:
                result = extract_attention(
                    model, processor, sample, image_token_id, args.max_seq_len
                )
                if result:
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
                    n_done += 1
                    print(
                        f"vis={result['n_visual_tokens']} "
                        f"attn_vis={result['attn_visual_mean']:.4f} "
                        f"attn_ins={result['attn_instruction_mean']:.4f} "
                        f"attn_sys={result['attn_system_mean']:.4f}"
                    )
                else:
                    n_skip += 1
            except Exception as exc:
                print(f"ERROR: {exc}")
                import traceback; traceback.print_exc()
                n_skip += 1

    print(f"\nDone. Extracted: {n_done}, skipped: {n_skip}")
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
