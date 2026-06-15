#!/usr/bin/env python3
"""
Extract attention weights from Gemma-4 via teacher-forcing.

Handles BOTH:
  - Normal samples  (from run_eval.py):      single-turn, one image group
  - Tool samples    (from run_eval_tool.py):  multi-turn, one image group per injected image

For multi-turn samples the full conversation (all turns) is reconstructed as one
sequence. Visual tokens form contiguous blocks — one per image injection — and
attention is tracked separately for each block:
  visual_turn0  = original image
  visual_turn1  = first re-examination (if tool was called)
  ...

This lets you plot: does attention to the re-examined image spike after the tool call?

Usage:
  # Normal run results
  python extract_attention.py --results results_normal/raw_results.jsonl \
      --out results_normal/attention_results.jsonl

  # Tool run results
  python extract_attention.py --results results_tool/tool_results.jsonl \
      --out results_tool/attention_results.jsonl

Requires sglang stopped first (need the GPU memory).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_PATH = "/capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it"

DIAGNOSE = False   # set via --diagnose; prints role/group structure per sample

# Must match the prompts used during inference
SYSTEM_PROMPT_NORMAL = (
    "You are a helpful visual question answering assistant. "
    "After your reasoning, answer ONLY with the single word 'Yes' or 'No'."
)
SYSTEM_PROMPT_TOOL = """You are a careful visual question answering assistant.

TOOL AVAILABLE:
If you need to re-examine the image before giving your final answer, respond with
EXACTLY one of:
  LOOK_AGAIN: full
  LOOK_AGAIN: top-left
  LOOK_AGAIN: top-right
  LOOK_AGAIN: bottom-left
  LOOK_AGAIN: bottom-right

You will immediately receive the requested image view and can then reason further.
Use this tool when visual details are unclear or when your reasoning depends on
a specific region you cannot verify from memory.

After all reasoning, answer ONLY with the single word Yes or No."""

REGION_BOXES = {
    "top-left":     (0.0, 0.0, 0.5, 0.5),
    "top-right":    (0.5, 0.0, 1.0, 0.5),
    "bottom-left":  (0.0, 0.5, 0.5, 1.0),
    "bottom-right": (0.5, 0.5, 1.0, 1.0),
    "top":          (0.0, 0.0, 1.0, 0.5),
    "bottom":       (0.0, 0.5, 1.0, 1.0),
    "left":         (0.0, 0.0, 0.5, 1.0),
    "right":        (0.5, 0.0, 1.0, 1.0),
    "full":         (0.0, 0.0, 1.0, 1.0),
}


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model_and_processor(model_path: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    print("Loading model (bf16, eager attn)...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",   # required — flash-attn doesn't return weights
    )
    model.eval()
    print(f"Model loaded. Devices: {set(model.hf_device_map.values())}")
    return model, processor


def get_image_token_id(processor) -> int:
    for attr in ("image_token_id", "image_token"):
        val = getattr(processor, attr, None)
        if val is not None:
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                tid = processor.tokenizer.convert_tokens_to_ids(val)
                if tid != processor.tokenizer.unk_token_id:
                    return tid
    for tok in ("<image_soft_token>", "<image>", "<img>", "█"):
        tid = processor.tokenizer.convert_tokens_to_ids(tok)
        if tid != processor.tokenizer.unk_token_id:
            return tid
    raise RuntimeError("Could not find image token ID.")


def get_token_id(tokenizer, token_str: str) -> int | None:
    """Robustly resolve a special token's id across tokenizer variants."""
    tid = tokenizer.convert_tokens_to_ids(token_str)
    if tid is not None and tid != tokenizer.unk_token_id:
        return tid
    # added-tokens table (special tokens often live here)
    enc = getattr(tokenizer, "added_tokens_encoder", None) or {}
    if token_str in enc:
        return enc[token_str]
    # added_tokens_decoder maps id -> AddedToken; scan it
    dec = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for tid, atok in dec.items():
        if str(atok) == token_str:
            return tid
    # last resort: encode literal, accept only if it maps to a single token
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None


# Start-of-turn delimiter candidates. This Gemma-4 uses '<|turn>' (id 105,
# the tokenizer's sot_token); older Gemma uses '<start_of_turn>'.
SOT_CANDIDATES = ("<|turn>", "<start_of_turn>")


def resolve_sot_id(tokenizer):
    """Resolve the start-of-turn token id, preferring the tokenizer's declared
    sot_token, then known string candidates."""
    cands = []
    sot_attr = getattr(tokenizer, "sot_token", None)
    if sot_attr:
        cands.append(sot_attr)
    cands.extend(SOT_CANDIDATES)
    for c in cands:
        tid = get_token_id(tokenizer, c)
        if tid is not None:
            return tid, c
    return None, None


def find_delim_positions(ids: list[int], tokenizer, delim: str | None = None):
    """
    Return (positions, token_id) of the start-of-turn delimiter in the sequence.
    Resolves the correct delimiter id (model-specific), then locates it. Falls
    back to decoding distinct token ids and matching any known candidate string.
    """
    if delim is not None:
        tid = get_token_id(tokenizer, delim)
        candidates = (delim,)
    else:
        tid, _ = resolve_sot_id(tokenizer)
        candidates = SOT_CANDIDATES
    if tid is not None:
        return [i for i, t in enumerate(ids) if t == tid], tid
    # decode-based fallback (cache per distinct id; sequence has few specials)
    cache: dict[int, str] = {}
    positions: list[int] = []
    found_id = None
    for i, t in enumerate(ids):
        s = cache.get(t)
        if s is None:
            s = tokenizer.decode([t])
            cache[t] = s
        if s.strip() in candidates:
            positions.append(i)
            found_id = t
    return positions, found_id


# ─── Image helpers ────────────────────────────────────────────────────────────

def open_crop(img_path: str, region: str = "full") -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    rx1, ry1, rx2, ry2 = REGION_BOXES.get(region.lower(), REGION_BOXES["full"])
    return img.crop((int(rx1*w), int(ry1*h), int(rx2*w), int(ry2*h)))


# ─── Message reconstruction ───────────────────────────────────────────────────

def is_tool_sample(sample: dict) -> bool:
    return "thinking_per_stage" in sample and sample.get("n_tool_calls", 0) > 0


def build_messages(sample: dict) -> list[dict]:
    """
    Reconstruct the full conversation as messages for teacher-forcing.
    For tool samples: includes all injected-image turns with <think> blocks.
    For normal samples: single [system, user, assistant] triple.
    """
    img_path = sample.get("image_path")

    def img_msg(path: str, region: str = "full") -> dict:
        return {"type": "image", "image": open_crop(path, region)}

    def text_msg(text: str) -> dict:
        return {"type": "text", "text": text}

    def user_content(text: str, path: str | None = None, region: str = "full") -> list | str:
        if path:
            return [img_msg(path, region), text_msg(text)]
        return text

    # ── Normal single-turn ────────────────────────────────────────────────────
    if not is_tool_sample(sample):
        thinking = (
            sample.get("thinking_content")
            or (sample.get("thinking_per_stage") or [""])[0]
        )
        answer = sample.get("answer_text", "")
        question = sample.get("question", "")
        return [
            {"role": "system", "content": SYSTEM_PROMPT_NORMAL},
            {"role": "user",   "content": user_content(question, img_path)},
            {"role": "assistant", "content":
                f"<think>\n{thinking}\n</think>\n{answer}" if thinking else answer},
        ]

    # ── Multi-turn tool sample ────────────────────────────────────────────────
    stages_info     = sample.get("stages", [])
    thinking_stages = sample.get("thinking_per_stage", [])
    tool_calls_data = sample.get("tool_calls", [])
    final_answer    = sample.get("answer_text", "")

    question_with_nudge = (
        sample.get("question", "")
        + "\n\nYou are encouraged to use LOOK_AGAIN: full during your reasoning "
          "if re-examining the image would help you answer with more confidence."
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT_TOOL},
        {"role": "user",   "content": user_content(question_with_nudge, img_path)},
    ]

    for turn_i, tc in enumerate(tool_calls_data):
        thinking_i  = thinking_stages[turn_i] if turn_i < len(thinking_stages) else ""
        answer_raw_i = (
            stages_info[turn_i].get("answer_raw", f"LOOK_AGAIN: {tc['region']}")
            if turn_i < len(stages_info) else f"LOOK_AGAIN: {tc['region']}"
        )
        # Assistant turn: thinking + tool-call answer
        messages.append({
            "role": "assistant",
            "content": (f"<think>\n{thinking_i}\n</think>\n{answer_raw_i}"
                        if thinking_i else answer_raw_i),
        })
        # User turn: injected image
        region = tc.get("region", "full")
        messages.append({
            "role": "user",
            "content": user_content(
                f"Here is the image view you requested ({region}). "
                "Continue your reasoning and give a final answer.",
                img_path,
                region,
            ),
        })

    # Final assistant turn
    last_thinking = thinking_stages[-1] if thinking_stages else ""
    messages.append({
        "role": "assistant",
        "content": (f"<think>\n{last_thinking}\n</think>\n{final_answer}"
                    if last_thinking else final_answer),
    })

    return messages


# ─── Token group identification ───────────────────────────────────────────────

def find_visual_groups(ids: list[int], image_token_id: int) -> list[list[int]]:
    """
    Find contiguous runs of image_token_id — one run per injected image.
    Returns list of position lists ordered by appearance in sequence.
    """
    groups: list[list[int]] = []
    cur: list[int] = []
    for i, t in enumerate(ids):
        if t == image_token_id:
            if not cur or i == cur[-1] + 1:
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        else:
            if cur:
                groups.append(cur)
                cur = []
    if cur:
        groups.append(cur)
    return groups


def identify_token_groups(
    ids: list[int],
    processor,
    image_token_id: int,
) -> dict[str, list[int]]:
    """
    Returns:
      visual_turn{N}   — position list for each image injection (N = 0, 1, ...)
      system           — system turn positions
      instruction      — user text positions (excluding image tokens)
      output           — all model-generation positions (all assistant turns)
    """
    tok = processor.tokenizer
    turn_starts, sot = find_delim_positions(ids, tok)
    sot_valid = len(turn_starts) > 0

    # All image token positions, grouped by contiguous run
    visual_groups = find_visual_groups(ids, image_token_id)
    all_visual    = set(pos for grp in visual_groups for pos in grp)

    result: dict[str, list[int]] = {}
    for gi, grp in enumerate(visual_groups):
        result[f"visual_turn{gi}"] = grp

    if sot_valid:

        # Determine each turn's ROLE by decoding the token(s) right after
        # We read the role token right after each start-of-turn delimiter.
        # This Gemma-4 has distinct system/user/model turns:
        #   <|turn>system\n ... <turn|><|turn>user\n ...<|turn>model\n ...
        def role_of(ts: int) -> str:
            snippet = tok.decode(ids[ts + 1: ts + 4]).strip().lower()
            if snippet.startswith("model"):
                return "model"
            if snippet.startswith("user"):
                return "user"
            if snippet.startswith("system"):
                return "system"
            return "unknown"

        roles = [role_of(ts) for ts in turn_starts]

        def turn_span(i: int) -> range:
            start = turn_starts[i]
            end   = turn_starts[i + 1] if i + 1 < len(turn_starts) else len(ids)
            return range(start, end)

        # system = leading tokens (BOS) + any explicit system-role turn, minus images
        system_range = set(range(0, turn_starts[0] if turn_starts else 0))
        for i, r in enumerate(roles):
            if r in ("system", "unknown"):
                system_range.update(turn_span(i))
        result["system"] = sorted(system_range - all_visual)

        # instruction = all USER-turn tokens minus image tokens
        user_range = set()
        for i, r in enumerate(roles):
            if r == "user":
                user_range.update(turn_span(i))
        result["instruction"] = sorted(user_range - all_visual)

        # output = all MODEL-turn tokens (the generations we score attention from)
        output_set = set()
        for i, r in enumerate(roles):
            if r == "model":
                output_set.update(turn_span(i))
        if not output_set:
            # Fallback: last 100 tokens
            result["output"] = list(range(max(0, len(ids) - 100), len(ids)))
        else:
            result["output"] = sorted(output_set)

        result["_roles"] = roles  # diagnostic

    else:
        # Heuristic: system = before first image, output = after last image
        first_vis = visual_groups[0][0]  if visual_groups else 0
        last_vis  = visual_groups[-1][-1] if visual_groups else 0
        result["system"]      = list(range(0, first_vis))
        result["instruction"] = []
        result["output"]      = list(range(last_vis + 1, len(ids)))

    return result


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

    messages = build_messages(sample)

    # Collect all PIL images in message order for the processor
    images_in_order: list[Image.Image] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image":
                    images_in_order.append(part["image"])
                    part["image"] = part["image"]  # keep reference

    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as e:
        print(f"  [processor error] {e}")
        return None

    input_ids = inputs["input_ids"]
    seq_len   = input_ids.shape[1]

    if seq_len > max_seq_len:
        print(f"  skip (seq_len={seq_len} > {max_seq_len})")
        return None

    ids    = input_ids[0].tolist()
    groups = identify_token_groups(ids, processor, image_token_id)

    output_pos = groups.get("output", [])
    n_output   = len(output_pos)
    if n_output == 0:
        print("  skip (no output positions)")
        return None

    if DIAGNOSE:
        vg = {k: len(v) for k, v in groups.items() if k.startswith("visual_turn")}
        _pos, sot_id = find_delim_positions(ids, processor.tokenizer)
        print(f"\n  [diagnose] seq_len={seq_len} sot_id={sot_id} n_delims={len(_pos)} "
              f"roles={groups.get('_roles')}")
        print(f"  [diagnose] visual_groups={vg} system={len(groups.get('system',[]))} "
              f"instruction={len(groups.get('instruction',[]))} output={n_output}")

    # Identify all visual groups (turn0, turn1, ...)
    visual_group_keys = sorted(k for k in groups if k.startswith("visual_turn"))
    all_named_groups  = {
        "system":      groups.get("system", []),
        "instruction": groups.get("instruction", []),
    }
    for vk in visual_group_keys:
        all_named_groups[vk] = groups[vk]

    # Clip to valid sequence positions
    group_arrays = {
        gname: np.array(sorted(set(pos) & set(range(seq_len))), dtype=int)
        for gname, pos in all_named_groups.items()
    }

    # Forward pass — capture attention per-layer via hooks to BOUND memory.
    # output_attentions=True makes each language self-attn module return weights;
    # a forward hook reduces them immediately on the layer's own GPU and returns
    # None for the weights, so the full [layers, heads, seq, seq] tensor is never
    # accumulated or gathered onto GPU 0 (which caused OOM on long sequences).
    inputs_dev = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
    out_idx    = np.array(output_pos, dtype=int)

    # Language-model decoder self-attn modules only (exclude vision/audio towers,
    # whose sequence dims differ from the language output positions).
    def _layer_idx(name: str) -> int:
        m = re.search(r"layers\.(\d+)\.", name)
        return int(m.group(1)) if m else -1

    attn_modules = [
        (name, mod) for name, mod in model.named_modules()
        if name.endswith(".self_attn") and "layers." in name
        and "vision" not in name.lower() and "audio" not in name.lower()
    ]
    attn_modules.sort(key=lambda x: _layer_idx(x[0]))
    n_layers = len(attn_modules)

    captured: dict[int, dict] = {}

    def make_hook(layer_idx: int):
        def hook(module, inp, out):
            if not isinstance(out, tuple) or len(out) < 2 or out[1] is None:
                return out
            w  = out[1]                          # [batch, heads, q, kv]
            wm = w[0].mean(dim=0)                # [q, kv] mean over heads (layer GPU)
            idx = torch.as_tensor(out_idx, device=wm.device)
            rows = wm.index_select(0, idx).float().cpu().numpy()  # [n_output, kv]
            captured[layer_idx] = {
                gname: (rows[:, cols].sum(axis=1) if cols.size
                        else np.zeros(len(out_idx), dtype=np.float32))
                for gname, cols in group_arrays.items()
            }
            # drop the weights so nothing large propagates / gathers to GPU 0
            return (out[0], None) + tuple(out[2:])
        return hook

    handles = [m.register_forward_hook(make_hook(i)) for i, (_, m) in enumerate(attn_modules)]
    try:
        model(**inputs_dev, output_attentions=True, return_dict=True)
    finally:
        for h in handles:
            h.remove()
    torch.cuda.empty_cache()

    # [n_layers, n_output] per group
    per_layer_per_pos = {
        g: np.zeros((n_layers, n_output), dtype=np.float32)
        for g in group_arrays
    }
    for li in range(n_layers):
        layer_res = captured.get(li)
        if layer_res is None:
            continue
        for gname in group_arrays:
            per_layer_per_pos[gname][li] = layer_res[gname]

    # ── Summarise ─────────────────────────────────────────────────────────────
    def thirds(arr: np.ndarray) -> dict:
        n = len(arr)
        if n < 3:
            v = float(arr.mean())
            return {"early": v, "mid": v, "late": v}
        t1, t2 = n // 3, 2 * n // 3
        return {"early": float(arr[:t1].mean()),
                "mid":   float(arr[t1:t2].mean()),
                "late":  float(arr[t2:].mean())}

    n_tool_calls = sample.get("n_tool_calls", 0)
    result: dict = {
        "sample_id":         sample["sample_id"],
        "category":          sample.get("category"),
        "subcategory":       sample.get("subcategory"),
        "visual_input":      sample.get("visual_input"),
        "is_correct":        sample.get("is_correct"),
        "n_tool_calls":      n_tool_calls,
        "total_image_tokens": sample.get("total_image_tokens"),

        # Sequence structure
        "seq_len":          seq_len,
        "n_layers":         n_layers,
        "n_output_tokens":  n_output,
        "n_visual_groups":  len(visual_group_keys),
        "n_visual_tokens_per_group": [
            len(groups.get(k, [])) for k in visual_group_keys
        ],
        "n_system_tokens":      len(groups.get("system", [])),
        "n_instruction_tokens": len(groups.get("instruction", [])),
    }

    # Per-group attention stats
    for gname, arr in per_layer_per_pos.items():
        result[f"attn_{gname}_mean"]        = float(arr.mean())
        result[f"attn_{gname}_per_pos"]     = arr.mean(axis=0).tolist()   # [n_output]
        result[f"attn_{gname}_per_layer"]   = arr.mean(axis=1).tolist()   # [n_layers]
        result[f"attn_{gname}_by_thirds"]   = thirds(arr.mean(axis=0))

    # Convenience: combined visual (all groups summed) for backwards-compat plots
    if visual_group_keys:
        combined_visual = sum(
            per_layer_per_pos[k] for k in visual_group_keys
        )
        result["attn_visual_mean"]        = float(combined_visual.mean())
        result["attn_visual_per_pos"]     = combined_visual.mean(axis=0).tolist()
        result["attn_visual_per_layer"]   = combined_visual.mean(axis=1).tolist()
        result["attn_visual_by_thirds"]   = thirds(combined_visual.mean(axis=0))

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",     default=None,
                    help="Path to raw_results.jsonl or tool_results.jsonl")
    ap.add_argument("--model",       default=MODEL_PATH)
    ap.add_argument("--out",         default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--diagnose", action="store_true",
                    help="Print per-sample role/group structure for verification")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Append mode: skip sample_ids already in --out, process "
                         "only the missing ones (e.g. samples OOM-skipped earlier).")
    args = ap.parse_args()

    global DIAGNOSE
    DIAGNOSE = args.diagnose

    script_dir   = Path(__file__).parent
    results_path = (Path(args.results) if args.results
                    else script_dir / "results" / "raw_results.jsonl")
    out_path     = (Path(args.out) if args.out
                    else results_path.parent / "attention_results.jsonl")

    samples = []
    with open(results_path) as f:
        for line in f:
            rec = json.loads(line)
            if "error" not in rec and rec.get("answer_text") is not None:
                samples.append(rec)

    # Resume: skip sample_ids already present in the output file
    done_ids: set = set()
    if args.skip_existing and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
        before = len(samples)
        samples = [s for s in samples if s["sample_id"] not in done_ids]
        print(f"--skip-existing: {len(done_ids)} already done, "
              f"{len(samples)} remaining (of {before})")

    if args.max_samples:
        samples = samples[:args.max_samples]

    n_tool = sum(1 for s in samples if is_tool_sample(s))
    print(f"Loaded {len(samples)} samples ({n_tool} tool-call samples)")
    print(f"Output: {out_path}\n")

    model, processor = load_model_and_processor(args.model)
    image_token_id   = get_image_token_id(processor)
    print(f"Image token ID: {image_token_id}\n")

    out_path.parent.mkdir(exist_ok=True)
    n_done = n_skip = 0

    write_mode = "a" if args.skip_existing else "w"
    with open(out_path, write_mode) as fout:
        for i, sample in enumerate(samples):
            tag = (f"[{i+1:3d}/{len(samples)}] {sample['sample_id']}"
                   f"  tool={'y' if is_tool_sample(sample) else 'n'}")
            print(tag, end="  ", flush=True)
            try:
                result = extract_attention(
                    model, processor, sample, image_token_id, args.max_seq_len
                )
                if result:
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
                    n_done += 1
                    vis_groups = result.get("n_visual_groups", 1)
                    print(
                        f"vis_groups={vis_groups} "
                        f"attn_vis={result.get('attn_visual_mean', 0):.4f} "
                        f"attn_ins={result.get('attn_instruction_mean', 0):.4f}"
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
