#!/usr/bin/env python3
"""
LOCAL (no-GPU, no model weights) validation of the turn-delimiter + role
detection used by extract_attention.py.

This is the piece that was failing on the cluster (<start_of_turn> resolving to
None). It runs on pure tokenizer logic, so we can verify the fix on a laptop
using ONLY the Gemma-4 processor/tokenizer files — no 31B weights, no GPU.

Setup (run on laptop; cluster LOGIN nodes are up even during compute maintenance):
  # 1. pull tokenizer/processor files only (excludes the multi-GB weights)
  rsync -avz --progress \
    --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' --exclude='*.gguf' \
    clariden:/capstor/store/cscs/swissai/a0174/models/gemma-4-31B-it/ \
    ~/gemma4_files/
  # 2. install deps (CPU only is fine)
  pip install 'transformers>=5.10' torch pillow numpy
  # 3. run
  python image_toolCalling/validate_tokens_local.py --model-dir ~/gemma4_files

Success = delimiter found, roles=['user','model','user','model'], 2 visual
groups, instruction>0. Then Sunday's cluster run is one-shot.
"""

import argparse

from PIL import Image
from transformers import AutoProcessor

# reuse the REAL functions from the extraction pipeline (no duplication)
from extract_attention import (
    get_image_token_id,
    find_delim_positions,
    identify_token_groups,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="Local dir with the rsynced Gemma-4 tokenizer/processor files")
    args = ap.parse_args()

    print(f"Loading processor from {args.model_dir} (no model weights)...")
    proc = AutoProcessor.from_pretrained(args.model_dir)
    tok  = proc.tokenizer

    img_id = get_image_token_id(proc)
    print(f"tokenizer class       : {type(tok).__name__}")
    print(f"image_token_id        : {img_id}")
    print(f"convert <start_of_turn>: {tok.convert_tokens_to_ids('<start_of_turn>')}  "
          f"(unk={tok.unk_token_id})")

    # Synthetic forced-style conversation: system folded into 1st user turn,
    # TWO image injections, TWO model turns. Dummy image — we only test the
    # token structure, not the pixels.
    dummy = Image.new("RGB", (224, 224), (127, 127, 127))
    messages = [
        {"role": "system",
         "content": "You are a helpful visual question answering assistant. "
                    "After your reasoning, answer ONLY with 'Yes' or 'No'."},
        {"role": "user", "content": [
            {"type": "image", "image": dummy},
            {"type": "text",  "text": "Is the value of x larger than 6?"}]},
        {"role": "assistant",
         "content": "<think>\nThe triangle has legs 8 and x.\n</think>\nNo"},
        {"role": "user", "content": [
            {"type": "image", "image": dummy},
            {"type": "text",  "text": "Here is the image again. Re-examine it. "
                                      "Answer ONLY with 'Yes' or 'No'."}]},
        {"role": "assistant",
         "content": "<think>\nLooking again, still no labels visible.\n</think>\nNo"},
    ]

    enc = proc.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True, return_dict=True,
    )
    ids = enc["input_ids"]
    # normalize to a flat python list
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    print(f"\nseq_len               : {len(ids)}")

    pos, sot = find_delim_positions(ids, tok, "<start_of_turn>")
    print(f"delimiter positions   : {pos}")
    print(f"resolved sot id       : {sot}")

    groups = identify_token_groups(ids, proc, img_id)
    roles = groups.get("_roles")
    vg = {k: len(v) for k, v in groups.items() if k.startswith("visual_turn")}
    print(f"roles                 : {roles}")
    print(f"visual_groups         : {vg}")
    print(f"system / instruction / output : "
          f"{len(groups.get('system', []))} / "
          f"{len(groups.get('instruction', []))} / "
          f"{len(groups.get('output', []))}")

    # ── Assertions ────────────────────────────────────────────────────────────
    ok = True
    if sot is None:
        print("\n[FAIL] delimiter not resolved"); ok = False
    if roles != ["user", "model", "user", "model"]:
        print(f"\n[FAIL] roles wrong: {roles}"); ok = False
    if len(vg) != 2:
        print(f"\n[FAIL] expected 2 visual groups, got {vg}"); ok = False
    if len(groups.get("instruction", [])) == 0:
        print("\n[FAIL] instruction tokens empty"); ok = False

    if ok:
        print("\n✓ ALL CHECKS PASSED — delimiter/role detection works locally.")
        print("  The Sunday cluster run is now a one-shot full extraction.")
    else:
        print("\n✗ Some checks failed — paste this output and we fix the logic "
              "now (no cluster needed).")


if __name__ == "__main__":
    main()
