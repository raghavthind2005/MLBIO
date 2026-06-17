#!/usr/bin/env python3
"""
Pre-flight verification for the BabyVision standard run on Clariden.

Confirms — BEFORE committing the full 388×3 run — that sglang/Gemma-4 actually
returns the data we plan to capture. Runs on 2 real BabyVision samples (1 blank,
1 choice) against a running sglang server.

CHECKS
  1. reasoning_content is returned separately (thinking trace captured).
  2. logprobs + top_logprobs are returned.
  3. *** THE CRITICAL ONE *** do logprobs cover the REASONING tokens, or only the
     visible answer? Compares len(logprobs.content) against completion_tokens.
       len(logprobs) ≈ completion_tokens  -> covers reasoning  ✅
       len(logprobs) ≈ a few tokens       -> answer-only       ❌ (use fallback)
  4. image token count = prompt_tokens(with image) - prompt_tokens(without image).
  5. finish_reason (truncation detection works).

FALLBACK if check 3 fails: relaunch sglang WITHOUT --reasoning-parser so the full
<think>...</think> lands in `content` with full logprobs, and parse thinking manually.

Usage (inside the sbatch, after server is healthy):
    python verify_capture.py --port 30000 --data-dir <.../babyvision_data>
"""

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def encode_image(path: Path) -> str:
    """Keep original format (no JPEG re-encode — fidelity contract)."""
    img = Image.open(path)
    fmt = (img.format or "PNG").upper()
    mime = "jpeg" if fmt in ("JPG", "JPEG") else "png"
    save_fmt = "JPEG" if mime == "jpeg" else "PNG"
    buf = BytesIO()
    img.save(buf, format=save_fmt)
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def build_question(item: dict) -> tuple[str, str]:
    suffix = "\nThink about the question and give your final answer in \\boxed{Answer} format."
    if item["ansType"] == "blank":
        return item["question"] + suffix, item["blankAns"]
    opts = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(item["options"]))
    return item["question"] + "\nChoices:\n" + opts + suffix, chr(65 + int(item["choiceAns"]))


def call(url, model, question, img_uri, max_tokens, with_logprobs=True):
    content = []
    if img_uri:
        content.append({"type": "image_url", "image_url": {"url": img_uri}})
    content.append({"type": "text", "text": question})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "chat_template_kwargs": {"enable_thinking": True},
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_k": 64,
        "top_p": 0.95,
    }
    if with_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 20
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=900)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--data-dir", required=True, help="dir containing meta_data.jsonl + images/")
    ap.add_argument("--max-tokens", type=int, default=4096, help="small for a quick smoke test")
    args = ap.parse_args()

    base = f"http://localhost:{args.port}"
    model = requests.get(f"{base}/v1/models", timeout=10).json()["data"][0]["id"]
    print(f"Model: {model}\n")

    data_dir = Path(args.data_dir)
    rows = [json.loads(l) for l in open(data_dir / "meta_data.jsonl") if l.strip()]
    blank = next(r for r in rows if r["ansType"] == "blank")
    choice = next(r for r in rows if r["ansType"] == "choice")

    for label, item in [("BLANK", blank), ("CHOICE", choice)]:
        print("=" * 78)
        print(f"[{label}] taskId={item['taskId']}  {item['type']} / {item['subtype']}")
        q, gt = build_question(item)
        img_uri = encode_image(data_dir / item["image"])

        resp = call(base, model, q, img_uri, args.max_tokens)
        choice0 = resp["choices"][0]
        msg = choice0["message"]
        usage = resp.get("usage", {})

        reasoning = (msg.get("reasoning_content") or "")
        answer = (msg.get("content") or "")
        comp_tok = usage.get("completion_tokens")
        lp = choice0.get("logprobs") or {}
        lp_content = lp.get("content") or []
        n_lp = len(lp_content)
        has_top = bool(lp_content and lp_content[0].get("top_logprobs"))
        k_top = len(lp_content[0]["top_logprobs"]) if has_top else 0

        print(f"  reasoning_content present : {bool(reasoning)}  ({len(reasoning)} chars)")
        print(f"  content (answer) present  : {bool(answer)}  -> {answer[:80]!r}")
        print(f"  ground truth              : {gt!r}")
        print(f"  finish_reason             : {choice0.get('finish_reason')}")
        print(f"  completion_tokens (usage) : {comp_tok}")
        print(f"  prompt_tokens (usage)     : {usage.get('prompt_tokens')}")
        print(f"  logprobs returned         : {bool(lp_content)}  (n_entries={n_lp})")
        print(f"  top_logprobs              : {has_top}  (k={k_top})")

        # CRITICAL verdict: do logprobs include reasoning?
        if n_lp and comp_tok:
            ratio = n_lp / comp_tok
            if ratio > 0.9:
                verdict = "✅ logprobs COVER reasoning (n_logprobs ~= completion_tokens)"
            elif n_lp <= len(answer.split()) + 10:
                verdict = "❌ logprobs are ANSWER-ONLY -> use no-reasoning-parser FALLBACK"
            else:
                verdict = f"⚠️ partial coverage (ratio={ratio:.2f}) — inspect manually"
            print(f"  >>> LOGPROB COVERAGE: {verdict}")

        # image token count via prompt_tokens diff (no image)
        resp_noimg = call(base, model, q, None, 16, with_logprobs=False)
        p_with = usage.get("prompt_tokens")
        p_without = resp_noimg.get("usage", {}).get("prompt_tokens")
        if p_with and p_without:
            print(f"  >>> IMAGE TOKENS (approx): {p_with - p_without}  "
                  f"(with={p_with}, without={p_without})")
        print()

    print("=" * 78)
    print("If LOGPROB COVERAGE shows ✅ on both -> proceed to full run_eval.")
    print("If ❌ -> relaunch sglang WITHOUT --reasoning-parser and re-run this check.")


if __name__ == "__main__":
    main()
