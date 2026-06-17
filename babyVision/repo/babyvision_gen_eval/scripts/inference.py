#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BabyVision Image Generation Inference Script

Generate annotated images for BabyVision tasks using image generation models via OpenRouter API.

Features:
- Multi-round generation with different seeds
- Checkpoint resume: automatically skips already generated images
- Outputs JSONL file with generated image paths
- Configurable via command-line arguments and environment variables
"""

import os
import sys
import time
import json
import argparse
import base64
import requests
import multiprocessing as mp
from io import BytesIO
from typing import List, Dict, Optional

from PIL import Image
from tqdm import tqdm

# ================== Default Configuration ==================
DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3-pro-image-preview"
DEFAULT_TARGET_SIZE = 1024
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5.0
DEFAULT_REQUEST_DELAY = 1.0

# ================== System Prompt ==================
SYSTEM_PROMPT = """CRITICAL INSTRUCTION: You are a visual annotation assistant. Your task is to add ONLY the requested annotations (circles, lines, arrows, text labels) to mark the answer on the image.

IMPORTANT RULES:
1. DO NOT modify, redraw, or alter ANY part of the original image content
2. DO NOT change colors, shapes, positions, or any visual elements of the original image
3. ONLY add overlay annotations (circles, lines, arrows, text) on TOP of the original image
4. The original image must remain 100% intact and unchanged
5. Use bright, visible colors (red, green, blue) for your annotations so they stand out
6. Keep annotations minimal and precise - only mark what is asked"""


def build_prompt(task_data: Dict) -> str:
    """Build complete prompt from task data."""
    user_prompt = task_data.get("generationPrompt", "") 

    parts = [SYSTEM_PROMPT]
    parts.append(f"\n\nYOUR TASK:\n{user_prompt}")
    parts.append("\n\nREMINDER: Keep the original image EXACTLY as it is. ONLY add annotation marks (circles, lines, arrows, or text) to indicate your answer. Do not redraw or modify any part of the original image.")

    return "\n".join(parts)


def resize_and_pad_image(image_path: str, target_size: int) -> Optional[Image.Image]:
    """Read image, scale proportionally, and pad with white to target resolution."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    ow, oh = img.size
    ratio = min(target_size / ow, target_size / oh)
    nw, nh = int(ow * ratio), int(oh * ratio)
    img_resized = img.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    px, py = (target_size - nw) // 2, (target_size - nh) // 2
    canvas.paste(img_resized, (px, py))
    return canvas


def pil_to_base64(pil_image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def is_valid_image(path: str, min_size: int = 1000) -> bool:
    """Check if file is a valid image."""
    if not os.path.exists(path) or os.path.getsize(path) < min_size:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def call_api(api_key: str, api_url: str, model: str, image_b64: str, prompt: str) -> Optional[str]:
    """Call OpenRouter API for image generation. Returns base64 encoded image or None."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://babyvision-eval.local",
        "X-Title": "BabyVision Generation Evaluation"
    }

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": prompt}
            ]
        }],
        "modalities": ["image", "text"],
        "max_tokens": 4096,
        "temperature": 0.7,
        "image_config": {"aspect_ratio": "1:1", "image_size": "1K"}
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=180)

    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text[:500]}")

    result = response.json()

    # Extract image from response (handles multiple response formats)
    if "choices" in result and len(result["choices"]) > 0:
        message = result["choices"][0].get("message", {})

        # Check for images array
        if "images" in message and len(message["images"]) > 0:
            img_data = message["images"][0]
            if isinstance(img_data, dict):
                img_url = img_data.get("image_url", {}).get("url", "")
                if img_url.startswith("data:image"):
                    parts = img_url.split(",", 1)
                    if len(parts) == 2:
                        return parts[1]

        # Check for content with images
        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    img_url = item.get("image_url", {}).get("url", "")
                    if img_url.startswith("data:image"):
                        parts = img_url.split(",", 1)
                        if len(parts) == 2:
                            return parts[1]
        elif isinstance(content, str) and "data:image" in content:
            import re
            match = re.search(r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)', content)
            if match:
                return match.group(1)

    return None


def worker_process(task: Dict) -> Dict:
    """Process a single task (used by multiprocessing pool)."""
    image_path = task["image_path"]
    prompt = task["prompt"]
    save_path = task["save_path"]
    api_key = task["api_key"]
    api_url = task["api_url"]
    model = task["model"]
    max_retries = task.get("max_retries", DEFAULT_MAX_RETRIES)
    retry_delay = task.get("retry_delay", DEFAULT_RETRY_DELAY)
    request_delay = task.get("request_delay", DEFAULT_REQUEST_DELAY)
    task_data = task["task_data"]

    start_t = time.time()
    retries = 0

    # Skip if already exists
    if is_valid_image(save_path):
        return {
            "success": True, "duration": time.time() - start_t, "error": "",
            "save_path": save_path, "skipped": True, "retries": 0, "task_data": task_data
        }

    # Preprocess image
    pil_img = resize_and_pad_image(image_path, task.get("target_size", DEFAULT_TARGET_SIZE))
    if pil_img is None:
        return {
            "success": False, "duration": time.time() - start_t,
            "error": f"Failed to open/resize image: {image_path}",
            "save_path": save_path, "skipped": False, "retries": 0, "task_data": task_data
        }

    image_b64 = pil_to_base64(pil_img)

    # API call with retry
    err_msg = ""
    while retries < max_retries:
        try:
            if request_delay > 0:
                time.sleep(request_delay)

            b64_data = call_api(api_key, api_url, model, image_b64, prompt)

            if not b64_data:
                err_msg = "No image data in response"
                retries += 1
                time.sleep(retry_delay)
                continue

            img_bytes = base64.b64decode(b64_data)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            tmp_path = save_path + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(img_bytes)

            if is_valid_image(tmp_path):
                os.rename(tmp_path, save_path)
                return {
                    "success": True, "duration": time.time() - start_t, "error": "",
                    "save_path": save_path, "skipped": False, "retries": retries, "task_data": task_data
                }
            else:
                os.remove(tmp_path)
                err_msg = "Generated image is invalid"
                retries += 1
                time.sleep(retry_delay)

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            retries += 1
            if retries < max_retries:
                time.sleep(retry_delay)

    return {
        "success": False, "duration": time.time() - start_t, "error": err_msg,
        "save_path": save_path, "skipped": False, "retries": retries, "task_data": task_data
    }


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    """Save JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def generate_output_jsonl(data: List[Dict], images_dir: str, round_dir: str):
    """Generate output JSONL with generated image paths."""
    output_data = []
    for item in data:
        new_item = item.copy()
        task_id = item.get("taskId", "unknown")
        image_rel = item.get("image", "")

        if image_rel:
            name = os.path.splitext(os.path.basename(image_rel))[0]
            save_name = f"{name}_task{task_id}.png"
            save_path = os.path.join(images_dir, save_name)

            if is_valid_image(save_path):
                new_item["generated_image"] = os.path.relpath(save_path, round_dir)
            else:
                new_item["generated_image"] = None

        output_data.append(new_item)

    output_path = os.path.join(round_dir, "results.jsonl")
    save_jsonl(output_data, output_path)
    print(f"[INFO] Results saved to: {output_path}")


def run_inference(
    data_root: str,
    jsonl_path: str,
    output_root: str,
    rounds: int,
    base_seed: int,
    api_key: str,
    api_url: str,
    model: str,
    num_workers: int,
    request_delay: float,
    target_size: int,
    max_retries: int,
    retry_delay: float,
):
    """Main inference function."""
    data = load_jsonl(jsonl_path)
    print(f"[INFO] Loaded {len(data)} tasks from {jsonl_path}")

    if not data:
        print("[ERROR] No data found in JSONL file.")
        return

    os.makedirs(output_root, exist_ok=True)

    print(f"[INFO] Configuration:")
    print(f"  - API URL: {api_url}")
    print(f"  - Model: {model}")
    print(f"  - Workers: {num_workers}")
    print(f"  - Request Delay: {request_delay}s")
    print(f"  - Target Size: {target_size}x{target_size}")

    for round_idx in range(1, rounds + 1):
        current_seed = base_seed + round_idx - 1
        round_dir = os.path.join(output_root, f"round{round_idx}")
        images_dir = os.path.join(round_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"  Round {round_idx}/{rounds} (seed={current_seed})")
        print(f"{'='*50}")

        # Build task list
        tasks = []
        skipped_count = 0

        for item in data:
            task_id = item.get("taskId", "unknown")
            image_rel = item.get("image", "")
            raw_prompt = item.get("generationPrompt", "")

            if not image_rel or not raw_prompt:
                print(f"[WARN] Task {task_id} missing image or prompt, skipping.")
                continue

            prompt = build_prompt(item)
            image_path = os.path.join(data_root, image_rel)

            if not os.path.exists(image_path):
                print(f"[WARN] Image not found: {image_path}, skipping.")
                continue

            name = os.path.splitext(os.path.basename(image_rel))[0]
            save_path = os.path.join(images_dir, f"{name}_task{task_id}.png")

            if is_valid_image(save_path):
                skipped_count += 1
                continue

            tasks.append({
                "image_path": image_path,
                "prompt": prompt,
                "save_path": save_path,
                "api_key": api_key,
                "api_url": api_url,
                "model": model,
                "max_retries": max_retries,
                "retry_delay": retry_delay,
                "request_delay": request_delay,
                "target_size": target_size,
                "task_data": item,
            })

        total_tasks = len(data)
        print(f"[INFO] Tasks: {len(tasks)} pending, {skipped_count} skipped (already completed)")

        if not tasks:
            print(f"[INFO] All tasks for round {round_idx} already completed.")
            generate_output_jsonl(data, images_dir, round_dir)
            continue

        workers = min(num_workers, len(tasks))
        print(f"[INFO] Starting {workers} workers...")

        success_count = skipped_count
        fail_count = 0
        pbar = tqdm(total=total_tasks, initial=skipped_count, desc=f"Round {round_idx}", unit="img")

        overall_start = time.time()
        with mp.Pool(processes=workers) as pool:
            try:
                for result in pool.imap_unordered(worker_process, tasks):
                    if result.get("success"):
                        success_count += 1
                    else:
                        fail_count += 1
                        if result.get("error"):
                            tqdm.write(f"[FAIL] task {result['task_data'].get('taskId')}: {result['error']}")

                    duration = result.get("duration", 0.0)
                    done = pbar.n + 1
                    elapsed = time.time() - overall_start
                    avg = elapsed / done if done > 0 else 0.0
                    eta = avg * (total_tasks - done)

                    pbar.set_postfix(ok=success_count, fail=fail_count, last=f"{duration:.1f}s", eta=f"{eta/60:.1f}m")
                    pbar.update(1)

            except KeyboardInterrupt:
                print("\n[INFO] Interrupted, terminating workers...")
                pool.terminate()
                pool.join()

        pbar.close()
        generate_output_jsonl(data, images_dir, round_dir)

        total_time = time.time() - overall_start
        print(f"[INFO] Round {round_idx} completed: {success_count}/{total_tasks} success, {fail_count} failed")
        print(f"[INFO] Time: {total_time:.1f}s ({total_time/60:.2f} min)")


def main():
    parser = argparse.ArgumentParser(
        description="BabyVision Image Generation Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python inference.py --data-root ./data --jsonl ./data/meta_data.jsonl --output ./generated/model_name

  # Multi-round generation
  python inference.py --data-root ./data --jsonl ./data/meta_data.jsonl --output ./generated/model_name --rounds 3

Environment Variables:
  OPENROUTER_API_KEY    API key for OpenRouter (required if not using --api-key)
  OPENROUTER_API_URL    API URL (default: https://openrouter.ai/api/v1/chat/completions)
  MODEL                 Model name (default: google/gemini-3-pro-image-preview)
"""
    )

    parser.add_argument("--data-root", type=str, required=True, help="Data root folder containing images/")
    parser.add_argument("--jsonl", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output folder for generated images")
    parser.add_argument("--rounds", type=int, default=1, help="Number of generation rounds (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--api-url", type=str, default=None, help="OpenRouter API URL")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY, help="Delay between requests (default: 1.0s)")
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE, help="Target image size (default: 1024)")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Max retries per task (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY, help="Delay between retries (default: 5.0s)")

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] API key required. Use --api-key or set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    # Get API URL and model from args or environment
    api_url = args.api_url or os.environ.get("OPENROUTER_API_URL", DEFAULT_API_URL)
    model = args.model or os.environ.get("MODEL", DEFAULT_MODEL)

    run_inference(
        data_root=args.data_root,
        jsonl_path=args.jsonl,
        output_root=args.output,
        rounds=args.rounds,
        base_seed=args.seed,
        api_key=api_key,
        api_url=api_url,
        model=model,
        num_workers=args.num_workers,
        request_delay=args.request_delay,
        target_size=args.target_size,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )


if __name__ == "__main__":
    main()
