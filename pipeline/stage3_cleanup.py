"""Stage 3: art cleanup.

For each panel with dialogue: refine the LLM's approximate bubble boxes
into precise pixel masks (bubbles are near-white blobs), inpaint them out
with LaMa, then upscale everything with Real-ESRGAN so zooms stay sharp
on a 1080x1920 canvas.

Runs locally: LaMa weights auto-download on first run, ESRGAN weights are
a manual one-time download (see README).

Output: work/clean/pPP_NN.png + work/clean.json
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def refine_bubble_mask(img_gray, bbox, pad=10):
    """Inside a padded bbox, find the near-white bubble blob precisely."""
    h, w = img_gray.shape
    x, y, bw, bh = bbox
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    roi = img_gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    _, white = cv2.threshold(roi, 225, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    if n <= 1:
        # nothing near-white found: fall back to the whole box
        mask = np.zeros((h, w), np.uint8)
        mask[y0:y1, x0:x1] = 255
        return mask

    # largest white component in the ROI = the bubble
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (labels == largest).astype(np.uint8) * 255
    blob = cv2.dilate(blob, np.ones((9, 9), np.uint8), iterations=1)

    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = blob
    return mask


def upscale(img_pil, model, device, target_long_side):
    long_side = max(img_pil.size)
    if long_side >= target_long_side:
        return img_pil

    arr = np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    result = Image.fromarray((out * 255).astype(np.uint8))

    # ESRGAN is fixed 4x; downscale if that overshoots the target a lot
    if max(result.size) > target_long_side * 1.5:
        scale = target_long_side / max(result.size)
        result = result.resize((int(result.width * scale),
                                int(result.height * scale)), Image.LANCZOS)
    return result


def run(config, workdir: Path):
    from simple_lama_inpainting import SimpleLama
    from spandrel import ModelLoader

    device = _device()
    print(f"device: {device}")

    lama = SimpleLama(device=torch.device(device))

    weights = Path(config["models"]["esrgan_weights"])
    if not weights.exists():
        raise SystemExit(f"ESRGAN weights not found at {weights} - see README")
    esrgan = ModelLoader().load_from_file(weights).model.eval().to(device)
    target = config["models"].get("upscale_target_long_side", 2200)

    panels = json.loads((workdir / "understanding.json").read_text())
    out_dir = workdir / "clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, panel in enumerate(panels, start=1):
        src = workdir / panel["file"]
        img = Image.open(src).convert("RGB")
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

        boxes = [d["bubble_box"] for d in panel.get("dialogue", [])
                 if "bubble_box" in d]
        full_mask = np.zeros(gray.shape, np.uint8)
        for bbox in boxes:
            m = refine_bubble_mask(gray, bbox)
            if m is not None:
                full_mask = cv2.bitwise_or(full_mask, m)

        if full_mask.any():
            img = lama(img, Image.fromarray(full_mask))

        old_size = img.size
        img = upscale(img, esrgan, device, target)

        out_name = f"{panel['id']}.png"
        img.save(out_dir / out_name)
        results.append({**panel, "clean_file": f"clean/{out_name}",
                        "clean_size": list(img.size)})
        print(f"[{i}/{len(panels)}] {panel['id']}: "
              f"{len(boxes)} bubbles removed, {old_size} -> {img.size}")

    (workdir / "clean.json").write_text(json.dumps(results, indent=2))
    print(f"\nDone -> {out_dir}. Skim the images: bubbles overlapping faces "
          "are the main inpaint failure mode, touch those up by hand.")
