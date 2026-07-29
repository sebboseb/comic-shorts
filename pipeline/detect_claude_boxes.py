"""Claude-vision panel detector.

Sends the full page to the vision model for panel boxes + reading order,
then snaps each box edge to the nearest clean gutter line locally using
the same paper-relative binarization as the flood-fill detector. The
model handles layout semantics (staggered grids, diagonal panels, art
overflowing borders — where classical CV fails); the snap fixes its
box imprecision against the actual pixels.

Costs cents per book (models.vision_model, one call per page).
Requires an Anthropic API key; stage 1 falls back to detect_floodfill
when the call fails.
"""

import base64
import io
import json

import anthropic
import cv2
import numpy as np
from PIL import Image

from pipeline.stage1_panels import adaptive_white_threshold

MAX_SEND_PX = 1400   # long side sent to the vision model (stage-2 convention)
SNAP_WINDOW = 0.015  # search +/- this fraction of page size around each edge
SNAP_MAX_INK = 0.06  # a gutter line has at most this much ink along the edge

# $/MTok input, output — for the per-page cost report
PRICING = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
}

SYSTEM_PROMPT = """You detect comic panels for an automated video pipeline.
You are given one full comic page. Respond with ONLY a JSON object, no
markdown fences, no commentary:

{
  "panels": [
    {"box": [x, y, w, h]},
    ...
  ]
}

Rules:
- Boxes use coordinates normalized to 0-1000 on both axes of the page image.
- List panels in reading order (left-to-right, top-to-bottom rows, unless
  the layout clearly dictates otherwise).
- One box per story panel. Include the panel border in the box. Exclude the
  page margins and the page header/title strip.
- A caption or narration strip attached to a panel belongs to that panel's
  box; a free-standing caption strip is its own panel.
- Art often overflows panel borders in old comics — box the intended panel
  rectangle, not the overflow.
- Diagonal or overlapping panels: box each panel's full extent even if the
  boxes overlap.
- A single full-page splash is one box."""


def _b64_page(img, max_px=MAX_SEND_PX):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if max(pil.size) > max_px:
        scale = max_px / max(pil.size)
        pil = pil.resize((int(pil.width * scale), int(pil.height * scale)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def _snap_edges(box, content, w, h):
    """Move each box edge to the cleanest nearby line if that line reads
    as gutter (near-zero ink along the edge's span)."""
    x0, y0, x1, y1 = box
    win_x = max(4, int(SNAP_WINDOW * w))
    win_y = max(4, int(SNAP_WINDOW * h))

    def best_line(profile, pos, win, lo, hi):
        a = max(lo, pos - win)
        b = min(hi, pos + win + 1)
        if b <= a:
            return pos
        segment = profile[a:b]
        idx = int(np.argmin(segment))
        return a + idx if segment[idx] <= SNAP_MAX_INK else pos

    # ink fraction per row/col measured across the box's other axis
    cols = content[max(0, y0):y1, :].mean(axis=0)
    rows = content[:, max(0, x0):x1].mean(axis=1)
    x0 = best_line(cols, x0, win_x, 0, w)
    x1 = best_line(cols, x1, win_x, 0, w)
    y0 = best_line(rows, y0, win_y, 0, h)
    y1 = best_line(rows, y1, win_y, 0, h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return box
    return (x0, y0, x1, y1)


def detect_panels(img, model, min_area_ratio=0.02):
    """Return (boxes, usage): boxes as (x, y, w, h) in reading order,
    usage as {input_tokens, output_tokens, cost_usd}. Raises on API
    failure — the caller decides on fallback."""
    h, w = img.shape[:2]
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg",
                            "data": _b64_page(img)}},
                {"type": "text", "text": "Detect the panels on this page."},
            ],
        }],
    )
    text = next(b.text for b in response.content if b.type == "text")
    data = _parse_json(text)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    content = (gray < adaptive_white_threshold(gray)).astype(np.uint8)
    content = cv2.morphologyEx(content, cv2.MORPH_OPEN,
                               np.ones((3, 3), np.uint8))

    min_area = min_area_ratio * w * h
    boxes = []
    for p in data.get("panels", []):
        x, y, bw, bh = p["box"]
        px = (round(x / 1000 * w), round(y / 1000 * h),
              round((x + bw) / 1000 * w), round((y + bh) / 1000 * h))
        px = _snap_edges(px, content, w, h)
        x0, y0, x1, y1 = px
        if (x1 - x0) * (y1 - y0) >= min_area:
            boxes.append((x0, y0, x1 - x0, y1 - y0))

    rate_in, rate_out = PRICING.get(model, (3.00, 15.00))
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": (response.usage.input_tokens * rate_in
                     + response.usage.output_tokens * rate_out) / 1e6,
    }
    return boxes, usage
