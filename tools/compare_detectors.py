#!/usr/bin/env python3
"""Compare panel detectors on the pages in config's pages_dir.

Runs claude_boxes and floodfill on every page, writes numbered overlay
images to work/detector_compare/{claude,floodfill}/, and prints a
side-by-side table with per-page API cost.

Usage:  .venv/bin/python tools/compare_detectors.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import yaml

from pipeline import detect_claude_boxes, detect_floodfill
from pipeline.stage1_panels import reading_order


def overlay(img, boxes, out_path):
    vis = img.copy()
    for k, (x, y, w, h) in enumerate(boxes, 1):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 12)
        cv2.putText(vis, str(k), (x + 30, y + 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 20)
    h, w = vis.shape[:2]
    cv2.imwrite(str(out_path), cv2.resize(vis, (w // 3, h // 3)))


def main():
    config = yaml.safe_load(open("config/comic.yaml"))
    pages_dir = Path(config["comic"]["pages_dir"])
    model = config.get("models", {}).get("vision_model", "claude-sonnet-4-6")

    out_claude = Path("work/detector_compare/claude")
    out_flood = Path("work/detector_compare/floodfill")
    out_claude.mkdir(parents=True, exist_ok=True)
    out_flood.mkdir(parents=True, exist_ok=True)

    pages = sorted(p for p in pages_dir.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))

    print(f"{'page':<10} {'claude':>7} {'flood':>6} {'cost':>8}")
    total = 0.0
    for page in pages:
        img = cv2.imread(str(page))

        try:
            c_boxes, usage = detect_claude_boxes.detect_panels(img, model=model)
            cost = usage["cost_usd"]
            total += cost
            overlay(img, c_boxes, out_claude / f"{page.stem}.jpg")
            c_n, c_cost = len(c_boxes), f"${cost:.4f}"
        except Exception as e:
            c_n, c_cost = "ERR", type(e).__name__

        f_boxes = detect_floodfill.detect_panels(img)
        f_boxes = reading_order(f_boxes)
        overlay(img, f_boxes, out_flood / f"{page.stem}.jpg")

        print(f"{page.name:<10} {c_n:>7} {len(f_boxes):>6} {c_cost:>8}")

    print(f"\ntotal API cost: ${total:.4f}")
    print(f"overlays: {out_claude}/  vs  {out_flood}/")


if __name__ == "__main__":
    main()
