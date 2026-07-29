"""Stage 1: panel extraction.

Detects panels on flat comic pages by finding ink regions separated by
white gutters. Works well on clean western layouts. Pages where detection
looks unreliable get flagged in the output so you can fix them manually
instead of silently getting garbage.

Output: work/panels/pPP_NN.png crops + work/panels.json
"""

import json
from pathlib import Path

import cv2
import numpy as np


def _boxes_from_mask(mask, w, h, min_area_ratio):
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * w * h
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh >= min_area:
            boxes.append((x, y, bw, bh))
    # drop boxes fully contained in a larger one
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        inside = any(b[0] >= k[0] and b[1] >= k[1]
                     and b[0] + b[2] <= k[0] + k[2]
                     and b[1] + b[3] <= k[1] + k[3] for k in kept)
        if not inside:
            kept.append(b)
    return kept


def _layout_score(boxes, w, h):
    """How much does this look like a real panel layout?"""
    if not boxes:
        return -1.0
    coverage = sum(bw * bh for _, _, bw, bh in boxes) / (w * h)
    if len(boxes) == 1 and coverage > 0.90:
        return 0.1  # whole-page fallback, weak
    score = min(len(boxes), 8)
    if 0.40 <= coverage <= 0.98:
        score += 2
    return score


def adaptive_white_threshold(gray):
    """Gutter cutoff derived from this page's paper, not an absolute value.

    p90 brightness lands inside the paper/gutter mass even on mostly-dark
    pages; 0.92x that keeps aged, yellowed paper (paper ~220 on 1950s scans,
    where a fixed 235 classifies the whole page as content) on the gutter
    side of the cut."""
    return 0.92 * float(np.percentile(gray, 90))


def detect_panels(img, white_threshold=None, dark_threshold=48,
                  min_area_ratio=0.02):
    """Try both gutter polarities (white gutters and black gutters) and
    keep whichever segmentation looks more like a panel layout."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if white_threshold is None:
        white_threshold = adaptive_white_threshold(gray)

    # content = anything darker than white gutters
    _, light_mask = cv2.threshold(gray, white_threshold, 255,
                                  cv2.THRESH_BINARY_INV)
    # content = anything brighter than black gutters
    _, dark_mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY)

    candidates = [_boxes_from_mask(m, w, h, min_area_ratio)
                  for m in (light_mask, dark_mask)]
    return max(candidates, key=lambda b: _layout_score(b, w, h))


def reading_order(boxes, direction="ltr"):
    """Cluster boxes into rows by vertical center, then sort within rows."""
    rows = []
    for b in sorted(boxes, key=lambda b: b[1]):
        placed = False
        for row in rows:
            ref = row[0]
            b_cy = b[1] + b[3] / 2
            ref_cy = ref[1] + ref[3] / 2
            if abs(b_cy - ref_cy) < 0.5 * max(b[3], ref[3]):
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])

    ordered = []
    for row in sorted(rows, key=lambda r: min(b[1] for b in r)):
        row_sorted = sorted(row, key=lambda b: b[0], reverse=(direction == "rtl"))
        ordered.extend(row_sorted)
    return ordered


def run(config, workdir: Path):
    pages_dir = Path(config["comic"]["pages_dir"])
    direction = config["comic"].get("reading_direction", "ltr")
    s1 = config.get("stage1", {})
    white_threshold = s1.get("white_threshold")  # None = per-page adaptive
    min_area_ratio = s1.get("min_panel_area_ratio", 0.02)
    margin = s1.get("crop_margin", 4)

    out_dir = workdir / "panels"
    out_dir.mkdir(parents=True, exist_ok=True)

    page_files = sorted(
        p for p in pages_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    if not page_files:
        raise SystemExit(f"No page images found in {pages_dir}")

    overrides_path = Path("config/panel_overrides.json")
    overrides = (json.loads(overrides_path.read_text())
                 if overrides_path.exists() else {})

    panels = []
    flagged_pages = []

    for page_idx, page_path in enumerate(page_files, start=1):
        img = cv2.imread(str(page_path))
        if img is None:
            flagged_pages.append({"page": page_idx, "file": page_path.name,
                                  "reason": "could not read image"})
            continue

        h, w = img.shape[:2]
        manual = overrides.get(page_path.name)
        if manual:
            # hand-drawn boxes from tools/tag_ui.py, already in reading order
            boxes = [tuple(b) for b in manual]
        else:
            boxes = detect_panels(img, white_threshold=white_threshold,
                                  dark_threshold=s1.get("dark_threshold", 48),
                                  min_area_ratio=min_area_ratio)
            boxes = reading_order(boxes, direction)

        coverage = sum(bw * bh for _, _, bw, bh in boxes) / (w * h) if boxes else 0
        if not manual and (len(boxes) == 0 or coverage < 0.45):
            flagged_pages.append({
                "page": page_idx, "file": page_path.name,
                "reason": f"{len(boxes)} panels found, coverage {coverage:.0%} "
                          "— check for bleed/overlapping layout, crop manually if wrong",
            })
        elif not manual and len(boxes) == 1 and coverage >= 0.90:
            flagged_pages.append({
                "page": page_idx, "file": page_path.name,
                "reason": f"whole-page fallback (1 panel, coverage {coverage:.0%}) "
                          "— real splash page, or gutters missed (aged paper above "
                          "white_threshold?)",
            })

        for panel_idx, (x, y, bw, bh) in enumerate(boxes, start=1):
            x0 = max(0, x - margin)
            y0 = max(0, y - margin)
            x1 = min(w, x + bw + margin)
            y1 = min(h, y + bh + margin)
            crop = img[y0:y1, x0:x1]
            name = f"p{page_idx:02d}_{panel_idx:02d}.png"
            cv2.imwrite(str(out_dir / name), crop)
            panels.append({
                "id": name.removesuffix(".png"),
                "page": page_idx,
                "panel": panel_idx,
                "file": f"panels/{name}",
                "bbox_on_page": [x0, y0, x1 - x0, y1 - y0],
            })

        print(f"page {page_idx:02d}: {len(boxes)} panels ({coverage:.0%} coverage)")

    result = {"panels": panels, "flagged_pages": flagged_pages}
    (workdir / "panels.json").write_text(json.dumps(result, indent=2))

    print(f"\n{len(panels)} panels -> {out_dir}")
    if flagged_pages:
        print(f"WARNING: {len(flagged_pages)} pages flagged, check panels.json "
              "and fix crops by hand before stage 2:")
        for f in flagged_pages:
            print(f"  - {f['file']}: {f['reason']}")
