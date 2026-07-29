"""Flood-fill panel detector — no-GPU fallback path.

Not wired into stage 1 yet; the planned config switch is
stage1.detector = magi | floodfill (see docs/plans).

Approach, built for aged scans where art overflows panel borders:
1. Binarize with the per-page adaptive threshold (paper-relative).
2. Background = paper pixels connected to the page border (margins +
   gutters). This handles staggered layouts that defeat projection
   cuts: no straight line has to cross the whole page.
3. Connected components of the rest are panel candidates.
4. Components merged by art bridges (a boot, a cape crossing a gutter)
   are split by eroding until they disconnect, then growing the labels
   back geodesically. A split is only accepted if the pieces are
   near-disjoint, none swallows the parent, and the corridor between
   them is mostly paper — so whites inside a panel can't fake a gutter.

Known limits on 1950s Fiction House-style art (measured on Planet
Comics 69): borderless action spreads and diagonal panels stay merged
or produce overlapping boxes. Stage 1's flags catch both; Magi is the
quality path.
"""

import cv2
import numpy as np

from pipeline.stage1_panels import adaptive_white_threshold

SPLIT_MIN_FRAC = 0.30    # only try splitting boxes bigger than this * page
ERODE_SCHEDULE = (4, 8, 12, 16, 20, 24)
MAX_OVERLAP = 0.20       # accepted split pieces must be near-disjoint
CORRIDOR_PAPER_MIN = 0.50


def _overlap_frac(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
    return ix * iy / small if small else 0


def _grow_labels(mask, markers):
    """Geodesic dilation: grow marker labels through mask pixels only."""
    lab = markers.astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(200):
        grown = cv2.dilate(lab, kernel)
        newly = (lab == 0) & (mask > 0) & (grown > 0)
        if not newly.any():
            break
        lab[newly] = grown[newly]
    return lab.astype(np.int32)


def _try_split(content, gutterish, box, min_area):
    x0, y0, x1, y1 = box
    sub = content[y0:y1, x0:x1]
    gsub = gutterish[y0:y1, x0:x1]
    for r in ERODE_SCHEDULE:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        eroded = cv2.erode(sub, k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(eroded)
        keep = [i for i in range(1, n) if stats[i][4] >= 0.3 * min_area]
        if len(keep) < 2:
            continue
        markers = np.zeros_like(lab, dtype=np.int32)
        for new_id, i in enumerate(keep, start=1):
            markers[lab == i] = new_id
        assigned = _grow_labels(sub, markers)
        boxes = []
        for i in range(1, len(keep) + 1):
            ys, xs = np.nonzero(assigned == i)
            if len(ys) < 0.5 * min_area:
                continue
            boxes.append((int(xs.min()), int(ys.min()),
                          int(xs.max()) + 1, int(ys.max()) + 1))
        if len(boxes) < 2:
            continue
        if any(_overlap_frac(a, b) > MAX_OVERLAP
               for i, a in enumerate(boxes) for b in boxes[i+1:]):
            continue
        parent_area = (x1-x0) * (y1-y0)
        if any((b[2]-b[0])*(b[3]-b[1]) > 0.95 * parent_area for b in boxes):
            continue
        corridor = np.ones((y1-y0, x1-x0), bool)
        for a, b, c, d in boxes:
            corridor[b:d, a:c] = False
        if corridor.sum() and gsub[corridor].mean() < CORRIDOR_PAPER_MIN:
            continue
        return [(x0+a, y0+b, x0+c, y0+d) for a, b, c, d in boxes]
    return None


def detect_panels(img, min_area_ratio=0.02):
    """Return panel boxes as (x, y, w, h), unordered."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    paper = float(np.percentile(gray, 90))
    content = (gray < adaptive_white_threshold(gray)).astype(np.uint8)
    # drop isolated halftone dots so gutters read as clean paper
    content = cv2.morphologyEx(content, cv2.MORPH_OPEN,
                               np.ones((3, 3), np.uint8))
    gutterish = (gray >= 0.94 * paper).astype(np.uint8)

    bg = (1 - content).astype(np.uint8)
    n, lab = cv2.connectedComponents(bg)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    panels_mask = (~np.isin(lab, list(border))).astype(np.uint8)

    min_area = min_area_ratio * w * h
    n2, _, stats, _ = cv2.connectedComponentsWithStats(panels_mask)
    queue = [(x, y, x+bw, y+bh)
             for x, y, bw, bh, area in (stats[i] for i in range(1, n2))
             if area >= min_area]

    final = []
    while queue:
        box = queue.pop()
        if (box[2]-box[0]) * (box[3]-box[1]) > SPLIT_MIN_FRAC * w * h:
            pieces = _try_split(content, gutterish, box, min_area)
            if pieces:
                queue.extend(pieces)
                continue
        final.append(box)

    return [(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
            for x0, y0, x1, y1 in final
            if (x1-x0) * (y1-y0) >= min_area]
