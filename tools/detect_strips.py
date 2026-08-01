"""Detect story (strip) boundaries in an issue by finding its title card.

Serial comics like It's Jeff open every strip with the same title logo.
Multi-scale template matching finds those pages locally and for free -
no vision API needed to know where one story ends and the next begins.
Output: strip page ranges, ready to drive per-episode grouping.

Usage:
  .venv/bin/python tools/detect_strips.py <pages_dir> <template.png> [threshold]

Make the template once per comic: crop the title logo from any page
(e.g. `.venv/bin/python tools/detect_strips.py --make-template page.jpg out.png
x y w h` with fractional coords).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

SCALES = np.linspace(0.5, 1.6, 12)   # logo size varies with panel layout
DEFAULT_THRESHOLD = 0.70


def make_template(page_path, out_path, fx, fy, fw, fh):
    img = cv2.imread(str(page_path))
    h, w = img.shape[:2]
    crop = img[int(fy * h):int((fy + fh) * h), int(fx * w):int((fx + fw) * w)]
    cv2.imwrite(str(out_path), crop)
    print(f"template {crop.shape[1]}x{crop.shape[0]} -> {out_path}")


def _logo_hue_range(tmpl):
    """Derive the logo's dominant hue band from the template itself, so the
    color channel generalizes to any comic's logo without configuration."""
    hsv = cv2.cvtColor(tmpl, cv2.COLOR_BGR2HSV)
    sat = (hsv[..., 1] > 60) & (hsv[..., 2] > 60) & (hsv[..., 2] < 230)
    if not sat.any():
        return None
    h = int(np.median(hsv[..., 0][sat]))
    return (max(h - 15, 0), 60, 60), (min(h + 15, 179), 255, 230)

def _scaled_best(band, tmpl):
    best = 0.0
    for s in SCALES:
        t = cv2.resize(tmpl, None, fx=s, fy=s)
        if t.shape[0] >= band.shape[0] or t.shape[1] >= band.shape[1]:
            continue
        r = cv2.matchTemplate(band.astype(np.float32), t.astype(np.float32),
                              cv2.TM_CCOEFF_NORMED)
        best = max(best, float(r.max()))
    return best

def best_match(page, tmpl, hue_range):
    """Combined score: the title card must match on BOTH grayscale structure
    and the logo's color mask. Either channel alone false-positives (gray on
    white-heavy pages, color on pages full of the logo's hue - pool water
    defeated the blue mask on its own); their minimum separated cleanly
    (validated on It's Jeff #1: 13/14 strips, zero false positives).
    Searched in the page's top band, where title cards live."""
    band = page[: page.shape[0] // 3]
    g = _scaled_best(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY),
                     cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY))
    if hue_range is None:
        return g
    lo, hi = hue_range
    c = _scaled_best(cv2.inRange(cv2.cvtColor(band, cv2.COLOR_BGR2HSV), lo, hi),
                     cv2.inRange(cv2.cvtColor(tmpl, cv2.COLOR_BGR2HSV), lo, hi))
    return min(g, c)


def main():
    if sys.argv[1] == "--make-template":
        make_template(sys.argv[2], sys.argv[3], *map(float, sys.argv[4:8]))
        return
    pages_dir, tmpl_path = Path(sys.argv[1]), sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_THRESHOLD
    tmpl = cv2.imread(tmpl_path)
    hue_range = _logo_hue_range(tmpl)
    pages = sorted(p for p in pages_dir.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    starts = []
    for i, page in enumerate(pages, start=1):
        img = cv2.imread(str(page))
        score = best_match(img, tmpl, hue_range)
        hit = score >= threshold
        if hit:
            starts.append(i)
        print(f"page {i:02d} {page.name:<28} {score:.2f} {'<- strip start' if hit else ''}")

    if not starts:
        print("\nno title cards found - lower the threshold or re-crop the template")
        return
    print("\nstrips:")
    bounds = starts + [len(pages) + 1]
    for a, b in zip(bounds, bounds[1:]):
        print(f"  pages {a}-{b - 1}")


if __name__ == "__main__":
    main()
