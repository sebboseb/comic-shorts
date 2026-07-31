import numpy as np

from pipeline.stage1_panels import _layout_score, detect_panels, reading_order

TL = (10, 10, 100, 100)
TR = (200, 10, 100, 100)
BL = (10, 200, 100, 100)
BR = (200, 200, 100, 100)


def test_reading_order_ltr_grid():
    assert reading_order([BR, TL, BL, TR], "ltr") == [TL, TR, BL, BR]


def test_reading_order_rtl_reverses_within_rows():
    assert reading_order([BR, TL, BL, TR], "rtl") == [TR, TL, BR, BL]


def test_reading_order_uneven_heights_same_row():
    a = (10, 10, 100, 100)   # center y = 60
    b = (200, 20, 100, 80)   # center y = 60
    assert reading_order([b, a], "ltr") == [a, b]


def test_reading_order_empty():
    assert reading_order([], "ltr") == []


def test_detect_panels_two_stacked_panels():
    img = np.full((400, 400, 3), 255, np.uint8)
    img[20:180, 20:380] = 30
    img[220:380, 20:380] = 30
    boxes = detect_panels(img)
    assert len(boxes) == 2
    ys = sorted(b[1] for b in boxes)
    assert ys[0] < 200 < ys[1]


def test_layout_score_empty_is_negative():
    assert _layout_score([], 400, 400) == -1.0


def test_layout_score_whole_page_fallback_scores_low():
    whole = [(0, 0, 395, 395)]
    halves = [(0, 0, 400, 190), (0, 210, 400, 190)]
    assert _layout_score(whole, 400, 400) < _layout_score(halves, 400, 400)
