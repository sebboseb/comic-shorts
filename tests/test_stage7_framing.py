from pipeline.stage7_render import (REFRAME, _band_window, _frame_window,
                                    _part_number)


def test_band_window_centers_on_focus_and_stays_inside():
    x, w, ncx = _band_window(4000, 1000, 0.5, 1.5)
    assert w == 1500 and x == 1250
    assert abs(ncx - 0.5) < 0.01
    # focus at the far right edge: band clamps, focus lands inside it
    x, w, ncx = _band_window(4000, 1000, 1.0, 1.5)
    assert x == 2500 and 0 <= ncx <= 1


def test_band_window_narrow_panel_uses_whole_width():
    x, w, _ = _band_window(1200, 1000, 0.5, 1.5)
    assert x == 0 and w == 1200


def test_part_number():
    assert _part_number("ep01") == 1
    assert _part_number("ep12") == 12
    assert _part_number("teaser") is None
    assert _part_number(None) is None


def test_frame_window_is_frame_aspect_and_inside_panel():
    x, y, w, h = _frame_window(2000, 1000, (900, 400, 200, 200), 2.4)
    assert 0 <= x and x + w <= 2000
    assert 0 <= y and y + h <= 1000
    assert abs(w / h - 1080 / 1920) < 0.01


def test_frame_window_huge_focus_clamps_to_panel():
    # focus box ~ the whole panel: any tightness clamps to the panel
    full = _frame_window(1000, 1000, (0, 0, 1000, 1000), 2.4)
    again = _frame_window(1000, 1000, (0, 0, 1000, 1000), 2.4 * 0.62)
    assert full == again  # why REFRAME needs max_frac


def test_frame_window_max_frac_forces_a_punch_in():
    full = _frame_window(1000, 1000, (0, 0, 1000, 1000), 2.4, 1.0)
    tight = _frame_window(1000, 1000, (0, 0, 1000, 1000), 2.4 * 0.62, 0.7)
    assert tight[2] < full[2] and tight[3] < full[3]
    assert tight[2] <= 700


def test_reframe_schedule_varies_consecutive_repeats():
    windows = [_frame_window(1000, 1000, (0, 0, 1000, 1000), 2.4 * tm, mf)
               for tm, mf in REFRAME]
    assert windows[0] != windows[1]  # first repeat visibly differs
