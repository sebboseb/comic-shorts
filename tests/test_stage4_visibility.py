from pipeline.stage4_story import _visibility_warnings

PANEL_CHARS = {"p01": [], "p02": ["Nova"], "p03": ["Nova", "Quill"]}


def _short(shots):
    return {"shots": [{"panel": p, "line": l} for p, l in shots]}


def test_intro_on_empty_panel_warns():
    short = _short([("p01", "Meet our boy Nova, last of the corps."),
                    ("p02", "And Nova is not having a good day.")])
    assert _visibility_warnings(short, PANEL_CHARS, ["Nova"]) == \
        [(0, "Nova", "p01")]


def test_intro_on_panel_showing_them_is_clean():
    short = _short([("p02", "Meet Nova."),
                    ("p01", "Nova again, now off-screen - fine.")])
    assert _visibility_warnings(short, PANEL_CHARS, ["Nova"]) == []


def test_only_first_mention_counts():
    short = _short([("p03", "Quill thinks he's the main character."),
                    ("p01", "Quill, again, over empty space.")])
    assert _visibility_warnings(short, PANEL_CHARS, ["Quill"]) == []


def test_matches_whole_words_only():
    short = _short([("p01", "The Novation project begins.")])
    assert _visibility_warnings(short, PANEL_CHARS, ["Nova"]) == []
