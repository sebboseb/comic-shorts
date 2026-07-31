import pytest

from pipeline.stage4_story import _parse_story, _response_text


class _Block:
    def __init__(self, type, text=""):
        self.type = type
        self.text = text


class _Resp:
    def __init__(self, blocks, stop_reason="end_turn"):
        self.content = blocks
        self.stop_reason = stop_reason


def test_response_text_single_block():
    assert _response_text(_Resp([_Block("text", '{"a": 1}')])) == '{"a": 1}'


def test_response_text_skips_non_text_blocks():
    resp = _Resp([_Block("thinking"), _Block("text", "out")])
    assert _response_text(resp) == "out"


def test_response_text_concatenates():
    resp = _Resp([_Block("text", "one"), _Block("text", "two")])
    assert _response_text(resp) == "onetwo"


def test_parse_story_happy_path(tmp_path):
    raw = tmp_path / "story_raw.txt"
    data = _parse_story('{"shorts": [{"short_id": "ep01"}]}', "end_turn", raw)
    assert data["shorts"][0]["short_id"] == "ep01"
    assert raw.read_text() == '{"shorts": [{"short_id": "ep01"}]}'


def test_parse_story_truncation_saves_raw_before_dying(tmp_path):
    raw = tmp_path / "story_raw.txt"
    with pytest.raises(SystemExit) as exc:
        _parse_story('{"shorts": [', "max_tokens", raw)
    assert "story_raw.txt" in str(exc.value)
    assert raw.read_text() == '{"shorts": ['


def test_parse_story_bad_json_saves_raw_before_dying(tmp_path):
    raw = tmp_path / "story_raw.txt"
    with pytest.raises(SystemExit) as exc:
        _parse_story("not json", "end_turn", raw)
    assert "story_raw.txt" in str(exc.value)
    assert raw.read_text() == "not json"


def test_parse_story_wrong_shape_saves_raw_before_dying(tmp_path):
    raw = tmp_path / "story_raw.txt"
    with pytest.raises(SystemExit) as exc:
        _parse_story('{"other": 1}', "end_turn", raw)
    assert "story_raw.txt" in str(exc.value)
    assert raw.read_text() == '{"other": 1}'
