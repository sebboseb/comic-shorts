import json

import pytest

from pipeline.stage2_understand import _norm_to_px
from pipeline.stage2_understand import _parse_json as parse_stage2
from pipeline.stage4_story import _parse_json as parse_stage4

# both copies are intentionally duplicated today - test each
PARSERS = [parse_stage2, parse_stage4]


@pytest.mark.parametrize("parse", PARSERS)
def test_parse_plain_json(parse):
    assert parse('{"a": 1}') == {"a": 1}


@pytest.mark.parametrize("parse", PARSERS)
def test_parse_json_fence(parse):
    assert parse('```json\n{"a": 1}\n```') == {"a": 1}


@pytest.mark.parametrize("parse", PARSERS)
def test_parse_bare_fence(parse):
    assert parse('```\n{"a": 1}\n```') == {"a": 1}


@pytest.mark.parametrize("parse", PARSERS)
def test_parse_garbage_raises(parse):
    with pytest.raises(json.JSONDecodeError):
        parse("not json at all")


def test_norm_to_px_full_box():
    assert _norm_to_px([0, 0, 1000, 1000], 800, 600) == [0, 0, 800, 600]


def test_norm_to_px_scales_axes_independently():
    assert _norm_to_px([500, 500, 100, 100], 1000, 2000) == [500, 1000, 100, 200]


def test_norm_to_px_rounds():
    assert _norm_to_px([333, 0, 334, 0], 3, 3) == [1, 0, 1, 0]
