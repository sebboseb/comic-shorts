import yaml

from tools import tag_ui

SEED = '''# comment that must survive
comic:
  title: "T"

characters:
  - name: "nova"
    description: "d1"
    ref_image: "nova.png"
    voice_id: "V1"
    speaking_style: "neutral"
  - name: "Thanos"
    description: "d2"
    voice_id: "V2"
    speaking_style: "neutral"

models:
  vision_model: "m"
'''


def _seeded(tmp_path, monkeypatch):
    cfg = tmp_path / "comic.yaml"
    cfg.write_text(SEED)
    monkeypatch.setattr(tag_ui, "CONFIG_PATH", cfg)
    return cfg


def test_slugify():
    assert tag_ui.slugify("Nova (no helmet)!") == "nova_no_helmet"
    assert tag_ui.slugify("日本語") == "unnamed"


def test_set_ref_image_rewrites_only_target(tmp_path, monkeypatch):
    cfg = _seeded(tmp_path, monkeypatch)
    assert tag_ui.set_ref_image("nova", "new.png") is True
    text = cfg.read_text()
    assert "# comment that must survive" in text
    data = yaml.safe_load(text)
    nova, thanos = data["characters"]
    assert nova["ref_image"] == "new.png"
    assert nova["voice_id"] == "V1"
    assert thanos == {"name": "Thanos", "description": "d2",
                      "voice_id": "V2", "speaking_style": "neutral"}


def test_set_ref_image_inserts_missing_line(tmp_path, monkeypatch):
    cfg = _seeded(tmp_path, monkeypatch)
    assert tag_ui.set_ref_image("Thanos", "t.png") is True
    data = yaml.safe_load(cfg.read_text())
    assert data["characters"][1]["ref_image"] == "t.png"
    assert data["models"] == {"vision_model": "m"}


def test_set_ref_image_unknown_name_is_noop(tmp_path, monkeypatch):
    cfg = _seeded(tmp_path, monkeypatch)
    assert tag_ui.set_ref_image("nobody", "x.png") is False
    assert cfg.read_text() == SEED


def test_append_character_lands_last_in_list(tmp_path, monkeypatch):
    cfg = _seeded(tmp_path, monkeypatch)
    tag_ui.append_character("Gamora", "gamora.png")
    data = yaml.safe_load(cfg.read_text())
    assert data["characters"][-1]["name"] == "Gamora"
    assert data["characters"][-1]["ref_image"] == "gamora.png"
    assert data["models"] == {"vision_model": "m"}
