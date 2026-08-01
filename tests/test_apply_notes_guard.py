"""Regression: apply_notes on a copied workdir must never touch the
original workdir's audio.

The manifest stores audio paths that embed the workdir name; a manifest
copied into a scratch workdir therefore points at the original. On
2026-07-31 a test run on such a copy deleted and moved every wav in the
real work_jeff. _resolve_audio re-anchors on the current workdir and
_guarded_unlink refuses to delete outside it.
"""
import json
import wave

import pytest

from pipeline import apply_notes


def _wav(path):
    with wave.open(str(path), "wb") as w:
        w.setparams((1, 2, 24000, 0, "NONE", "NONE"))
        w.writeframes(b"\x00\x00" * 2400)


def _workdir(root, name, audio_path):
    wd = root / name
    (wd / "audio/ep01").mkdir(parents=True)
    (wd / "manifests").mkdir()
    (wd / "understanding.json").write_text("[]")
    manifest = {"short_id": "ep01", "shots": [
        {"panel": "x", "line": "hi", "speaker": "narrator",
         "audio": audio_path, "duration_frames": 3}]}
    (wd / "manifests/ep01.json").write_text(json.dumps(manifest))
    return wd


def test_copied_workdir_cannot_touch_original(tmp_path):
    stale = "orig/audio/ep01/shot000.wav"  # path embeds the ORIGINAL workdir
    orig = _workdir(tmp_path, "orig", stale)
    copy = _workdir(tmp_path, "copy", stale)
    _wav(orig / "audio/ep01/shot000.wav")
    (copy / "review_notes.json").write_text(json.dumps(
        {"ep01": {"0": {"tags": ["reroll"], "note": ""}}}))

    apply_notes.run({"models": {"story_model": "unused"}}, copy)

    assert (orig / "audio/ep01/shot000.wav").exists()


def test_guarded_unlink_refuses_outside(tmp_path):
    outside = tmp_path / "elsewhere.wav"
    _wav(outside)
    with pytest.raises(RuntimeError, match="outside workdir"):
        apply_notes._guarded_unlink(outside, tmp_path / "wd")
    assert outside.exists()
