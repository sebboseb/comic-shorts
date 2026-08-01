"""Stage notes: compile review annotations into manifest edits.

Reads work/review_notes.json (written by tools/review_server.py). Two
classes of annotation:

- mechanical tags, applied in code: `reroll` clears the shot's audio so
  stage 6 re-synthesizes it; `slomo` sets the motion and clears audio (the
  slomo audio treatment happens at synth); `cut` removes the shot and
  renumbers the surviving wavs so indices stay aligned.
- free-text notes ("say motherfucking hulk here", "this fact is wrong"),
  compiled into field edits by the story model in ONE call per episode.
  The model may only touch noted shots (plus any shot for an episode-level
  note) and only these fields: line, emotion, motion, sfx, hero, focus
  (0-1000 panel coords). Changed lines get their audio cleared.

Applied notes are archived to review_notes.applied.json, so the gate
starts clean for the next watch.
"""
import json
import os
import re
import time
from pathlib import Path

import anthropic

EDIT_PROMPT = """You edit shot manifests for a comic-recap video pipeline, applying a
human reviewer's notes. You receive the episode's shots (index, panel scene,
line, emotion, motion, sfx, hero) and the reviewer's notes.

Respond with ONLY a JSON object mapping shot index (string) to the fields to
change, e.g.:
{"7": {"line": "Even the motherfucking Hulk.", "motion": "slomo", "emotion": "slow deadpan astonishment"},
 "12": {"sfx": "impact"}}

Rules:
- Only edit shots the notes refer to. A note attached to shot N is about shot N.
- Allowed fields: line, emotion, motion (zoom_face|slow_pan|shake|hold|zoom_out|slomo),
  sfx (none|whoosh|impact|heartbeat|sting), hero (bool),
  focus (=[x,y,w,h] in 0-1000 coords on that shot's panel, when a note asks
  to frame something specific).
- Keep the episode's narration voice (you can see it in the other lines).
- Delivery directions stay bone-dry: never gleeful/amused/joyful (the TTS
  giggles); the comedy lives in the words.
- A slomo shot's line must be a short punch phrase; if a note asks for slomo
  on a long line, trim the line to its punch.
- If a note is unclear, make the smallest reasonable change."""


def _actionable(note):
    """The human-written part of a note. Lines starting with "QA:" are
    diagnostics from qa_takes - they explain WHY a shot was flagged, they
    don't request an edit (the re-roll tag they ride with is applied
    mechanically). Sending them to the story model gave it nothing to do
    and it answered in prose instead of JSON."""
    lines = [l for l in note.splitlines()
             if l.strip() and not l.strip().startswith("QA:")]
    return "\n".join(lines).strip()


def _apply_llm_edits(config, manifest, understanding, notes, ep):
    """One story-model call turning free-text notes into field edits."""
    noted = {i: {**n, "note": _actionable(n.get("note", ""))}
             for i, n in notes.items() if i != "_episode"}
    noted = {i: n for i, n in noted.items() if n["note"]}
    ep_note = notes.get("_episode", {}).get("note", "").strip()
    if not noted and not ep_note:
        return {}
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("free-text notes need ANTHROPIC_API_KEY "
                         "(tags-only runs don't)")
    scenes = {p["id"]: p.get("scene", "") for p in understanding}
    shots_repr = [{"index": i, "panel": s.get("panel"),
                   "scene": scenes.get(s.get("panel"), "")[:160],
                   "line": s.get("line"), "emotion": s.get("emotion"),
                   "motion": s.get("motion"), "sfx": s.get("sfx"),
                   "hero": s.get("hero")}
                  for i, s in enumerate(manifest["shots"])]
    user = (f"Episode {ep} shots:\n{json.dumps(shots_repr, indent=1)}\n\n"
            f"Reviewer notes (by shot index):\n"
            + json.dumps({i: n["note"] for i, n in noted.items()}, indent=1)
            + (f"\n\nEpisode-level note: {ep_note}" if ep_note else ""))
    client = anthropic.Anthropic(max_retries=3)
    resp = client.messages.create(
        model=config["models"]["story_model"], max_tokens=4000,
        system=EDIT_PROMPT, messages=[{"role": "user", "content": user}])
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # the model wrapped or prefaced its JSON; salvage the object, and
        # if there is none, apply no edits rather than kill the whole run
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        print(f"  WARNING: {ep}: story model returned no usable JSON "
              f"({text[:120]!r}); free-text notes not applied this run")
        return {}


def _resolve_audio(audio, workdir):
    """A shot's wav, ALWAYS inside this workdir. The manifest stores paths
    that embed the workdir name ("work_jeff/audio/ep01/shot002.wav"), so a
    manifest copied into another workdir still points at the original -
    following that stored path once let a test on a scratch copy destroy
    the real workdir's audio. Re-anchor on the current workdir and never
    trust the stored directory."""
    p = Path(audio)
    return workdir / "audio" / p.parent.name / p.name


def _guarded_unlink(p, workdir):
    p = p.resolve()
    if workdir.resolve() not in p.parents:
        raise RuntimeError(f"refusing to delete outside workdir: {p}")
    p.unlink(missing_ok=True)


def _clear_audio(shot, workdir):
    audio = shot.get("audio")
    shot["audio"] = None
    shot["duration_frames"] = None
    shot.pop("segments", None)
    if audio:
        _guarded_unlink(_resolve_audio(audio, workdir), workdir)


def _renumber_wavs(shots, workdir, ep):
    """After cuts, wav filenames must match the new shot indices, or the
    next stage-6 run overwrites the wrong files (learned the hard way)."""
    audio_dir = workdir / "audio" / ep
    moves = []
    for new_i, shot in enumerate(shots):
        audio = shot.get("audio")
        if not audio:
            continue
        old = _resolve_audio(audio, workdir)
        if workdir.resolve() not in old.resolve().parents:
            raise RuntimeError(f"refusing to move outside workdir: {old}")
        new = audio_dir / f"shot{new_i:03d}.wav"
        if old.resolve() != new.resolve():
            moves.append((old, new, shot))
    for old, new, shot in moves:  # two-pass: through temp names, no clobber
        old.rename(old.with_suffix(".tmpmove"))
    for old, new, shot in moves:
        old.with_suffix(".tmpmove").rename(new)
        shot["audio"] = str(new.relative_to(workdir.parent)
                            if new.is_relative_to(workdir.parent) else new)


def run(config, workdir: Path):
    notes_path = workdir / "review_notes.json"
    if not notes_path.exists():
        print("no review_notes.json - nothing to apply")
        return
    all_notes = json.loads(notes_path.read_text())
    understanding = json.loads((workdir / "understanding.json").read_text())
    sizes = {p["id"]: p.get("size") for p in understanding}

    for ep, notes in all_notes.items():
        mpath = workdir / "manifests" / f"{ep}.json"
        if not mpath.exists():
            print(f"{ep}: manifest missing, skipping")
            continue
        manifest = json.loads(mpath.read_text())
        shots = manifest["shots"]

        edits = _apply_llm_edits(config, manifest, understanding, notes, ep)
        for i_str, fields in edits.items():
            i = int(i_str)
            if not 0 <= i < len(shots):
                continue
            shot = shots[i]
            focus = fields.pop("focus", None)
            if focus and sizes.get(shot.get("panel")):
                pw, ph = sizes[shot["panel"]]
                shot["focus_box"] = [round(focus[0] / 1000 * pw),
                                     round(focus[1] / 1000 * ph),
                                     round(focus[2] / 1000 * pw),
                                     round(focus[3] / 1000 * ph)]
            changed_speech = any(fields.get(k) is not None
                                 and fields[k] != shot.get(k)
                                 for k in ("line", "emotion", "motion"))
            shot.update({k: v for k, v in fields.items() if v is not None})
            if changed_speech:
                _clear_audio(shot, workdir)
            print(f"  {ep} shot{i:03d}: {', '.join(fields)} updated")

        cuts = set()
        for i_str, n in notes.items():
            if i_str == "_episode":
                continue
            i, tags = int(i_str), n.get("tags", [])
            if not 0 <= i < len(shots):
                continue
            if "cut" in tags:
                cuts.add(i)
                continue
            if "slomo" in tags and shots[i].get("motion") != "slomo":
                shots[i]["motion"] = "slomo"
                _clear_audio(shots[i], workdir)
                print(f"  {ep} shot{i:03d}: slomo")
            if "reroll" in tags:
                _clear_audio(shots[i], workdir)
                print(f"  {ep} shot{i:03d}: re-roll")

        if cuts:
            for i in sorted(cuts):
                _clear_audio(shots[i], workdir)  # delete the orphaned wav
                print(f"  {ep} shot{i:03d}: cut")
            manifest["shots"] = [s for i, s in enumerate(shots)
                                 if i not in cuts]
            _renumber_wavs(manifest["shots"], workdir, ep)

        mpath.write_text(json.dumps(manifest, indent=2))
        print(f"{ep}: notes applied")

    archive = workdir / "review_notes.applied.json"
    history = json.loads(archive.read_text()) if archive.exists() else []
    history.append({"applied_at": time.strftime("%Y-%m-%d %H:%M"),
                    "notes": all_notes})
    archive.write_text(json.dumps(history, indent=2))
    notes_path.unlink()
    print("notes archived; run --stage 6 7 to hear and see the changes")
