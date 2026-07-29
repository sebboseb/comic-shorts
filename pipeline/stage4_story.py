"""Stage 4: story compilation.

Sends the full ordered panel sequence (scenes + dialogue) to an LLM and
gets back the shorts manifests: episode boundaries, hooks, narrator glue
lines, motion intent per shot, hero flags, music mood, and per-platform
metadata. This is the creative stage - review its output at the gate
before spending money on TTS.

Output: work/manifests/epNN.json (one per short)
"""

import json
import os
from pathlib import Path

import anthropic

SYSTEM_PROMPT = """You are a shorts editor turning a comic into a numbered series of
vertical videos (TikTok/YouTube Shorts/Reels). You receive the comic's panels in
reading order, each with a scene description and dialogue lines.

Respond with ONLY a JSON object, no markdown fences:

{
  "shorts": [
    {
      "short_id": "ep01",
      "title": "clickable but honest title, <60 chars",
      "description": "1-2 sentence video description",
      "hashtags": ["#comic", ...],
      "music_mood": "one of: tense_buildup, melancholy, action, playful, ominous, triumphant",
      "hook_note": "one sentence on why this opening will stop the scroll",
      "shots": [
        {
          "panel": "p03_02",
          "speaker": "character name or 'narrator'",
          "line": "the spoken line",
          "emotion": "delivery direction",
          "motion": "one of: zoom_face, slow_pan, shake, hold, zoom_out",
          "hero": false,
          "sfx": "one of: none, whoosh, impact, heartbeat, sting"
        }
      ],
      "cliffhanger_note": "what unresolved question the ending leaves"
    }
  ]
}

Rules:
- COMPREHENSION BEATS CLEVERNESS. The viewer has never read this comic, cannot
  pause, and hears the audio once. If a choice is punchy but confusing, make the
  clear choice instead.
- Each short must OPEN on its most dramatic or intriguing beat, BUT the opening
  must stand on its own without context the viewer doesn't have yet. Do not open
  on a line that refers to a character or event not yet introduced. Reordering
  panels for the hook is allowed only when the reordered opening is
  self-explanatory; otherwise stay in reading order. Prefer a flash-forward you
  can make legible in one narrator sentence over a cryptic cold open.
- Establish the situation fast: within the first two shots the viewer must know
  where we are and who we are following. Name each character in narration the
  first time they appear ("the hooded leader, Wad" - not "he"). Never introduce
  a character by an unexplained pronoun.
- Each short (except the last) must END mid-tension, never at a resolved beat.
- THE SHORTS ARE ONE CONTINUOUS SERIES, watched back to back in order. Every
  short after the first must open with a narrator shot that (a) says in one
  sentence where we left off, and (b) re-identifies by name anyone about to
  speak. Assume the viewer saw the previous short days ago and remembers only
  the broad situation. Write the pair so the last shot of short N and the first
  shot of short N+1 read as a single continuous beat, not two separate videos:
  if short N ends on "the spirit of Mars descends", short N+1 opens by saying
  what Mars then does and to whom.
- Do not restate the same event as a cliffhanger in one short and again as the
  opening beat of the next. Move the story forward.
- Write narrator lines to glue panel transitions that don't self-explain. Favour
  clarity over economy: if a jump in place, time, or who-is-speaking would
  confuse a first-time viewer, spend a narrator line on it. Dialogue still
  carries the drama; narration carries the orientation.
- Keep character dialogue verbatim from the panels. You may trim redundant lines
  but never rewrite a character's words.
- A shot with no line (silent dramatic beat) is allowed: set line to "" and give
  it motion + sfx.
- 'shake' motion is reserved for impact/action beats. 'zoom_face' for emotional
  beats. Vary motion; three identical motions in a row reads as robotic.
- Mark at most 2-3 shots per short as hero (the most dramatic single images).
- Aim for each short to fit the target length: roughly 2.5 words per second of
  spoken text plus ~1.2s per silent shot."""


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def run(config, workdir: Path):
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("Set ANTHROPIC_API_KEY")
    client = anthropic.Anthropic()
    model = config["models"]["story_model"]
    shorts_cfg = config["shorts"]

    panels = json.loads((workdir / "clean.json").read_text())

    panel_lines = []
    for p in panels:
        entry = {"panel": p["id"], "scene": p.get("scene", ""),
                 "dialogue": [{"speaker": d.get("speaker", "unknown"),
                               "text": d.get("text", ""),
                               "emotion": d.get("emotion", "")}
                              for d in p.get("dialogue", [])]}
        if p.get("sfx_text"):
            entry["sfx_in_art"] = p["sfx_text"]
        panel_lines.append(entry)

    user_msg = (
        f"Comic: {config['comic']['title']}\n"
        f"Target: about {shorts_cfg['target_count']} shorts, "
        f"{shorts_cfg['min_seconds']}-{shorts_cfg['max_seconds']} seconds each.\n"
        f"Characters: "
        f"{', '.join(c['name'] + ' (' + c.get('speaking_style', '') + ')' for c in config.get('characters', []))}\n\n"
        f"Panels in reading order:\n{json.dumps(panel_lines, indent=1)}"
    )

    print("compiling story...")
    resp = client.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    data = _parse_json(resp.content[0].text)

    out_dir = workdir / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_panels = {p["id"] for p in panels}
    for short in data["shorts"]:
        bad = [s["panel"] for s in short["shots"]
               if s["panel"] not in valid_panels]
        if bad:
            print(f"  WARNING {short['short_id']}: references unknown panels "
                  f"{bad} - fix before rendering")
        for shot in short["shots"]:
            shot.setdefault("audio", None)
            shot.setdefault("duration_frames", None)
        path = out_dir / f"{short['short_id']}.json"
        path.write_text(json.dumps(short, indent=2))
        n_words = sum(len(s.get("line", "").split()) for s in short["shots"])
        est = n_words / 2.5 + 1.2 * sum(1 for s in short["shots"] if not s.get("line"))
        print(f"  {short['short_id']}: {len(short['shots'])} shots, "
              f"~{est:.0f}s - \"{short['title']}\"")

    print(f"\n{len(data['shorts'])} manifests -> {out_dir}\n"
          "Now run stage 5 (review) and READ the manifests before TTS.")
