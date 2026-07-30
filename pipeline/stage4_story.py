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

PROMPT_TEMPLATE = """You are a shorts editor turning a comic into a numbered series of
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
- OPEN ON A PERSON, NOT A PREMISE. The first shot names a specific character
  and puts a hook on them - who they are and what is about to happen to them
  ("This is Richard Rider. In six months he is the only Nova Corps officer
  left alive."). Pick the most striking or most doomed figure available, and
  prefer a character the audience may already know. Do NOT open on scene
  setting, on a faction, or on an event with nobody in it; a premise is not a
  hook, a person in trouble is.
- Establish the situation fast: within the first two shots the viewer must know
  where we are and who we are following. Name each character in narration the
  first time they appear - give the role and the name together, the way
  "the tribe's leader, <name>" does, never a bare "he" or "they".
- Each short (except the last) must END mid-tension, never at a resolved beat.
- THE SHORTS ARE ONE CONTINUOUS SERIES, watched back to back in order. Every
  short after the first must open with a narrator shot that (a) says in one
  sentence where we left off, and (b) re-identifies by name anyone about to
  speak. Assume the viewer saw the previous short days ago and remembers only
  the broad situation. Write the pair so the last shot of short N and the first
  shot of short N+1 read as a single continuous beat, not two separate videos:
  if short N ends on some force or figure arriving unseen, short N+1 opens by
  naming it and saying what it does next, and to whom.
- Do not restate the same event as a cliffhanger in one short and again as the
  opening beat of the next. Move the story forward.
- Write narrator lines to glue panel transitions that don't self-explain. Favour
  clarity over economy: if a jump in place, time, or who-is-speaking would
  confuse a first-time viewer, spend a narrator line on it. Dialogue still
  carries the drama; narration carries the orientation.
- NARRATE, DON'T SUMMARISE. The narrator is a storyteller who already knows how
  this ends, not a caption writer describing a picture. This matters more than
  any other rule for whether the short holds a viewer.
    * Never describe what is visibly happening in the panel (as in "<name>
      examines an artifact"). The viewer can see it. Say what it MEANS, what it
      COSTS, or what is about to go wrong ("He'd crossed a hundred million miles
      to find that stone. It would cost him everything.").
    * Lead with the consequence or the stakes, not the stage direction.
    * Vary sentence length. Short sentences hit. Let a long one build, then cut
      it off with a short one.
    * Use concrete, physical nouns and hard verbs. No "suddenly", no "little did
      they know", no "meanwhile", no rhetorical questions to camera.
    * Withhold. Name the threat late in the line, not early.
    * Address nobody. No "you won't believe", no direct address to the viewer.
  The synthesised voice is flat and literal, so the WRITING has to carry the
  performance: rhythm and word choice are the only prosody you get.
{DIALOGUE_RULES}
- A shot with no line (silent dramatic beat) is allowed: set line to "" and give
  it motion + sfx.
- 'shake' motion is reserved for impact/action beats. 'zoom_face' for emotional
  beats. Vary motion; three identical motions in a row reads as robotic.
- Mark at most 2-3 shots per short as hero (the most dramatic single images).
- Aim for each short to fit the target length: roughly 2.5 words per second of
  spoken text plus ~1.2s per silent shot."""


DIALOGUE_RULES = """- Keep character dialogue verbatim from the panels. You may trim redundant lines
  but never rewrite a character's words.
- CAST DISCIPLINE. A short can only carry a handful of voices before the viewer
  loses track of who is who. Only the characters listed as PRINCIPALS may be
  given a spoken shot. For every other line in the panels:
    * If it matters to the story, hand the information to the narrator in their
      own words ("the Kree line breaks" rather than a soldier shouting it).
    * If it is background noise - crowd chatter, orders barked by unnamed
      troops, someone reacting off-panel - cut it.
    * A line whose speaker is 'unknown' is never given a spoken shot. Narrate it
      or drop it; guessing a speaker puts the wrong voice in the viewer's ear.
  Prefer FEWER speaking characters than the panels contain."""


SINGLE_VOICE_RULES = """- ONE VOICE. Every single shot has speaker "narrator". No character is ever
  given a spoken shot, not even a principal. The whole episode is one person
  telling the story, which is what lets it hold together over its full length.
- REPORT DIALOGUE, DON'T STAGE IT. A character's words become the narrator's
  report of them, keeping the sense and any wording that is doing real work:
    * "Quill tells him the force fields will hold, and asks if that is what he
      wanted to hear."   (not Quill saying it)
    * "'How could they?' replies Damien."  - a short quote INSIDE a narrator
      line is fine when the exact words land harder than a paraphrase. The
      narrator still speaks it; attribute it in the same breath.
  Never leave a line of dialogue unattributed - always say who said it, or drop
  it. Unattributed speech is the main way a single-voice recap loses a viewer.
- ANNOUNCE EVERY MOVE. Any change of place, time or focus gets an explicit
  marker at the start of the line: "Back on the Kree line", "Elsewhere",
  "Seconds later", "On the other side of the battlefield", "Just before he
  leaves". Never let the viewer work out for themselves that we have cut away.
- ONE BEAT PER SHOT, AND KEEP THEM SHORT. Aim for 8-18 spoken words per shot,
  each carrying exactly one thing that happens. This is what gives the edit its
  rhythm: with no speaker changes to honour, the picture can change as often as
  the story does. Expect MANY MORE shots than there are panels - reusing a panel
  across two or three consecutive shots is correct and expected.
- DRIVE FORWARD WITH CONNECTIVES. Chain beats with "until", "but", "then",
  "before", "and now" so each line sets up the next. Setup then reversal is the
  engine: state what someone expects, then break it."""


def _voice_brief(config):
    """The narrator persona and the character epithets.

    Sent last and marked as overriding, because a house voice is the whole
    identity of a channel: two shorts with the same facts and different
    narrators are different products. Epithets are configured rather than
    improvised so a character sounds the same across every episode.
    """
    shorts = config.get("shorts", {})
    style = (shorts.get("narration_style") or "").strip()
    eps = [(c["name"], c["epithet"]) for c in config.get("characters", [])
           if c.get("epithet")]
    single_voice = shorts.get("narration_mode", "single_voice") == "single_voice"
    principals = [] if single_voice else [
        c["name"] for c in config.get("characters", []) if c.get("principal")]
    if not style and not eps and not principals:
        return ""

    out = []
    if principals:
        out.append("\n\n=== PRINCIPALS - the only characters who may speak ===\n"
                   + ", ".join(principals)
                   + "\nEvery other character's lines must be narrated or cut, "
                     "per the cast-discipline rule.")
    out.append("\n\n=== NARRATOR VOICE (overrides the general guidance above) ===")
    if style:
        out.append("Write every narrator line in this voice:\n" + style)
    if eps:
        out.append(
            "\nCharacter epithets - use these instead of plain names when "
            "introducing or referring to someone, and vary them rather than "
            "repeating one. Pick whichever fits the beat; adapt the wording "
            "to the sentence. Do NOT apply them to a character's own spoken "
            "dialogue, only to narration:")
        out.extend(f"  {n}: {e}" for n, e in eps)
    out.append(
        "\nThe voice applies to NARRATION ONLY. Character dialogue stays "
        "verbatim from the panels - never rewrite what a character says to "
        "be funnier. The comedy is the narrator's, not theirs.")
    return "\n".join(out)


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def _system_prompt(config):
    """The dialogue rules differ fundamentally by narration mode, so the prompt
    is assembled rather than fixed. single_voice matches how the reference
    channels actually work: one narrator throughout, dialogue reported rather
    than performed."""
    mode = config.get("shorts", {}).get("narration_mode", "single_voice")
    rules = SINGLE_VOICE_RULES if mode == "single_voice" else DIALOGUE_RULES
    return PROMPT_TEMPLATE.replace("{DIALOGUE_RULES}", rules)


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
        + _voice_brief(config)   # last, so the voice is the freshest instruction
    )

    print("compiling story...")
    resp = client.messages.create(
        model=model,
        max_tokens=16000,
        system=_system_prompt(config),
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
