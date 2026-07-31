"""Stage 2: panel understanding.

One vision call per panel. Sends your character reference crops alongside
each panel so speaker attribution is face-matching, not guessing. Returns
per panel: scene description, dialogue lines (text, speaker, emotion,
bubble box), and a focus box (where the camera should zoom).

Coordinates come back normalized 0-1000 and are converted to pixels here.

Output: work/understanding.json
"""

import base64
import io
import json
import os
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from PIL import Image

MAX_SEND_PX = 1400  # long side sent to the vision model

BATCH_STATE_FILE = "batch_id.txt"

SYSTEM_PROMPT = """You analyze comic book panels for an automated video pipeline.
You are given reference images of the recurring characters first, then one panel.
Respond with ONLY a JSON object, no markdown fences, no commentary:

{
  "scene": "one sentence describing what happens in this panel",
  "characters_present": ["Name", ...],
  "dialogue": [
    {
      "text": "exact dialogue text from the bubble",
      "speaker": "character name from the roster, or 'narrator' for caption boxes, or 'unknown'",
      "emotion": "short delivery direction, e.g. 'whispered, afraid' or 'shouting, furious'",
      "bubble_box": [x, y, w, h]
    }
  ],
  "focus_box": [x, y, w, h],
  "sfx_text": ["any onomatopoeia drawn into the art, e.g. BOOM"]
}

Rules:
- All boxes use coordinates normalized to 0-1000 on both axes of the panel image.
- dialogue is in reading order (left-to-right, top-to-bottom unless told otherwise).
- bubble_box must cover the ENTIRE bubble including its outline and tail, not just the text.
- focus_box is the most important subject, prefer the speaking character's face.
- If there is no dialogue, return an empty dialogue list.
- Transcribe dialogue exactly, preserving punctuation. Do not invent text."""


def _b64_image(path: Path, max_px: int = MAX_SEND_PX):
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_px:
        scale = max_px / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def _get_or_create_batch(client, requests, workdir: Path):
    """Resume a previously submitted batch if its id is on disk and still
    retrievable; otherwise submit a new one and persist its id."""
    state = workdir / BATCH_STATE_FILE
    if state.exists():
        batch_id = state.read_text().strip()
        try:
            batch = client.messages.batches.retrieve(batch_id)
            print(f"resuming batch {batch_id} ({batch.processing_status})")
            return batch
        except anthropic.NotFoundError:
            print(f"stale batch id {batch_id} (not found on API), "
                  "submitting fresh")
    batch = client.messages.batches.create(requests=requests)
    state.write_text(batch.id)
    print(f"batch {batch.id} submitted ({len(requests)} panels), polling...")
    return batch


def _norm_to_px(box, width, height):
    x, y, w, h = box
    return [round(x / 1000 * width), round(y / 1000 * height),
            round(w / 1000 * width), round(h / 1000 * height)]


def run(config, workdir: Path):
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("Set ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(max_retries=5)  # ride out transient 5xx/502
    model = config["models"]["vision_model"]

    panels_data = json.loads((workdir / "panels.json").read_text())
    panels = panels_data["panels"]

    roster = config.get("characters", [])
    roster_blocks = []
    for ch in roster:
        ref = (Path("config/characters") / ch["ref_image"]
               if ch.get("ref_image") else None)
        if ref is not None and ref.exists():
            roster_blocks.append({"type": "text",
                                  "text": f"Reference image of {ch['name']} "
                                          f"({ch['description']}):"})
            roster_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": _b64_image(ref, 384)},
            })
        else:
            roster_blocks.append({"type": "text",
                                  "text": f"Character (no reference image): "
                                          f"{ch['name']} - {ch['description']}"})
    if roster_blocks:
        # identical prefix across all requests -> cache the roster
        roster_blocks[-1]["cache_control"] = {"type": "ephemeral"}

    # Build one batch request per panel (Batches API = 50% of standard price)
    sizes = {}
    requests = []
    for panel in panels:
        panel_path = workdir / panel["file"]
        sizes[panel["id"]] = Image.open(panel_path).size

        content = list(roster_blocks)
        content.append({"type": "text", "text": "Now analyze this panel:"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": _b64_image(panel_path)},
        })
        requests.append(Request(
            custom_id=panel["id"],
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            ),
        ))

    batch = _get_or_create_batch(client, requests, workdir)

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        c = batch.request_counts
        print(f"  processing... {c.succeeded} done, {c.errored} errored, "
              f"{c.processing} in flight")
        time.sleep(20)

    parsed_by_id = {}
    for result in client.messages.batches.results(batch.id):
        pid = result.custom_id
        if result.result.type != "succeeded":
            print(f"  {pid}: batch request {result.result.type}")
            parsed_by_id[pid] = {"scene": "", "characters_present": [],
                                 "dialogue": [], "focus_box": [0, 0, 1000, 1000],
                                 "sfx_text": [],
                                 "parse_error": f"batch {result.result.type}"}
            continue
        msg = result.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            parsed_by_id[pid] = _parse_json(text)
        except json.JSONDecodeError as e:
            print(f"  {pid}: JSON parse failed ({e}), storing raw")
            parsed_by_id[pid] = {"scene": "", "characters_present": [],
                                 "dialogue": [], "focus_box": [0, 0, 1000, 1000],
                                 "sfx_text": [], "parse_error": text}

    canon = {ch["name"].lower(): ch["name"] for ch in roster}
    canon.update({"narrator": "narrator", "unknown": "unknown"})

    results = []
    for panel in panels:  # keep reading order regardless of result order
        pw, ph = sizes[panel["id"]]
        parsed = parsed_by_id.get(panel["id"], {
            "scene": "", "characters_present": [], "dialogue": [],
            "focus_box": [0, 0, 1000, 1000], "sfx_text": [],
            "parse_error": "missing from batch results"})
        for line in parsed.get("dialogue", []):
            s = line.get("speaker", "unknown")
            if s not in canon.values() and s.lower() in canon:
                line["speaker"] = canon[s.lower()]  # fix casing drift
            if "bubble_box" in line:
                line["bubble_box"] = _norm_to_px(line["bubble_box"], pw, ph)
        if "focus_box" in parsed:
            parsed["focus_box"] = _norm_to_px(parsed["focus_box"], pw, ph)
        results.append({**panel, "size": [pw, ph], **parsed})
        n_lines = len(parsed.get("dialogue", []))
        print(f"{panel['id']}: {n_lines} lines - {parsed.get('scene', '')[:60]}")

    unknowns = sum(1 for p in results for d in p.get("dialogue", [])
                   if d.get("speaker") == "unknown")
    (workdir / "understanding.json").write_text(json.dumps(results, indent=2))
    (workdir / BATCH_STATE_FILE).unlink(missing_ok=True)
    print(f"\nDone -> understanding.json"
          f" ({unknowns} lines with unknown speaker to fix at review)")
