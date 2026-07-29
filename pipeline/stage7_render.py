"""Stage 7: render manifests to vertical video with ffmpeg.

One clip per shot: the panel art fitted onto a 1080x1920 canvas over a
blurred, darkened copy of itself, with a per-shot Ken Burns move driven by
the manifest's `motion` intent and aimed at the panel's `focus_box`. Clips
carry their own audio (the stage-6 wav, or generated silence), so
concatenating them keeps A/V in sync without a global offset.

Captions are burned in from stage 6's per-sentence timings (see
captions.py), and a music bed is mixed under the narration per the
episode's music_mood. Still scrappy: no transitions, no sfx stings.
Remotion is the plan of record for the polished version.

Runs without stage 6: shots with no audio get a duration estimated from
the line length, so you can eyeball the motion before spending TTS time.

Output: work/renders/epNN.mp4
"""

import json
import shutil
import subprocess
from pathlib import Path

from pipeline import captions

FPS = 30
W, H = 1080, 1920
SILENT_FRAMES = round(1.2 * FPS)
WORDS_PER_SEC = 2.5      # matches the pacing stage 4 writes to
TAIL_FRAMES = 8          # breath after a line so cuts don't clip the audio


def _est_frames(line):
    if not line:
        return SILENT_FRAMES
    words = len(line.split())
    return max(SILENT_FRAMES, round(words / WORDS_PER_SEC * FPS) + TAIL_FRAMES)


def _focus_norm(panel):
    """focus_box is in original-panel pixels; return its centre as 0-1
    fractions, which are resolution independent."""
    box = panel.get("focus_box")
    size = panel.get("size")
    if not box or not size:
        return 0.5, 0.5
    x, y, bw, bh = box
    pw, ph = size
    if not pw or not ph:
        return 0.5, 0.5
    cx = (x + bw / 2) / pw
    cy = (y + bh / 2) / ph
    return min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)


def _fit_size(pw, ph):
    """Largest even-dimensioned box with the panel's aspect that fits the
    canvas. zoompan scales its window to `s`, so `s` must keep the input
    aspect or the art is stretched."""
    scale = min(W / pw, H / ph)
    fw = max(2, round(pw * scale) // 2 * 2)
    fh = max(2, round(ph * scale) // 2 * 2)
    return fw, fh


def _zoompan(motion, n, cx, cy, fw, fh):
    """Build a zoompan filter for one shot.

    Driven by `on` (output frame index) with d=1, so each looped input
    frame yields exactly one output frame. x/y are clamped to the input
    so a focus point near an edge can't push the window off the image.
    Moves are deliberately gentle: the art still carries its speech
    bubbles, and a hard zoom clips them.
    """
    n = max(n, 2)
    span = n - 1
    cxc = f"max(0,min(iw-iw/zoom,{cx:.4f}*iw-(iw/zoom)/2))"
    cyc = f"max(0,min(ih-ih/zoom,{cy:.4f}*ih-(ih/zoom)/2))"

    if motion == "zoom_face":
        z, x, y = f"min(1+0.18*on/{span},1.18)", cxc, cyc
    elif motion == "zoom_out":
        z, x, y = f"max(1.18-0.18*on/{span},1.0)", cxc, cyc
    elif motion == "slow_pan":
        z = "1.08"
        x = f"(iw-iw/zoom)*on/{span}"
        y = cyc
    elif motion == "shake":
        z = "1.06"
        x = f"max(0,min(iw-iw/zoom,{cxc}+14*sin(on*1.7)))"
        y = f"max(0,min(ih-ih/zoom,{cyc}+10*cos(on*2.3)))"
    else:  # hold, and anything unrecognised
        z, x, y = f"1+0.03*on/{span}", cxc, cyc

    return f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={fw}x{fh}:fps={FPS}"


def _render_shot(img_path, audio_path, frames, motion, cx, cy,
                 panel_size, out_path):
    dur = frames / FPS
    fw, fh = _fit_size(*panel_size)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-loop", "1", "-framerate", str(FPS), "-i", str(img_path)]
    if audio_path:
        cmd += ["-i", str(audio_path)]
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]

    # background: same art, cropped to fill, blurred and dimmed so the
    # fitted panel reads as the subject on any aspect ratio
    vf = (
        "[0:v]split=2[bg][fg];"
        f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},boxblur=32:2,eq=brightness=-0.22[bgb];"
        f"[fg]{_zoompan(motion, frames, cx, cy, fw, fh)}[fgz];"
        "[bgb][fgz]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
    )
    cmd += [
        "-filter_complex", vf,
        "-map", "[v]", "-map", "1:a",
        "-af", f"apad,atrim=0:{dur:.4f},asetpts=N/SR/TB",
        "-frames:v", str(frames),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-r", str(FPS), "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        "-ac", "2", str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _music_track(config, mood):
    """Pick the bed for an episode's music_mood, or None if disabled."""
    s = config.get("shorts", {})
    if not s.get("music_volume", 0):
        return None
    by_mood = s.get("music_by_mood") or {}
    name = by_mood.get(mood) or by_mood.get("default")
    if not name or not s.get("music_dir"):
        return None
    path = Path(s["music_dir"]).expanduser() / name
    if not path.exists():
        print(f"  WARNING: music track {path} missing, rendering without")
        return None
    return path


def run(config, workdir: Path):
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH")

    clean_path = workdir / "clean.json"
    if not clean_path.exists():
        raise SystemExit("no clean.json - run stage 3 (or 3p) first")
    panels = {p["id"]: p for p in json.loads(clean_path.read_text())}

    manifests = sorted((workdir / "manifests").glob("ep*.json"))
    if not manifests:
        raise SystemExit("no manifests in work/manifests - run stage 4 first")

    captions_on = config.get("shorts", {}).get("captions", True)

    out_dir = workdir / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = workdir / "renders" / "_clips"

    for mf in manifests:
        data = json.loads(mf.read_text())
        short_id = data.get("short_id", mf.stem)
        tmp = tmp_root / short_id
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

        clips = []
        estimated = 0
        for i, shot in enumerate(data.get("shots", [])):
            panel = panels.get(shot.get("panel"))
            if panel is None:
                print(f"  WARNING: {short_id} shot{i:03d} references unknown "
                      f"panel {shot.get('panel')!r}, skipping")
                continue

            img = workdir / panel["clean_file"]
            audio = shot.get("audio")
            audio_path = None
            if audio:
                p = Path(audio)
                p = p if p.is_absolute() else workdir.parent / p
                if p.exists():
                    audio_path = p
                else:
                    print(f"  WARNING: {short_id} shot{i:03d} audio missing "
                          f"({audio}), rendering silent")

            frames = shot.get("duration_frames")
            if not frames:
                frames = _est_frames(shot.get("line", ""))
                estimated += 1

            cx, cy = _focus_norm(panel)
            clip = tmp / f"shot{i:03d}.mp4"
            _render_shot(img, audio_path, frames,
                         shot.get("motion", "hold"), cx, cy,
                         panel["clean_size"], clip)
            clips.append(clip)
            print(f"  {short_id} shot{i:03d} {shot.get('motion','hold'):<9} "
                  f"{frames:>4}f  {panel['id']}")

        if not clips:
            print(f"{short_id}: no renderable shots, skipped")
            continue

        listing = tmp / "concat.txt"
        listing.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
        out_path = out_dir / f"{short_id}.mp4"

        if captions_on:
            # burn captions in one pass over the joined video: doing it
            # per-shot would re-encode every clip and still need this join
            joined = tmp / "joined.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                 "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)],
                check=True, capture_output=True, text=True)
            cue_list = captions.cues(data.get("shots", []), FPS)
            pngs, y = captions.render_pngs(cue_list, W, H, tmp / "cues")
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(joined)]
            for p in pngs:
                cmd += ["-i", str(p)]
            chain, prev = [], "0:v"
            for n, (a, b, _) in enumerate(cue_list):
                label = f"v{n}"
                chain.append(
                    f"[{prev}][{n+1}:v]overlay=0:{y}:"
                    f"enable='between(t,{a/FPS:.3f},{b/FPS:.3f})'[{label}]")
                prev = label
            cmd += ["-filter_complex", ";".join(chain),
                    "-map", f"[{prev}]", "-map", "0:a",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "copy", str(out_path)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        else:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                 "-safe", "0", "-i", str(listing), "-c", "copy", str(out_path)],
                check=True, capture_output=True, text=True)

        music = _music_track(config, data.get("music_mood"))
        if music:
            scored = tmp / "scored.mp4"
            out_path.rename(scored)
            dur = sum(s.get("duration_frames") or 0
                      for s in data.get("shots", [])) / FPS
            vol = config.get("shorts", {}).get("music_volume", 0.11)
            # bed sits under the narration, fading in and out so it never
            # starts or stops on a hard edge
            af = (f"[1:a]volume={vol},afade=t=in:st=0:d=2,"
                  f"afade=t=out:st={max(0, dur - 3):.2f}:d=3[m];"
                  # normalize=0: amix otherwise scales every input by
                  # 1/n, which would drop the narration ~3dB
                  f"[0:a][m]amix=inputs=2:duration=first:normalize=0:"
                  f"dropout_transition=0[a]")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(scored),
                 "-i", str(music), "-filter_complex", af,
                 "-map", "0:v", "-map", "[a]", "-c:v", "copy",
                 "-c:a", "aac", "-b:a", "160k", "-shortest", str(out_path)],
                check=True, capture_output=True, text=True)

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(out_path)],
            capture_output=True, text=True)
        total_s = float(probe.stdout.strip()) if probe.stdout.strip() else 0.0
        note = f" ({estimated} shots with estimated timing)" if estimated else ""
        print(f"{short_id}: {len(clips)} shots, {total_s:.1f}s -> "
              f"{out_path}{note}\n")
