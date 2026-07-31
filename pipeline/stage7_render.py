"""Stage 7: render manifests to vertical video with ffmpeg.

One clip per shot: a frame-shaped window cropped around the panel's
`focus_box` fills the whole 1080x1920 canvas (frame_mode=fill, how the
reference channels frame everything), with a gentle drift driven by the
manifest's `motion` intent. Consecutive shots on the same panel re-frame
(tighter, wider) so every beat reads as a cut. frame_mode=fit keeps the
old letterbox-over-blur look. Clips carry their own audio (the stage-6
wav, or generated silence), so concatenating them keeps A/V in sync
without a global offset.

Captions are burned in from stage 6's per-sentence timings (see
captions.py), and a music bed is mixed under the narration per the
episode's music_mood. Still scrappy: no transitions, no sfx stings.
Remotion is the plan of record for the polished version.

Runs without stage 6: shots with no audio get a duration estimated from
the line length, so you can eyeball the motion before spending TTS time.

Output: work/renders/epNN.mp4
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

from pipeline import captions

FPS = 30
W, H = 1080, 1920
SILENT_FRAMES = round(1.2 * FPS)
WORDS_PER_SEC = 2.5      # matches the pacing stage 4 writes to
TAIL_FRAMES = 8          # breath after a line so cuts don't clip the audio
PRESCALE = 2             # upscale source before zoompan (smaller rounding step)
SUPERSAMPLE = 2          # render the move at Nx target, then average down
DRIFT_RATE = 0.055       # fraction of frame width the picture may travel per
                         # second; a full traverse per shot reads as swooping

# Re-framing for consecutive shots on the SAME panel. Stage 4 deliberately
# reuses a panel across beats; rendering those beats with the same window
# makes them one long static shot on screen, and the felt cut rate collapses
# (the reference channels re-frame on every beat: wide, then the face, then
# wider again). Each entry is (tightness multiplier, max window fraction of
# the panel): the multiplier varies subject room, and the cap guarantees a
# visible punch-in even when the focus box is so large that any tightness
# clamps to the whole panel. Punch-ins are deliberately moderate: harder caps
# (0.7/0.55) made dense panels unreadable - "hard to tell what's going on".
REFRAME = ((1.0, 1.0), (0.75, 0.82), (1.5, 1.0), (0.9, 0.68))

# Wide panels that can't fill the frame render as a full-height band this
# many times wider than tall, letterboxed over blur: wide enough to read
# the scene, capped so the art stays large (a 1.5:1 band fills ~38% of the
# frame's height vs ~14% for a whole 4:1 panel).
BAND_ASPECT = 1.5


def _est_frames(line):
    if not line:
        return SILENT_FRAMES
    words = len(line.split())
    return max(SILENT_FRAMES, round(words / WORDS_PER_SEC * FPS) + TAIL_FRAMES)


# The reference channels cut roughly every 2 seconds: one narration beat,
# one visual change, even when the picture is the same panel re-framed.
# A shot longer than MAX_SHOT_FRAMES is therefore split into sub-clips -
# same wav, sliced - and each sub-clip re-frames via the REFRAME schedule,
# so a long line reads as several cuts instead of one static hold.
MAX_SHOT_FRAMES = 78          # ~2.6s; splits aim for ~2.2s pieces
SPLIT_TARGET = 66


def _split_points(shot, frames):
    """Frame offsets [(start, n), ...] carving a shot into sub-clips.

    Cuts prefer sentence boundaries (stage 6's measured segments) when one
    lands near the even split point; a cut inside a word is masked by the
    caption staying put, but a cut on a sentence gap looks intentional.
    """
    if frames <= round(MAX_SHOT_FRAMES * 1.35):
        return [(0, frames)]
    n = max(2, round(frames / SPLIT_TARGET))
    targets = [round(frames * i / n) for i in range(1, n)]
    bounds = []
    acc = 0
    for seg in (shot.get("segments") or [])[:-1]:
        acc += seg.get("frames", 0)
        bounds.append(acc)
    cuts = []
    for t in targets:
        near = [b for b in bounds if abs(b - t) <= 18 and b not in cuts]
        cut = min(near, key=lambda b: abs(b - t)) if near else t
        if 12 < cut < frames - 12 and (not cuts or cut - cuts[-1] > 24):
            cuts.append(cut)
    pieces, prev = [], 0
    for c in cuts + [frames]:
        pieces.append((prev, c - prev))
        prev = c
    return pieces


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
        # was a sine wobble on both axes, which read as a mechanical shudder
        # rather than an impact. Hold steady, slightly tighter than a hold.
        z, x, y = "1.05", cxc, cyc
    else:  # hold, and anything unrecognised
        # genuinely static: a slow creep here bought nothing visually and was
        # pure jitter, because any changing zoom re-rounds the window origin
        z, x, y = "1", cxc, cyc

    return f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={fw}x{fh}:fps={FPS}"


def _band_window(pw, ph, cx, band_aspect):
    """Full-height band around the subject for a wide panel: wider than the
    frame (context survives) but capped at band_aspect (the subject stays
    big). Returns (x, w, ncx): the crop and the focus centre within it."""
    bw = min(pw, max(2, round(ph * band_aspect)))
    x = min(max(round(cx * pw - bw / 2), 0), pw - bw)
    ncx = (cx * pw - x) / bw if bw else 0.5
    return x, bw, min(max(ncx, 0.0), 1.0)


def _part_number(short_id):
    """ep01 -> 1, for the series badge. None when the id carries no number."""
    m = re.search(r"(\d+)", short_id or "")
    return int(m.group(1)) if m else None


def _frame_window(pw, ph, focus, tightness, max_frac=1.0):
    """A frame-shaped window sized to the SUBJECT, not to the panel.

    The previous approach cropped a window the panel's full height, which
    filled the screen but left the subject exactly as small as it was in the
    original art -- the median focus box is 17% of its panel, so nothing was
    ever actually framed. Sizing the window from the focus box instead means
    every shot arrives on something.

    Returns (x, y, w, h) in panel pixels, always frame-aspect and always
    inside the panel, so scaling it to the canvas fills without letterboxing.
    """
    aspect = W / H
    fx, fy, fw, fh = focus if focus else (0, 0, pw, ph)
    fw, fh = max(fw, 8), max(fh, 8)

    # give the subject room, then force the canvas aspect by growing the
    # deficient axis (never shrinking, so the subject stays fully inside)
    tw, th = fw * tightness, fh * tightness
    if tw / th < aspect:
        tw = th * aspect
    else:
        th = tw / aspect

    # a window can't exceed the panel (or, on a re-framed repeat, max_frac of
    # it); clamping either axis re-derives the other
    lw, lh = pw * max_frac, ph * max_frac
    if tw > lw:
        tw, th = lw, lw / aspect
    if th > lh:
        th, tw = lh, lh * aspect
    if tw > lw:                     # panel narrower than one frame of its height
        tw, th = lw, lw / aspect

    cx, cy = fx + fw / 2, fy + fh / 2
    x = min(max(cx - tw / 2, 0), max(0, pw - tw))
    y = min(max(cy - th / 2, 0), max(0, ph - th))
    return round(x), round(y), round(tw), round(th)


def _framed_chain(pw, ph, focus, motion, dur, tightness, max_frac=1.0):
    """Crop the subject-framed window, drift gently inside it, fill the frame."""
    x, y, w, h = _frame_window(pw, ph, focus, tightness, max_frac)
    s = PRESCALE
    sx, sy, sw, sh = x * s, y * s, w * s, h * s

    # room to move without leaving the panel
    room_x = max(0, pw * s - sw)
    budget = min(room_x, round(DRIFT_RATE * sw * dur))
    if motion in ("hold", "shake") or budget < 8:
        xexpr = str(sx)
    else:
        start = sx - budget if sx - budget >= 0 else sx + budget
        start = max(0, min(room_x, start))
        xexpr = f"{start}+({sx}-{start})*min(1,t/{max(dur, 0.1):.3f})"

    return (f"scale=iw*{s}:ih*{s}:flags=lanczos,"
            f"crop={sw}:{sh}:x='{xexpr}':y={sy},"
            f"scale={W}:{H}:flags=lanczos")


def _render_shot(img_path, audio_path, frames, motion, cx, cy,
                 panel_size, out_path,
                 focus=None, tightness=2.4, frame_mode="fill", max_frac=1.0,
                 min_coverage=0.5, audio_offset=0.0):
    dur = frames / FPS
    pw, ph = panel_size
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-loop", "1", "-framerate", str(FPS), "-i", str(img_path)]
    if audio_path:
        # sub-clips of a split shot each carry their slice of the one wav
        cmd += (["-ss", f"{audio_offset:.4f}"] if audio_offset else []) \
            + ["-i", str(audio_path)]
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]

    if frame_mode == "fill":
        # subject-framed crop: fills the canvas but discards whatever falls
        # outside the frame-shaped window, including neighbouring panel art.
        # On a WIDE panel that window is a narrow slice of a busy scene -
        # "hard to tell what's going on" - so when it would show less than
        # min_coverage of the panel's width, letterbox a full-height band
        # around the subject instead (the reference channels run wide
        # panels as bands, faces full-frame). The band is capped at
        # BAND_ASPECT so it stays tall on screen; whole-panel fit made a
        # 4:1 battle panel a sliver.
        w = _frame_window(pw, ph, focus, tightness, max_frac)[2]
        if w / pw < min_coverage:
            bx, bw, ncx = _band_window(pw, ph, cx, BAND_ASPECT)
            vf = _fit_chain(*_fit_size(bw, ph), frames, motion, ncx, cy,
                            crop=(bw, ph, bx, 0))
        else:
            vf = (f"[0:v]{_framed_chain(pw, ph, focus, motion, dur, tightness, max_frac)},"
                  f"format=yuv420p[v]")
    else:
        # fit: the whole panel, letterboxed over a blurred copy of itself, with
        # a Ken Burns move aimed at the focus. Keeps the panel's composition
        # intact, which is what the art was drawn for.
        vf = _fit_chain(*_fit_size(pw, ph), frames, motion, cx, cy)

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


def _fit_chain(fw, fh, frames, motion, cx, cy, crop=None):
    # zoompan truncates its window origin to whole INPUT pixels every frame,
    # so a continuously changing zoom makes the image twitch by a fraction of
    # an output pixel -- which comic halftone dots turn into visible shimmer.
    # Two mitigations: pre-scale the source so one input pixel is a smaller
    # step, and render the move at SS x target then average it back down.
    ss_w, ss_h = fw * SUPERSAMPLE, fh * SUPERSAMPLE

    # crop: letterbox a region of the panel rather than all of it
    # (wide-panel band), for both the subject and the blurred backdrop
    src = ("[0:v]" if crop is None else
           f"[0:v]crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]},")
    # background: same art, cropped to fill, blurred and dimmed so the
    # fitted panel reads as the subject on any aspect ratio
    vf = (
        f"{src}split=2[bg][fg];"
        f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},boxblur=32:2,eq=brightness=-0.22[bgb];"
        f"[fg]scale=iw*{PRESCALE}:ih*{PRESCALE}:flags=lanczos,"
        f"{_zoompan(motion, frames, cx, cy, ss_w, ss_h)},"
        f"scale={fw}:{fh}:flags=lanczos[fgz];"
        "[bgb][fgz]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
    )
    return vf


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
    part_badge = config.get("shorts", {}).get("part_badge", False)
    global DRIFT_RATE
    DRIFT_RATE = config.get("shorts", {}).get("drift_rate", DRIFT_RATE)
    tightness = config.get("shorts", {}).get("frame_tightness", 2.4)
    frame_mode = config.get("shorts", {}).get("frame_mode", "fill")
    min_coverage = config.get("shorts", {}).get("fill_min_coverage", 0.5)

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
        prev_panel, repeat = None, 0
        for i, shot in enumerate(data.get("shots", [])):
            panel = panels.get(shot.get("panel"))
            if panel is None:
                print(f"  WARNING: {short_id} shot{i:03d} references unknown "
                      f"panel {shot.get('panel')!r}, skipping")
                continue
            # the render loop advances `repeat` once per sub-clip; a new
            # panel resets the re-frame schedule
            if shot.get("panel") != prev_panel:
                repeat = 0
            prev_panel = shot.get("panel")

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
            fb, size, csize = (panel.get("focus_box"), panel.get("size"),
                               panel["clean_size"])
            focus = None
            if fb and size and size[0] and size[1]:
                k = csize[0] / size[0], csize[1] / size[1]
                focus = (fb[0] * k[0], fb[1] * k[1], fb[2] * k[0], fb[3] * k[1])
            pieces = _split_points(shot, frames)
            for j, (off, nf) in enumerate(pieces):
                clip = tmp / f"shot{i:03d}_{j}.mp4"
                tm, max_frac = REFRAME[repeat % len(REFRAME)]
                _render_shot(img, audio_path, nf,
                             shot.get("motion", "hold"), cx, cy,
                             csize, clip, focus,
                             tightness * tm, frame_mode, max_frac,
                             min_coverage, audio_offset=off / FPS)
                clips.append(clip)
                repeat += 1  # next sub-clip (or same-panel shot) re-frames
            print(f"  {short_id} shot{i:03d} {shot.get('motion','hold'):<9} "
                  f"{frames:>4}f  {panel['id']}"
                  + (f"  ({len(pieces)} cuts)" if len(pieces) > 1 else ""))

        if not clips:
            print(f"{short_id}: no renderable shots, skipped")
            continue

        listing = tmp / "concat.txt"
        listing.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
        out_path = out_dir / f"{short_id}.mp4"

        # burned-in overlays: (png, x, y, enable-window or None for always).
        # One pass over the joined video: doing it per-shot would re-encode
        # every clip and still need the join.
        overlays = []
        n_part = _part_number(short_id)
        if part_badge and n_part is not None:
            badge = captions.render_badge(
                f"Part - {n_part}", H, tmp / "badge.png",
                config.get("shorts", {}).get("badge_style"))
            overlays.append(badge + (None,))
        if captions_on:
            cue_list = captions.cues(data.get("shots", []), FPS)
            style = config.get("shorts", {}).get("caption_style") or {}
            if style.get("highlight_color"):
                # word-level karaoke: one overlay per word, not per cue
                entries, y = captions.render_karaoke(cue_list, W, H,
                                                     tmp / "cues", style)
                overlays += [(p, 0, y, (a / FPS, b / FPS))
                             for p, a, b in entries]
            else:
                pngs, y = captions.render_pngs(cue_list, W, H, tmp / "cues",
                                               style)
                overlays += [(p, 0, y, (a / FPS, b / FPS))
                             for p, (a, b, *_) in zip(pngs, cue_list)]

        if overlays:
            joined = tmp / "joined.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                 "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)],
                check=True, capture_output=True, text=True)
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(joined)]
            for p, _, _, _ in overlays:
                cmd += ["-i", str(p)]
            chain, prev = [], "0:v"
            for n, (_, x, y, window) in enumerate(overlays):
                enable = ("" if window is None else
                          f":enable='between(t,{window[0]:.3f},{window[1]:.3f})'")
                chain.append(f"[{prev}][{n+1}:v]overlay={x}:{y}{enable}[v{n}]")
                prev = f"v{n}"
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
