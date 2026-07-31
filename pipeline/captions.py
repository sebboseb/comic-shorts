"""Burned-in captions for stage 7.

Timing comes from stage 6, not from speech-to-text. We sent each sentence
to the TTS engine and measured the wav it returned, so we already know
exactly what is said and exactly when. Running whisper over our own
output would only add transcription error -- and it errs precisely on the
proper nouns a comic depends on -- character and place names.

Long sentences are split into short on-screen chunks, with the sentence's
measured duration shared out across them by character count. That last
step is an approximation: within one sentence a chunk boundary can land a
few frames early or late. Sentence boundaries themselves are exact.
"""

from pathlib import Path

MAX_CHARS = 26          # short bursts; uppercase at 4.4% of height is wide
MAX_WORDS = 5


# a caption shouldn't end on one of these -- "...LOOKING FOR" then a cut
# reads as a mistake, and the eye has nowhere to rest
WEAK_TAIL = {
    "a", "an", "the", "and", "or", "but", "of", "for", "to", "in", "on",
    "at", "by", "with", "from", "into", "as", "is", "was", "his", "her",
    "its", "their", "this", "that", "he", "she", "they", "it",
}


def _chunk(text):
    """Break a sentence into caption-sized pieces on word boundaries.

    Prefers to break after a comma and never leaves a chunk ending on a
    function word, so each caption reads as a phrase rather than a slice.
    """
    words = text.split()
    if not words:
        return []

    out, cur = [], []
    for w in words:
        trial = " ".join(cur + [w])
        full = cur and (len(trial) > MAX_CHARS or len(cur) >= MAX_WORDS)
        if full:
            # pull trailing function words forward into the next chunk
            while len(cur) > 1 and cur[-1].strip(".,!?;:").lower() in WEAK_TAIL:
                w, cur = cur[-1] + " " + w, cur[:-1]
            out.append(" ".join(cur))
            cur = w.split()
        else:
            cur.append(w)
            # a comma is a natural rest: break here if we already have enough
            if cur[-1].endswith(",") and len(cur) >= 3:
                out.append(" ".join(cur))
                cur = []
    if cur:
        if out and len(cur) == 1 and cur[0].strip(".,!?;:").lower() in WEAK_TAIL:
            out[-1] += " " + cur[0]      # never a lone function word
        else:
            out.append(" ".join(cur))
    return out


def _word_spans(piece, start, span):
    """Apportion a chunk's frames across its words by character weight.

    The same approximation the chunks themselves use: within one sentence
    the wav is one take, so word boundaries are estimated, not measured.
    Good enough for a karaoke highlight, whose job is to feel synced, not
    to be a phoneme aligner. Returns [(w_start, w_end), ...] per word.
    """
    words = piece.split()
    weights = [len(w) + 1 for w in words]
    total = sum(weights) or 1
    spans, spent = [], 0
    for i, wt in enumerate(weights):
        share = (span - spent if i == len(words) - 1
                 else round(span * wt / total))
        spans.append((start + spent, start + spent + max(share, 0)))
        spent += share
    return spans


def cues(shots, fps):
    """(start_frame, end_frame, text, speaker, word_spans) over the episode
    timeline. word_spans has one (start, end) per word of text, for the
    karaoke highlight; consumers that unpack only the first four fields
    keep working.

    The shot's duration_frames is authoritative for where the video is:
    it covers the whole wav, including the trailing shot-gap silence that
    belongs to no sentence. Advancing the clock by the sum of segment
    frames instead loses that gap every shot and drifts the captions
    steadily earlier, so re-anchor to the shot boundary each time.
    """
    out = []
    clock = 0
    for shot in shots:
        dur = shot.get("duration_frames") or 0
        segs = shot.get("segments")
        if not segs and shot.get("line"):
            # no per-sentence timing (silent shot, or pre-segments manifest)
            segs = [{"text": shot["line"], "frames": dur}]

        t = clock
        for seg in segs or []:
            frames = seg.get("frames", 0)
            # span the words, not the silence after them
            speech = seg.get("speech_frames") or frames
            pieces = _chunk(seg["text"])
            if pieces:
                total = sum(len(p) for p in pieces) or 1
                spent = 0
                for i, piece in enumerate(pieces):
                    share = (speech - spent if i == len(pieces) - 1
                             else round(speech * len(piece) / total))
                    if share > 0:
                        out.append((t + spent, t + spent + share, piece,
                                    shot.get("speaker") or "narrator",
                                    _word_spans(piece, t + spent, share)))
                    spent += share
            t += frames
        clock += dur  # never the segment sum
    return out


FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

DEFAULTS = {
    "font": None,             # explicit path, else first of FONT_CANDIDATES
    "size_pct": 0.046,        # of frame height
    "uppercase": True,
    "outline_ratio": 9,       # size // ratio
    "shadow": True,
    "color": "#FFFFFF",
    "highlight_color": None,  # set (e.g. "#FFD54A") for word-level karaoke:
                              # the word being spoken renders in this colour,
                              # one overlay per word (how Comicfy captions work)
    "speaker_colors": {},     # {"Ada": "#FF80AB", ...}; narrator falls to color
    # where the text block sits. The reference channels all keep captions in
    # the middle band of the frame, where the thumb doesn't cover them and
    # the eye is already parked; "bottom" is the classic subtitle position.
    "position": "center",     # center | bottom
    "center_pct": 0.55,       # position=center: text bottom lands here
    "bottom_pct": 0.10,       # position=bottom: margin under the text
}


def _hex(c):
    c = (c or "#FFFFFF").lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) + (255,)


def _font(size, path=None):
    from PIL import ImageFont
    for candidate in ([path] if path else []) + FONT_CANDIDATES:
        if candidate and Path(candidate).expanduser().exists():
            try:
                return ImageFont.truetype(str(Path(candidate).expanduser()), size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def render_pngs(cue_list, width, height, out_dir, style=None):
    """One transparent strip per cue.

    This ffmpeg build ships without libass *and* without drawtext, so the
    subtitles/drawtext filters don't exist. Pillow is already a project
    dependency, so draw the captions ourselves and composite them with
    `overlay`, which is available.
    """
    from PIL import Image, ImageDraw

    st = {**DEFAULTS, **(style or {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    size = round(height * st["size_pct"])
    font = _font(size, st.get("font"))
    strip_h = round(height * 0.22)
    if st.get("position", "center") == "center":
        # text is drawn bottom-aligned inside the strip, so placing the
        # strip's bottom edge at center_pct puts the block mid-frame
        margin_b = height - round(height * st["center_pct"])
    else:
        margin_b = round(height * st["bottom_pct"])
    stroke = max(4, size // st["outline_ratio"])
    base = _hex(st["color"])
    by_speaker = {k: _hex(v) for k, v in (st.get("speaker_colors") or {}).items()}

    paths = []
    for i, cue in enumerate(cue_list):
        text, speaker = cue[2], (cue[3] if len(cue) > 3 else "narrator")
        fill = by_speaker.get(speaker, base)
        img = _cue_image(text, width, strip_h, font, size, st, stroke, fill)
        p = out_dir / f"cue{i:04d}.png"
        img.save(p)
        paths.append(p)
    return paths, height - strip_h - margin_b


def _cue_image(text, width, strip_h, font, size, st, stroke, fill,
               highlight_idx=None, highlight=None):
    """Draw one caption strip. With highlight_idx, that word (by position
    in text.split()) renders in the highlight colour - drawing word by
    word with a running x offset, so the layout is identical across all
    of a cue's karaoke variants and only the colour changes."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (width, strip_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # uppercase reads as a title card rather than a subtitle
    shown = text.upper() if st["uppercase"] else text
    lines = _layout(shown.split(), draw, font, width - 110)
    line_h = round(size * 1.22)
    y = strip_h - len(lines) * line_h
    space_w = draw.textlength(" ", font=font)
    word_i = 0
    for line_words in lines:
        line = " ".join(line_words)
        x = (width - draw.textlength(line, font=font)) / 2
        if st["shadow"]:
            off = max(2, size // 14)
            draw.text((x + off, y + off), line, font=font,
                      fill=(0, 0, 0, 150), stroke_width=stroke,
                      stroke_fill=(0, 0, 0, 150))
        for w in line_words:
            colour = (highlight if highlight is not None
                      and word_i == highlight_idx else fill)
            draw.text((x, y), w, font=font, fill=colour,
                      stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
            x += draw.textlength(w, font=font) + space_w
            word_i += 1
        y += line_h
    return img


def render_karaoke(cue_list, width, height, out_dir, style=None):
    """Word-level karaoke variant of render_pngs: one PNG per WORD, each
    shown for that word's span from cues(). Same strip geometry as
    render_pngs, so the two are drop-in alternatives for stage 7.

    Returns ([(path, start_frame, end_frame)], y).
    """
    st = {**DEFAULTS, **(style or {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    size = round(height * st["size_pct"])
    font = _font(size, st.get("font"))
    strip_h = round(height * 0.22)
    if st.get("position", "center") == "center":
        margin_b = height - round(height * st["center_pct"])
    else:
        margin_b = round(height * st["bottom_pct"])
    stroke = max(4, size // st["outline_ratio"])
    base = _hex(st["color"])
    highlight = _hex(st["highlight_color"] or "#FFD54A")
    by_speaker = {k: _hex(v) for k, v in (st.get("speaker_colors") or {}).items()}

    entries = []
    for i, cue in enumerate(cue_list):
        start, end, text = cue[0], cue[1], cue[2]
        speaker = cue[3] if len(cue) > 3 else "narrator"
        spans = cue[4] if len(cue) > 4 else None
        fill = by_speaker.get(speaker, base)
        words = text.split()
        if not spans or len(spans) != len(words):
            spans = _word_spans(text, start, end - start)
        for j, (w0, w1) in enumerate(spans):
            if w1 <= w0:
                continue
            img = _cue_image(text, width, strip_h, font, size, st, stroke,
                             fill, highlight_idx=j, highlight=highlight)
            p = out_dir / f"cue{i:04d}_w{j:02d}.png"
            img.save(p)
            entries.append((p, w0, w1))
    return entries, height - strip_h - margin_b


def render_badge(text, height, out_path, style=None):
    """The episode tag the reference channels put top-left ("Part - 33"):
    a small rounded amber chip with dark text, present for the whole video.
    It is what tells a mid-scroll viewer this is a series worth following.

    Returns (path, x, y) for the overlay.
    """
    from PIL import Image, ImageDraw

    st = {"chip": "#F5B301", "text": "#1A1A1A", **(style or {})}
    size = round(height * 0.017)
    font = _font(size, st.get("font"))
    pad_x, pad_y = round(size * 0.9), round(size * 0.55)

    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    tw = round(probe.textlength(text, font=font))
    w, h = tw + 2 * pad_x, size + 2 * pad_y
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 4,
                           fill=_hex(st["chip"]))
    draw.text((pad_x, pad_y - round(size * 0.12)), text, font=font,
              fill=_hex(st["text"]))
    img.save(out_path)
    margin = round(height * 0.018)
    return out_path, margin, margin


def _layout(words, draw, font, max_w):
    """Greedy wrap, kept as word lists so callers can address single words."""
    lines, cur = [], []
    for word in words:
        trial = " ".join(cur + [word])
        if cur and draw.textlength(trial, font=font) > max_w:
            lines.append(cur)
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append(cur)
    return lines
