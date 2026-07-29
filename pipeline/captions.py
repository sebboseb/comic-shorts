"""Burned-in captions for stage 7.

Timing comes from stage 6, not from speech-to-text. We sent each sentence
to the TTS engine and measured the wav it returned, so we already know
exactly what is said and exactly when. Running whisper over our own
output would only add transcription error -- and it errs precisely on the
proper nouns a comic depends on (Wad, Ada, Dick Warren).

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


def cues(shots, fps):
    """(start_frame, end_frame, text) over the episode timeline.

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
            pieces = _chunk(seg["text"])
            if pieces:
                total = sum(len(p) for p in pieces) or 1
                spent = 0
                for i, piece in enumerate(pieces):
                    share = (frames - spent if i == len(pieces) - 1
                             else round(frames * len(piece) / total))
                    if share > 0:
                        out.append((t + spent, t + spent + share, piece))
                    spent += share
            t += frames
        clock += dur  # never the segment sum
    return out


FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _font(size):
    from PIL import ImageFont
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def render_pngs(cue_list, width, height, out_dir):
    """One transparent strip per cue.

    This ffmpeg build ships without libass *and* without drawtext, so the
    subtitles/drawtext filters don't exist. Pillow is already a project
    dependency, so draw the captions ourselves and composite them with
    `overlay`, which is available.
    """
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    size = round(height * 0.044)   # punchy, not a subtitle strip
    font = _font(size)
    strip_h = round(height * 0.20)
    margin_b = round(height * 0.10)
    stroke = max(4, size // 9)

    paths = []
    for i, (_, _, text) in enumerate(cue_list):
        img = Image.new("RGBA", (width, strip_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # uppercase reads as a title card rather than a subtitle
        lines = _wrap(text.upper(), draw, font, width - 110)
        line_h = round(size * 1.25)
        y = strip_h - len(lines) * line_h
        for line in lines:
            w = draw.textlength(line, font=font)
            draw.text(((width - w) / 2, y), line, font=font, fill=(255, 255, 255, 255),
                      stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
            y += line_h
        p = out_dir / f"cue{i:04d}.png"
        img.save(p)
        paths.append(p)
    return paths, height - strip_h - margin_b


def _wrap(text, draw, font, max_w):
    words, lines, cur = text.split(), [], []
    for word in words:
        trial = " ".join(cur + [word])
        if cur and draw.textlength(trial, font=font) > max_w:
            lines.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))
    return lines
