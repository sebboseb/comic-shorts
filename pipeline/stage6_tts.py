"""Stage 6: TTS via the Voicebox desktop app's local API.

Fills `audio` and `duration_frames` on every shot in every manifest.
- speaker -> Voicebox profile via VOICE_MAP (default: everything -> "Narrator")
- silent shots (line == "") get audio=None, duration_frames = SILENT_S * FPS
- idempotent: shots with audio already set are skipped
- wavs land in work/audio/<short_id>/shotNNN.wav

Requires the Voicebox app to be open. Run AFTER the stage-5 review gate.
"""
import io, json, re, struct, subprocess, sys, tempfile, time, urllib.request, wave
from pathlib import Path

FPS = 30
SILENT_S = 1.2
DEFAULT_PROFILE = "Narrator"
# fallback only - profiles carry their own default_engine. kokoro is retired
# as a narrator voice (flat delivery, ignores instruct); qwen_custom_voice is
# voicebox's default engine and honors delivery direction.
ENGINE = "qwen_custom_voice"
# engines that actually act a delivery direction. kokoro accepts `instruct`
# and ignores it; qwen visibly changes pace and register (a "whispered,
# afraid" reading of the same line ran 69% longer than a plain one).
INSTRUCT_ENGINES = {"qwen", "qwen_custom_voice", "chatterbox",
                    "chatterbox_turbo", "luxtts", "tada"}
PAUSE_MS = 600          # silence inserted at . ! ?
ELLIPSIS_PAUSE_MS = 850  # ... trails off, give it room
SHOT_GAP_MS = 450        # beat after each line, so cuts don't clip speech
SPEECH_TEMPO = 1.0       # >1 speeds up the words without touching the pauses
MIN_SEG_CHARS = 18       # below this, an autoregressive engine loses the plot:
                         # "Bad." alone came back as 6.7s of looped breath
VOICE_MAP = {}  # e.g. {"Mark": "MarkVoice", "narrator": "Narrator"}

def _find_port():
    out = subprocess.run(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"],
                         capture_output=True, text=True).stdout
    ports = {int(l.rsplit(":", 1)[-1].split()[0])
             for l in out.splitlines() if "127.0.0.1:" in l}
    for p in sorted(ports):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{p}/", timeout=1) as r:
                if "voicebox API" in r.read(200).decode(errors="ignore"):
                    return p
        except Exception:
            pass
    sys.exit("Voicebox API not found - open the app first.")

def _api(port, method, path, body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        raw = r.read()
    try:
        return json.loads(raw)
    except ValueError:
        return raw

def _profiles(port):
    """name -> (profile_id, engine). Profiles are engine-bound in Voicebox:
    generating a chatterbox profile with engine=kokoro is a 400, so carry
    each profile's own default_engine rather than assuming one."""
    r = _api(port, "GET", "/profiles")
    plist = r if isinstance(r, list) else r.get("profiles", [])
    return {p.get("name", "").lower():
            (p.get("id") or p.get("profile_id"),
             p.get("default_engine") or ENGINE)
            for p in plist}

def _speakable(text):
    """Comic lettering is ALL CAPS, and TTS reads a short all-caps token as
    an initialism -- "MY PEOPLE" comes out "em-why people". Fold shouty
    lines to sentence case for synthesis only; the manifest keeps the
    verbatim comic text for the review page.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return text
    if sum(c.isupper() for c in letters) / len(letters) < 0.6:
        return text  # already mixed case, leave it alone

    out = text.lower()
    # re-capitalise the first letter of each sentence
    out = re.sub(r"(^|[.!?]\s+|\.\.\.\s*)([a-z])",
                 lambda m: m.group(1) + m.group(2).upper(), out)
    # the pronoun I and its contractions
    out = re.sub(r"\bi\b", "I", out)
    out = re.sub(r"\bi'(m|ll|ve|d)\b", lambda m: "I'" + m.group(1), out)
    return out


def _sentences(text, pause_ms=None, ellipsis_ms=None):
    """Split a line into (sentence, pause_ms_after) pairs.

    Kokoro gives ~420ms at a full stop regardless of context, which reads
    as rushed for narration, and it has no pause control. Synthesising
    sentence by sentence and rejoining with our own silence is the only
    way to set the beat. Comic punctuation is messy ("STRANGER!.. WE"),
    so treat any run of terminal marks as one boundary and let a trailing
    ellipsis buy a longer beat.
    """
    parts = re.split(r"([.!?][.!?]*)", text)
    out = []
    for i in range(0, len(parts) - 1, 2):
        body, mark = parts[i].strip(), parts[i + 1]
        if not body:
            continue
        pause = (ellipsis_ms or ELLIPSIS_PAUSE_MS) if len(mark) > 1 \
            or "…" in mark else (pause_ms or PAUSE_MS)
        out.append((body + mark, pause))
    tail = parts[-1].strip()
    if tail:
        out.append((tail, pause_ms or PAUSE_MS))
    # Merge fragments too short to synthesise on their own. Splitting exists
    # to control the pause between sentences, but a one-word sentence sent
    # alone makes the model stutter or hallucinate, which is far worse than
    # inheriting the engine's own shorter internal pause.
    merged = []
    for seg, pause in out:
        if merged and len(merged[-1][0]) < MIN_SEG_CHARS:
            prev, _ = merged.pop()
            merged.append((prev + " " + seg, pause))
        else:
            merged.append((seg, pause))
    if len(merged) > 1 and len(merged[-1][0]) < MIN_SEG_CHARS:
        prev, _ = merged[-2]
        merged[-2:] = [(prev + " " + merged[-1][0], merged[-1][1])]
    out = merged

    if out:
        out[-1] = (out[-1][0], 0)  # no trailing pad; shot joins handle it
    return out or [(text, 0)]


def _trim(frames, sw, ch, sr, keep_ms=60):
    """Strip leading/trailing near-silence so joined pauses are exact."""
    n = len(frames) // (sw * ch)
    if sw != 2 or n == 0:
        return frames
    mono = struct.unpack(f"<{n * ch}h", frames)[::ch]
    peak = max((abs(x) for x in mono), default=0)
    if peak == 0:
        return frames
    floor = peak * 0.03
    first = next((i for i, x in enumerate(mono) if abs(x) > floor), 0)
    last = n - next((i for i, x in enumerate(reversed(mono)) if abs(x) > floor), 0)
    keep = int(sr * keep_ms / 1000)
    a = max(0, first - keep) * sw * ch
    b = min(n, last + keep) * sw * ch
    return frames[a:b]


def _retempo(raw, tempo):
    """Speed the words up without shifting pitch.

    Applied to the spoken audio only, before pauses are inserted, so the
    beats between sentences stay exactly as configured rather than
    shrinking along with the speech.
    """
    if abs(tempo - 1.0) < 0.01:
        return raw
    # ffmpeg cannot seek back to fix the RIFF size when writing wav to a pipe,
    # so it emits a placeholder length that the wave module rejects. Write a
    # real file instead.
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in.wav", Path(d) / "out.wav"
        src.write_bytes(raw)
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src),
             "-filter:a", f"atempo={tempo:.3f}", str(dst)],
            capture_output=True)
        if r.returncode == 0 and dst.exists() and dst.stat().st_size > 1000:
            return dst.read_bytes()
    return raw


def _synth_one(port, profile_id, text, engine, instruct=None):
    body = {"profile_id": profile_id, "text": text,
            "language": "en", "engine": engine}
    if instruct and engine in INSTRUCT_ENGINES:
        body["instruct"] = instruct
    gen = _api(port, "POST", "/generate", body)
    gen_id = gen.get("id") or gen.get("generation_id")
    if not gen_id:
        raise RuntimeError(f"bad /generate response: {json.dumps(gen)[:200]}")
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            audio = _api(port, "GET", f"/audio/{gen_id}")
            if isinstance(audio, (bytes, bytearray)) and len(audio) > 1000:
                return bytes(audio)
        except Exception:
            pass
        time.sleep(1.5)
    raise RuntimeError("synthesis timed out")


def _respell(text, mapping):
    """TTS-only pronunciation respellings (config shorts.tts_pronounce).
    Applied to what the engine hears, never to segments, so captions keep
    the comic's true spelling. Exists because qwen degenerates into vocal
    noise on some proper nouns (House Fiyero -> a laughing fit)."""
    for word, spoken in (mapping or {}).items():
        text = text.replace(word, spoken)
    return text


def _synth(port, profile_id, text, out_path: Path, engine=ENGINE,
           pace=None, instruct=None, respell=None):
    """Synthesise a line sentence by sentence, joined with explicit pauses.

    Returns the per-sentence segments with their frame counts, which is
    ground-truth caption timing: we know what each sentence says because
    we sent it, and how long it runs because we measured the wav. No
    speech-to-text round trip, so proper nouns can't be mangled.
    """
    pieces, params = [], None
    pace = pace or {}
    tempo = pace.get("speech_tempo", SPEECH_TEMPO)
    segments = []
    for sentence, pause_ms in _sentences(
            _speakable(text), pace.get("pause_ms"), pace.get("ellipsis_pause_ms")):
        raw = _retempo(_synth_one(port, profile_id,
                                  _respell(sentence, respell),
                                  engine, instruct),
                       tempo)
        with wave.open(io.BytesIO(raw), "rb") as w:
            params = params or w.getparams()
            frames = w.readframes(w.getnframes())
        sw, ch, sr = params.sampwidth, params.nchannels, params.framerate
        body = _trim(frames, sw, ch, sr)
        pieces.append(body)
        gap_bytes = int(sr * pause_ms / 1000) * sw * ch if pause_ms else 0
        if gap_bytes:
            pieces.append(b"\x00" * gap_bytes)
        # `frames` is what the video advances by; `speech_frames` is how long
        # the words actually last. Captions must be apportioned over the
        # latter, or every chunk after the first drifts into the pause.
        speech_samples = len(body) // (sw * ch)
        seg_samples = (len(body) + gap_bytes) // (sw * ch)
        segments.append({"text": sentence,
                         "frames": round(seg_samples / sr * FPS),
                         "speech_frames": round(speech_samples / sr * FPS)})

    # trailing beat: _trim leaves only 60ms, so without this the next
    # shot's first word lands almost on top of this one's last
    sw, ch, sr = params.sampwidth, params.nchannels, params.framerate
    gap = pace.get("shot_gap_ms", SHOT_GAP_MS)
    pieces.append(b"\x00" * (int(sr * gap / 1000) * sw * ch))

    with wave.open(str(out_path), "wb") as out:
        out.setparams(params)
        out.writeframes(b"".join(pieces))
    return segments

def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()

def run(config, workdir: Path):
    port = _find_port()
    profiles = _profiles(port)

    # Voicebox profile names are per-install, so they live in config:
    # shorts.voicebox_profile is the fallback, and any character with a
    # voicebox_profile overrides it for that speaker.
    default_profile = config.get("shorts", {}).get("voicebox_profile",
                                                   DEFAULT_PROFILE)
    # a character's constant voice description, prepended to each shot's
    # emotion: without it, strong per-shot direction drifts the register far
    # enough that one character reads as several
    styles = {c["name"]: c.get("speaking_style")
              for c in config.get("characters", []) if c.get("speaking_style")}
    # the narrator isn't a character; its constant delivery direction gets
    # its own key so per-shot emotion rides on a stable persona
    if config.get("shorts", {}).get("narrator_instruct"):
        styles.setdefault("narrator",
                          config["shorts"]["narrator_instruct"])
    voice_map = dict(VOICE_MAP)
    for ch in config.get("characters", []):
        if ch.get("voicebox_profile"):
            voice_map[ch["name"]] = ch["voicebox_profile"]

    pace = {k: v for k, v in config.get("shorts", {}).items()
            if k in ("pause_ms", "ellipsis_pause_ms", "shot_gap_ms",
                     "speech_tempo")}

    audio_root = workdir / "audio"
    manifests = sorted((workdir / "manifests").glob("ep*.json"))
    if not manifests:
        sys.exit("no manifests in work/manifests - run stage 4 first")

    for mf in manifests:
        data = json.loads(mf.read_text())
        short_id = data.get("short_id", mf.stem)
        out_dir = audio_root / short_id
        out_dir.mkdir(parents=True, exist_ok=True)
        changed = False
        for i, shot in enumerate(data.get("shots", [])):
            existing = shot.get("audio")
            if existing:
                # idempotent, but only if the wav is actually still there:
                # trusting the field alone silently renders the shot mute
                ep = Path(existing)
                if (ep if ep.is_absolute() else workdir.parent / ep).exists():
                    continue
                print(f"  {short_id} shot{i:03d}: wav missing, re-synthesising")
            line = (shot.get("line") or "").strip()
            if not line:
                shot["audio"] = None
                shot["duration_frames"] = round(SILENT_S * FPS)
                changed = True
                continue
            speaker = (shot.get("speaker") or "narrator")
            pname = voice_map.get(speaker, default_profile)
            entry = profiles.get(pname.lower())
            if not entry:
                sys.exit(f"no Voicebox profile named '{pname}' "
                         f"(have: {', '.join(profiles)})")
            pid, engine = entry
            wav = out_dir / f"shot{i:03d}.wav"
            print(f"  {short_id} shot{i:03d} [{speaker}] {line[:50]!r}")
            base = styles.get(speaker)
            emotion = shot.get("emotion")
            instruct = "; ".join(x for x in (base, emotion) if x) or None
            segments = _synth(port, pid, line, wav, engine, pace, instruct,
                              config.get("shorts", {}).get("tts_pronounce"))
            shot["audio"] = str(wav.relative_to(workdir.parent)
                                if wav.is_relative_to(workdir.parent) else wav)
            shot["duration_frames"] = round(_wav_seconds(wav) * FPS)
            shot["segments"] = segments
            changed = True
        if changed:
            mf.write_text(json.dumps(data, indent=1))
            print(f"{mf.name}: updated")
        else:
            print(f"{mf.name}: nothing to do")
