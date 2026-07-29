"""Stage 6: TTS via the Voicebox desktop app's local API.

Fills `audio` and `duration_frames` on every shot in every manifest.
- speaker -> Voicebox profile via VOICE_MAP (default: everything -> "Narrator")
- silent shots (line == "") get audio=None, duration_frames = SILENT_S * FPS
- idempotent: shots with audio already set are skipped
- wavs land in work/audio/<short_id>/shotNNN.wav

Requires the Voicebox app to be open. Run AFTER the stage-5 review gate.
"""
import json, subprocess, sys, time, urllib.request, wave
from pathlib import Path

FPS = 30
SILENT_S = 1.2
DEFAULT_PROFILE = "Narrator"
ENGINE = "kokoro"
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
    r = _api(port, "GET", "/profiles")
    plist = r if isinstance(r, list) else r.get("profiles", [])
    return {p.get("name", "").lower(): (p.get("id") or p.get("profile_id"))
            for p in plist}

def _synth(port, profile_id, text, out_path: Path):
    gen = _api(port, "POST", "/generate",
               {"profile_id": profile_id, "text": text,
                "language": "en", "engine": ENGINE})
    gen_id = gen.get("id") or gen.get("generation_id")
    if not gen_id:
        raise RuntimeError(f"bad /generate response: {json.dumps(gen)[:200]}")
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            audio = _api(port, "GET", f"/audio/{gen_id}")
            if isinstance(audio, (bytes, bytearray)) and len(audio) > 1000:
                out_path.write_bytes(audio)
                return
        except Exception:
            pass
        time.sleep(1.5)
    raise RuntimeError("synthesis timed out")

def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()

def run(config, workdir: Path):
    port = _find_port()
    profiles = _profiles(port)
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
            if shot.get("audio"):
                continue  # already done
            line = (shot.get("line") or "").strip()
            if not line:
                shot["audio"] = None
                shot["duration_frames"] = round(SILENT_S * FPS)
                changed = True
                continue
            speaker = (shot.get("speaker") or "narrator")
            pname = VOICE_MAP.get(speaker, DEFAULT_PROFILE)
            pid = profiles.get(pname.lower())
            if not pid:
                sys.exit(f"no Voicebox profile named '{pname}' "
                         f"(have: {', '.join(profiles)})")
            wav = out_dir / f"shot{i:03d}.wav"
            print(f"  {short_id} shot{i:03d} [{speaker}] {line[:50]!r}")
            _synth(port, pid, line, wav)
            shot["audio"] = str(wav.relative_to(workdir.parent)
                                if wav.is_relative_to(workdir.parent) else wav)
            shot["duration_frames"] = round(_wav_seconds(wav) * FPS)
            changed = True
        if changed:
            mf.write_text(json.dumps(data, indent=1))
            print(f"{mf.name}: updated")
        else:
            print(f"{mf.name}: nothing to do")
