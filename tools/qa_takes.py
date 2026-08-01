"""Post-batch take QA: transcribe every shot wav and flag suspects.

Catches what synth-time verification can miss (it transcribes once per
sentence and whisper is stochastic): stutter loops, laughter heard as
clean text but costing seconds, wrong words. Slomo shots are transcribed
after restoring normal speed/pitch, since whisper can't decode slowed
pitch-dropped audio.

Usage: .venv/bin/python tools/qa_takes.py work_jeff [work_jeff_pool ...]
       [--to-notes]   # also queue flagged shots as pre-tagged re-rolls in
                      # review_notes.json, so they appear in the review GUI
Exit code 1 if any shot is flagged.
"""
import difflib
import json
import re
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.stage6_tts import (_find_port, _transcribe, _speakable,  # noqa: E402
                                 SLOMO_SPEED, SLOMO_PITCH, LAUGH_MARKERS)

MAX_SPW = 0.45          # absolute ceiling; episodes run 0.2-0.38 s/word
REL_HARD = 1.45         # >1.45x episode median: flagged for re-roll
REL_SOFT = 1.25         # 1.25-1.45x: queued as a "check this" note only -
                        # a mild drag is a taste call, not a defect
MIN_SIM = 0.55          # transcript vs line, after case/punct folding
MIN_RELIABLE_S = 2.0    # whisper hallucinates below this; rate check only


def _restore_slomo(path):
    """Undo the slomo treatment so whisper hears normal speech."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(path),
         "-af", f"atempo={1 / (SLOMO_SPEED / SLOMO_PITCH):.4f},"
                f"asetrate={sr}/{SLOMO_PITCH},aresample={sr}",
         str(tmp)], check=True, capture_output=True)
    return tmp


def _fold(t):
    tokens = re.sub(r"[^a-z' ]", " ", t.lower()).split()
    # whisper appends a hallucinated "NO NO NO..." run after speech on many
    # of these clips; it's a transcription artifact, not audio. Strip a
    # trailing run of 3+ 'no's before judging the take.
    end = len(tokens)
    while end > 0 and tokens[end - 1] == "no":
        end -= 1
    if len(tokens) - end >= 3:
        tokens = tokens[:end]
    return tokens


def _audio_loop(path, min_lag_s=0.15, max_lag_s=1.0, sim_threshold=0.95):
    """Does the waveform physically repeat? A real synth stutter loops the
    audio itself; a whisper hallucination ("fat fat fat" on clean speech -
    observed, and unstable across sessions) does not. Looks for THREE
    consecutive near-identical periods in the local RMS envelope at any
    stutter-plausible lag. Lags under 0.15s are excluded: that is syllable
    cadence, where clean speech legitimately self-similars (measured up to
    0.99); true loops score ~1.0 at their period (measured on a seeded
    defect). Normal speech does not produce three
    high-similarity repeats in a row."""
    import numpy as np
    with wave.open(str(path), "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    mono = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    hop = int(sr * 0.02)
    frames = len(mono) // hop
    if frames < 20:
        return False
    env = np.sqrt((mono[:frames * hop].reshape(frames, hop) ** 2).mean(axis=1))
    loud = env.max() * 0.05
    lo, hi = max(int(min_lag_s / 0.02), 3), min(int(max_lag_s / 0.02), frames // 3)
    for lag in range(lo, hi):
        for s in range(0, frames - 3 * lag, max(lag // 2, 1)):
            a, b, c = (env[s:s + lag], env[s + lag:s + 2 * lag],
                       env[s + 2 * lag:s + 3 * lag])
            if a.mean() < loud:      # don't match silence against silence
                continue
            def cos(x, y):
                x = x - x.mean(); y = y - y.mean()
                d = (np.linalg.norm(x) * np.linalg.norm(y)) or 1.0
                return float((x * y).sum() / d)
            if cos(a, b) > sim_threshold and cos(b, c) > sim_threshold:
                return True
    return False


def _stutter(tokens):
    """A token (or pair) repeated 4+ times in a row is a synth loop."""
    for n in (1, 2):
        run = 1
        for i in range(n, len(tokens) - n + 1, n):
            if tokens[i:i + n] == tokens[i - n:i]:
                run += 1
                if run >= 4:
                    return " ".join(tokens[i:i + n])
            else:
                run = 1
    return None


def write_notes(flagged, eps="ep01"):
    """Queue QA findings in each workdir's review_notes.json - the robot's
    suspicions land in the review GUI queue for the human to confirm or
    dismiss. Hard problems arrive pre-tagged as re-rolls; soft warns arrive
    as notes only. Merge-only: never clobbers or removes an operator's own
    annotations. Accepts (wd, i, problems[, warns]) tuples (legacy, uses
    `eps`) or (wd, ep, i, problems, warns)."""
    by_wd = {}
    for entry in flagged:
        if len(entry) == 5:
            wd, ep, i, problems, warns = entry
        else:
            wd, i, problems = entry[0], entry[1], entry[2]
            warns = entry[3] if len(entry) > 3 else []
            ep = eps
        by_wd.setdefault(wd, []).append((ep, i, problems, warns))
    for wd, items in by_wd.items():
        path = Path(wd) / "review_notes.json"
        notes = json.loads(path.read_text()) if path.exists() else {}
        for ep, i, problems, warns in items:
            entry = notes.setdefault(ep, {}).setdefault(
                str(i), {"tags": [], "note": ""})
            if problems and "reroll" not in entry["tags"]:
                entry["tags"].append("reroll")
            qa = "QA: " + "; ".join(problems + warns)
            if qa not in entry["note"]:
                entry["note"] = (entry["note"] + "\n" + qa).strip()
        path.write_text(json.dumps(notes, indent=2))
        print(f"{wd}: {len(items)} QA finding(s) queued in review_notes.json")


def main():
    to_notes = "--to-notes" in sys.argv
    workdirs = [a for a in sys.argv[1:] if not a.startswith("--")]
    port = _find_port()
    flagged = []
    for wd in workdirs or ["work_jeff"]:
      for mpath in sorted(Path(f"{wd}/manifests").glob("ep*.json")):
        ep = mpath.stem
        d = json.loads(mpath.read_text())
        print(f"--- {wd} {ep}")
        # episode median rate for the relative slow check (slomo excluded)
        rates = []
        for i, shot in enumerate(d["shots"]):
            if not shot.get("audio") or shot.get("motion") == "slomo":
                continue
            with wave.open(f"{wd}/audio/{ep}/shot{i:03d}.wav", "rb") as w:
                sec = w.getnframes() / w.getframerate()
            rates.append(sec / max(len(shot["line"].split()), 1))
        rates.sort()
        median = rates[len(rates) // 2] if rates else 0.3
        for i, shot in enumerate(d["shots"]):
            if not shot.get("audio"):
                continue
            wav = Path(f"{wd}/audio/{ep}/shot{i:03d}.wav")
            slomo = shot.get("motion") == "slomo"
            src = _restore_slomo(wav) if slomo else wav
            with wave.open(str(src), "rb") as w:
                sec = w.getnframes() / w.getframerate()
            words = max(len(shot["line"].split()), 1)
            spw = sec / words
            problems, warns = [], []
            if not slomo and (spw > MAX_SPW or spw > REL_HARD * median):
                # slomo shots are exempt: their delivery is DIRECTED slow
                problems.append(f"slow ({spw:.2f}s/w vs median {median:.2f})")
            elif not slomo and spw > REL_SOFT * median:
                warns.append(f"slowish ({spw:.2f}s/w vs median {median:.2f})")
            heard = _transcribe(port, src.read_bytes()) or ""
            ht = _fold(heard)
            if sec >= MIN_RELIABLE_S and ht:  # empty after strip = whisper
                if any(m in heard.lower() for m in LAUGH_MARKERS):
                    problems.append("laughter")
                loop = _stutter(ht)
                if loop and _audio_loop(src):
                    # transcript loop CONFIRMED by waveform periodicity
                    problems.append(f"stutter loop ({loop!r}, audio-confirmed)")
                elif loop:
                    # whisper hallucinated the loop (unstable across
                    # sessions); its whole transcript is untrustworthy, so
                    # skip the similarity check too
                    pass
                elif len(set(ht)) <= 2 and len(ht) < 6:
                    # degenerate transcript ("Male Male Male") on a real
                    # line: whisper failed on the isolated clip - skip, the
                    # rate check stands (verified: same take transcribes
                    # perfectly with neighbor context)
                    pass
                else:
                    want = _fold(_speakable(shot["line"]))
                    sim = difflib.SequenceMatcher(None, ht[:len(want) + 6],
                                                  want).ratio()
                    if sim < MIN_SIM:
                        problems.append(
                            f"mismatch (sim {sim:.2f}: {heard[:60]!r})")
            status = ("FLAG " + "; ".join(problems) if problems else
                      ("check " + "; ".join(warns) if warns else "ok"))
            print(f"shot{i:03d} {sec:5.2f}s {spw:.2f}s/w  {status}")
            if problems or warns:
                flagged.append((wd, ep, i, problems, warns))
            if slomo:
                src.unlink()
    if flagged:
        print("\nflagged:", flagged)
        if to_notes:
            write_notes(flagged)
        sys.exit(1)
    print("\nall takes clean")


if __name__ == "__main__":
    main()
