"""Config check: what to edit, why, and where.

`python run.py --stage check` — run it before a new comic and again after
stage 2. It answers the only question that matters when setting up a
book: what is wrong right now, and what do I type to fix it.

Every finding is either ERROR (a stage will fail or produce silence) or
WARN (it will run but the result will be worse than it should be), and
each one names the file, the key, and the fix.
"""

import json
from pathlib import Path

ERROR, WARN, OK = "ERROR", "warn ", "ok   "


def _pl(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def run(config, workdir: Path):
    findings = []      # (level, message, fix)
    cfg_path = "config/comic.yaml"
    shorts = config.get("shorts", {})
    roster = config.get("characters", [])
    names = {c["name"] for c in roster}

    # --- pages -------------------------------------------------------
    pages_dir = Path(config["comic"]["pages_dir"])
    imgs = sorted(p for p in pages_dir.iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                  ) if pages_dir.exists() else []
    if not imgs:
        findings.append((ERROR, f"no page images in {pages_dir}/",
                         f"copy the comic's pages there, or point "
                         f"comic.pages_dir at them in {cfg_path}"))
    else:
        findings.append((OK, f"{_pl(len(imgs), 'page')} in {pages_dir}/", None))

    # --- roster basics ----------------------------------------------
    if not roster:
        findings.append((ERROR, "characters roster is empty",
                         f"add the comic's cast under `characters:` in {cfg_path}"))
    for c in roster:
        if not c.get("voicebox_profile"):
            findings.append((ERROR, f"{c['name']}: no voicebox_profile",
                             f"add `voicebox_profile: \"<name>\"` under "
                             f"{c['name']} in {cfg_path} (stage 6 will use the "
                             f"default voice for every line otherwise)"))
        if c.get("ref_image"):
            p = Path("config/characters") / c["ref_image"]
            if not p.exists():
                findings.append((WARN, f"{c['name']}: ref_image {p} missing",
                                 "drop a face crop there, or remove the "
                                 "ref_image key — attribution is notably "
                                 "worse without one"))
        if (c.get("speaking_style") or "").strip().lower() in ("", "neutral"):
            findings.append((WARN, f"{c['name']}: speaking_style is "
                             f"{c.get('speaking_style')!r}",
                             "describe the voice instead (it anchors identity "
                             "across shots; \"neutral\" anchors nothing)"))
        if not c.get("epithet"):
            findings.append((WARN, f"{c['name']}: no epithet",
                             "add one so narration can name them in the house "
                             "voice rather than plainly"))

    if not any(c.get("principal") for c in roster):
        findings.append((WARN, "no character marked `principal: true`",
                         "mark the 2-4 characters who should actually speak "
                         "on camera; everyone else's dialogue gets narrated "
                         "over or dropped"))

    # --- captions ----------------------------------------------------
    colours = (shorts.get("caption_style") or {}).get("speaker_colors") or {}
    uncoloured = names - set(colours)
    if uncoloured:
        findings.append((WARN, f"no caption colour for "
                         f"{', '.join(sorted(uncoloured))}",
                         f"add them under shorts.caption_style.speaker_colors "
                         f"in {cfg_path} (they fall back to white, so two "
                         f"characters can look like the same speaker)"))

    # --- narration voice --------------------------------------------
    if not (shorts.get("narration_style") or "").strip():
        findings.append((WARN, "shorts.narration_style is empty",
                         "this is the single biggest lever on how the short "
                         "feels — write the narrator's persona"))

    # --- music -------------------------------------------------------
    if shorts.get("music_volume"):
        mdir = Path(shorts.get("music_dir", "")).expanduser()
        missing = [n for n in (shorts.get("music_by_mood") or {}).values()
                   if not (mdir / n).exists()]
        if missing:
            findings.append((WARN, f"{_pl(len(missing), 'music track')} missing "
                             f"from {mdir}", "fix the filenames under "
                             "shorts.music_by_mood, or set music_volume: 0"))

    # --- post-stage-2: who actually speaks in this comic ------------
    u_path = workdir / "understanding.json"
    if not u_path.exists():
        findings.append((OK, "run stage 2, then run this check again to see "
                         "who the comic actually needs in the roster", None))
    else:
        panels = json.loads(u_path.read_text())
        seen, unknown, total = {}, 0, 0
        for p in panels:
            for d in p.get("dialogue", []):
                s = d.get("speaker", "unknown")
                total += 1
                if s == "unknown":
                    unknown += 1
                elif s != "narrator":
                    seen[s] = seen.get(s, 0) + 1
        offroster = {k: v for k, v in seen.items() if k not in names}
        if offroster:
            listing = ", ".join(f"{k} ({v})" for k, v in
                               sorted(offroster.items(), key=lambda x: -x[1]))
            findings.append((WARN, f"speakers found in the art but not in the "
                             f"roster: {listing}",
                             f"add them under `characters:` in {cfg_path} — a "
                             f"description alone is enough, no crop needed — "
                             f"then re-run stage 2. Their lines are currently "
                             f"unattributed."))
        if total:
            pct = unknown / total
            # unattributed lines are not automatically a problem: stage 4's
            # cast-discipline rule narrates or cuts them rather than guessing a
            # voice. They only matter if a line someone SHOULD have said is in
            # there, so show a sample and let the operator judge.
            findings.append((OK if pct <= 0.35 else WARN,
                             f"{unknown}/{total} dialogue lines ({pct:.0%}) have "
                             f"no speaker - these will be narrated or cut, not "
                             f"voiced",
                             None))
            samples = [d["text"][:64] for p in panels
                       for d in p.get("dialogue", [])
                       if d.get("speaker") == "unknown"][:5]
            if samples:
                print("\n  unattributed lines (a sample) - if a PRINCIPAL says "
                      "any of these,\n  add them to the roster and re-run "
                      "stage 2; if it is background\n  chatter, nothing to do:")
                for s in samples:
                    print(f"      {s!r}")

    # --- report ------------------------------------------------------
    print(f"\nconfig: {cfg_path}\n")
    for level, msg, fix in findings:
        print(f"  [{level}] {msg}")
        if fix:
            print(f"          -> {fix}")
    errs = sum(1 for f in findings if f[0] == ERROR)
    warns = sum(1 for f in findings if f[0] == WARN)
    print(f"\n{_pl(errs, 'error')}, {_pl(warns, 'warning')}.")
    if errs:
        print("Errors will break a stage or render it silent. Fix those first.")
