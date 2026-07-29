"""Review gate 1: generate work/review/index.html.

Shows every short shot-by-shot: the cleaned panel next to the assigned
speaker, line, emotion, and motion, plus per-panel unknown-speaker warnings.
You correct mistakes by editing work/manifests/epNN.json directly, then
regenerate this page to confirm.

This is deliberately a dumb static page - the manifest stays the single
source of truth.
"""

import html
import json
from pathlib import Path

CSS = """
body { font-family: system-ui, sans-serif; margin: 24px; background: #fafafa;
       color: #222; max-width: 1100px; }
h1 { font-size: 22px; } h2 { font-size: 18px; margin-top: 40px; }
.meta { color: #666; font-size: 14px; margin-bottom: 6px; }
.shot { display: flex; gap: 16px; padding: 12px; margin: 8px 0;
        background: #fff; border: 1px solid #ddd; border-radius: 8px; }
.shot img { width: 220px; height: auto; align-self: flex-start;
            border-radius: 4px; }
.shot .info { font-size: 14px; line-height: 1.6; }
.speaker { font-weight: 600; }
.line { font-size: 16px; margin: 4px 0; }
.warn { color: #b00; font-weight: 600; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px;
         background: #eee; font-size: 12px; margin-right: 6px; }
.hero { background: #ffe9c7; }
"""


def run(config, workdir: Path):
    manifest_dir = workdir / "manifests"
    manifests = sorted(manifest_dir.glob("ep*.json"))
    if not manifests:
        raise SystemExit("No manifests found - run stage 4 first")

    clean = {p["id"]: p for p in
             json.loads((workdir / "clean.json").read_text())}

    parts = [f"<style>{CSS}</style>", "<h1>Review gate 1</h1>",
             "<p class='meta'>Fix errors directly in work/manifests/*.json, "
             "then rerun this stage to confirm. Check: wrong speakers, weak "
             "hooks, lines assigned to the wrong panel.</p>"]

    for mpath in manifests:
        short = json.loads(mpath.read_text())
        parts.append(f"<h2>{short['short_id']} - "
                     f"{html.escape(short['title'])}</h2>")
        parts.append(f"<p class='meta'>hook: "
                     f"{html.escape(short.get('hook_note', ''))}<br>"
                     f"cliffhanger: "
                     f"{html.escape(short.get('cliffhanger_note', ''))}<br>"
                     f"music: {short.get('music_mood', '')}</p>")

        for i, shot in enumerate(short["shots"], start=1):
            panel = clean.get(shot["panel"])
            img_src = f"../{panel['clean_file']}" if panel else ""
            missing = "" if panel else \
                "<div class='warn'>panel not found in clean.json</div>"
            unknown = "<span class='warn'>UNKNOWN SPEAKER - fix me</span>" \
                if shot.get("speaker") == "unknown" else ""
            hero = "<span class='badge hero'>hero</span>" \
                if shot.get("hero") else ""
            line = html.escape(shot.get("line", "")) or "<i>(silent beat)</i>"

            parts.append(
                f"<div class='shot'>"
                f"<img src='{img_src}' loading='lazy'>"
                f"<div class='info'>"
                f"<div><span class='badge'>shot {i}</span>"
                f"<span class='badge'>{shot.get('motion', '')}</span>"
                f"<span class='badge'>sfx: {shot.get('sfx', 'none')}</span>"
                f"{hero}</div>"
                f"<div class='speaker'>{html.escape(shot.get('speaker', ''))} "
                f"{unknown}</div>"
                f"<div class='line'>{line}</div>"
                f"<div class='meta'>emotion: "
                f"{html.escape(shot.get('emotion', ''))}</div>"
                f"{missing}</div></div>"
            )

    out_dir = workdir / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "index.html"
    out.write_text("\n".join(parts))
    print(f"Review page -> {out}\nOpen it in a browser "
          "(python -m http.server from the work dir works).")
