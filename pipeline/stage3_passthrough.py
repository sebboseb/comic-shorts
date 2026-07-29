"""Stage 3 (passthrough): no inpaint, no GPU upscale.

Produces the same clean.json contract as stage3_cleanup so stages 4-7 run
unchanged, but keeps the original panel art: speech bubbles stay visible
and panels are only upscaled with Lanczos when they're too small to fill
the render canvas without visible softness.

Use when you want an end-to-end pass without the torch/LaMa/ESRGAN stack
(`--stage 3p`). Switch to the real stage 3 before publishing: bubbles
duplicate the narration on screen, and Lanczos can't invent detail the
way ESRGAN does.

Output: work/clean/pPP_NN.png + work/clean.json
"""

import json
from pathlib import Path

from PIL import Image


def run(config, workdir: Path):
    target = config.get("models", {}).get("upscale_target_long_side", 2200)

    panels = json.loads((workdir / "understanding.json").read_text())
    out_dir = workdir / "clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, panel in enumerate(panels, start=1):
        img = Image.open(workdir / panel["file"]).convert("RGB")
        old_size = img.size

        if max(img.size) < target:
            scale = target / max(img.size)
            img = img.resize((round(img.width * scale), round(img.height * scale)),
                             Image.LANCZOS)

        out_name = f"{panel['id']}.png"
        img.save(out_dir / out_name)
        results.append({**panel, "clean_file": f"clean/{out_name}",
                        "clean_size": list(img.size)})
        print(f"[{i}/{len(panels)}] {panel['id']}: {old_size} -> {img.size}")

    (workdir / "clean.json").write_text(json.dumps(results, indent=2))
    print(f"\nDone -> {out_dir} (passthrough: bubbles NOT removed, "
          "Lanczos not ESRGAN)")
