# comic-shorts

Automated pipeline: flat comic pages -> faceless AI-voiced vertical shorts.
This repo currently covers stages 1-5 (extraction through review gate).
Stages 6+ (TTS, sound design, Remotion render, publish) come next.

## Setup (on the 24GB GPU box)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

One-time model weights for the upscaler:

```bash
mkdir -p weights
curl -L -o weights/RealESRGAN_x4plus_anime_6B.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
```

LaMa inpainting weights auto-download on the first stage-3 run.

## Prepare your comic

1. Put page images in `pages/`, named so they sort in order (`p01.png`...).
2. Edit `config/comic.yaml`: title, target shorts count, character roster.
3. Tag one clean face per character (this is what makes speaker attribution
   accurate). Use the tagging UI instead of cropping by hand:

   ```bash
   python tools/tag_ui.py
   ```

   Drag a box around a face, pick or type the character's name. Crops land in
   `config/characters/` and `comic.yaml` is updated automatically (new names
   get a roster entry with a TODO description). Keyboard: arrows for pages,
   cmd+scroll to zoom, click a roster entry to "arm" it so the next box saves
   without asking.

## Run

```bash
python run.py --stage 1        # panels -> work/panels/, check flagged pages
python run.py --stage 2        # vision pass -> work/understanding.json (API cost: cents)
                               # interrupted? just rerun - it re-attaches to the
                               # submitted batch via work/batch_id.txt instead of
                               # paying again; delete that file to force a fresh submit
python run.py --stage 3        # inpaint + upscale -> work/clean/ (GPU, local)
python run.py --stage 4        # story compile -> work/manifests/ep*.json
python run.py --stage 5        # review page -> work/review/index.html
```

## What to check at each gate

- **After stage 1**: any flagged pages in the console/`panels.json`. Fix bad
  crops by hand (drop corrected PNGs into `work/panels/` with the same naming
  and edit `panels.json`).
- **After stage 3**: skim `work/clean/`. Bubbles overlapping faces are the main
  inpaint failure - touch those up manually, the pipeline won't notice.
- **After stage 5 (review gate)**: open `work/review/index.html`. Fix wrong
  speakers, weak hooks, and misassigned lines directly in
  `work/manifests/epNN.json`, rerun stage 5 to confirm. Only proceed to TTS
  when the manifests read well - everything after this costs money per run.

## Design notes

- `work/manifests/epNN.json` is the single contract for all later stages.
  `audio` and `duration_frames` are null until the TTS stage fills them in.
- Every stage is re-runnable in isolation via `--stage N`.
- Stage 1 uses gutter contour detection - fine for clean western layouts.
  If your pages have heavy bleeds/overlaps, the upgrade path is Magi
  (https://github.com/ragavsachdeva/magi), drop-in at stage 1.
