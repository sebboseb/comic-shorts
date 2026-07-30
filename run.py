#!/usr/bin/env python3
"""Comic shorts pipeline runner.

Usage:
  python run.py --stage 1                 # panel extraction
  python run.py --stage 2                 # vision understanding (API)
  python run.py --stage 3                 # inpaint + upscale (GPU)
  python run.py --stage 4                 # story compile -> manifests (API)
  python run.py --stage 5                 # generate review page
  python run.py --stage 1 2 3 4 5         # chain them
  python run.py --config config/other.yaml --workdir work_other
"""

import argparse
from pathlib import Path

import yaml

def _stage1(config, workdir):
    from pipeline import stage1_panels
    stage1_panels.run(config, workdir)


def _stage2(config, workdir):
    from pipeline import stage2_understand
    stage2_understand.run(config, workdir)


def _stage3(config, workdir):
    from pipeline import stage3_cleanup  # imports torch — only when needed
    stage3_cleanup.run(config, workdir)


def _stage4(config, workdir):
    from pipeline import stage4_story
    stage4_story.run(config, workdir)


def _stage6(config, workdir):
    from pipeline import stage6_tts
    stage6_tts.run(config, workdir)

def _stage5(config, workdir):
    from pipeline import review
    review.run(config, workdir)


def _stage3p(config, workdir):
    from pipeline import stage3_passthrough
    stage3_passthrough.run(config, workdir)


def _check(config, workdir):
    from pipeline import doctor
    doctor.run(config, workdir)


def _stage7(config, workdir):
    from pipeline import stage7_render
    stage7_render.run(config, workdir)


STAGES = {
    "1": ("panel extraction", _stage1),
    "2": ("understanding pass", _stage2),
    "3": ("art cleanup", _stage3),
    "3p": ("art cleanup (passthrough, no GPU)", _stage3p),
    "4": ("story compile", _stage4),
    "5": ("review page", _stage5),
    "6": ("tts (voicebox)", _stage6),
    "7": ("render (ffmpeg)", _stage7),
    "check": ("config check", _check),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=list(STAGES))
    ap.add_argument("--config", default="config/comic.yaml")
    ap.add_argument("--workdir", default="work")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    for s in args.stage:
        name, fn = STAGES[s]
        print(f"\n=== stage {s}: {name} ===")
        fn(config, workdir)


if __name__ == "__main__":
    main()
