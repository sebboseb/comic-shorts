"""Stage-2 local vision worker: mlx_vlm, run under its own environment.

The pipeline venv doesn't carry mlx; this script is executed via
`uvx --from mlx-vlm python pipeline/stage2_local.py <job.json> <out.json>`
by stage2_understand when models.vision_provider is "local". The job file
carries the model id, prompt, roster text, and panel list; results are the
same raw (0-1000 coord) annotations the batch path produces, so caching
and _finish() are shared.

Qualified on Qwen3-VL-30B-A3B (2026-07-31): scene/focus near Claude
quality, but it INVENTS speech bubbles on wordless art - hence the
anti-invention rule below and the comic.wordless escape hatch upstream.
"""
import json
import sys

ANTI_INVENTION = (
    "\n- CRITICAL: only transcribe dialogue when an actual speech bubble or "
    "caption box is DRAWN in the panel. Most panels in many comics have no "
    "words at all - for those, dialogue MUST be []. Never imagine what a "
    "character might be saying. The same applies to sfx_text: only report "
    "onomatopoeia that is literally lettered into the art."
)


def main():
    job = json.loads(open(sys.argv[1]).read())

    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(job["model"])
    system = job["system_prompt"] + ANTI_INVENTION

    results = {}
    for panel in job["panels"]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",
             "content": job["roster_text"] + "\n\nNow analyze this panel:"},
        ]
        prompt = apply_chat_template(processor, model.config, messages,
                                     num_images=1)
        out = generate(model, processor, prompt, image=[panel["path"]],
                       max_tokens=1600, temperature=0.0, verbose=False)
        text = (out.text if hasattr(out, "text") else str(out)).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            text = text[4:] if text.startswith("json") else text
        try:
            results[panel["id"]] = json.loads(text)
            print(f"{panel['id']}: ok", file=sys.stderr)
        except json.JSONDecodeError:
            results[panel["id"]] = {
                "scene": "", "characters_present": [], "dialogue": [],
                "focus_box": [0, 0, 1000, 1000], "sfx_text": [],
                "parse_error": text[:500]}
            print(f"{panel['id']}: unparseable", file=sys.stderr)

    open(sys.argv[2], "w").write(json.dumps(results))


if __name__ == "__main__":
    main()
