"""Probe: can a local VLM replace Claude for stage 2?

Runs stage 2's exact SYSTEM_PROMPT against a local OpenAI-compatible VLM
server (mlx-openai-server) on a sample of panels, and prints the local
answer next to the Claude baseline from work*/understanding.json so the
two can be judged side by side.

Usage: .venv/bin/python tools/stage2_local_probe.py [panel_id ...]
"""
import base64
import io
import json
import sys
import urllib.request
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.stage2_understand import SYSTEM_PROMPT, _parse_json  # noqa: E402

BASE = "http://127.0.0.1:8083/v1"
WORKDIR = Path("work_jeff")
MAX_SEND_PX = 1120  # slightly under stage 2's 1400: keeps 7B prompt small

DEFAULT_SAMPLE = ["p01_02", "p02_03", "p03_05", "p04_06",
                  "p05_05", "p06_01", "p09_05", "p10_01"]

ROSTER_TEXT = """Character (no reference image): Jeff - Jeff the Land Shark: small grey baby-shark pet with stubby legs, big black eyes, permanently delighted. Walks on land, acts like a dog.
Character (no reference image): Kate Bishop - Kate Bishop, Hawkeye: young woman, long black hair, purple sleeveless outfit with black gloves
Character (no reference image): Iceman - Iceman (Bobby Drake): X-Man made of translucent blue ice
Character (no reference image): Doctor - burly bearded human doctor in a white coat at an animal-friendly clinic"""


def b64(path):
    img = Image.open(path).convert("RGB")
    if max(img.size) > MAX_SEND_PX:
        s = MAX_SEND_PX / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.standard_b64encode(buf.getvalue()).decode()


def ask_local(panel_path):
    body = {
        "model": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        "max_tokens": 1200,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": ROSTER_TEXT},
                {"type": "text", "text": "Now analyze this panel:"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64(panel_path)}"}},
            ]},
        ],
    }
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer local"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"]


def main():
    sample = sys.argv[1:] or DEFAULT_SAMPLE
    baseline = {p["id"]: p for p in
                json.loads((WORKDIR / "understanding.json").read_text())}
    for pid in sample:
        panel = baseline.get(pid)
        if not panel:
            print(f"{pid}: not in understanding.json, skipping")
            continue
        raw = ask_local(WORKDIR / panel["file"])
        try:
            local = _parse_json(raw)
            local_scene = local.get("scene", "")
            local_extra = {k: local.get(k) for k in
                           ("characters_present", "dialogue", "sfx_text")}
        except json.JSONDecodeError:
            local_scene, local_extra = f"UNPARSEABLE: {raw[:200]}", {}
        print(f"\n=== {pid}")
        print(f"  claude: {panel.get('scene', '')}")
        print(f"  local : {local_scene}")
        print(f"          {json.dumps(local_extra, default=str)[:220]}")


if __name__ == "__main__":
    main()
