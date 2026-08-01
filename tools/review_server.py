"""Interactive review gate: annotate shots, don't edit JSON.

Serves one page per episode: the rendered video on top, the shot timeline
under it. Click a shot (or just pause the video - the current shot is
auto-selected), hit a quick tag (Re-roll / Slomo / Cut) or write a note in
plain words ("remove the laugh", "say motherfucking hulk here"). Notes are
saved to work_*/review_notes.json as data; `python run.py --stage notes`
compiles them into manifest edits (pipeline/apply_notes.py) and the next
`--stage 6 7` picks them up.

Usage: .venv/bin/python tools/review_server.py --workdir work_jeff [--port 8420]
"""
import argparse
import json
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FPS = 30

PAGE = """<!doctype html><meta charset="utf-8">
<title>Review — {ep}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 0; background: #111;
       color: #eee; }}
#top {{ display: flex; gap: 20px; padding: 16px; }}
video {{ height: 62vh; border-radius: 8px; background: #000; }}
#panel {{ flex: 1; min-width: 320px; }}
#panel h2 {{ margin: 4px 0 10px; font-size: 17px; color: #ffd54a; }}
#panel .line {{ font-size: 15px; color: #ccc; margin-bottom: 10px; }}
.chips button {{ margin: 0 6px 8px 0; padding: 7px 14px; border-radius: 16px;
  border: 1px solid #555; background: #222; color: #ddd; cursor: pointer;
  font-size: 14px; }}
.chips button.on {{ background: #ffd54a; color: #111; border-color: #ffd54a;
  font-weight: 600; }}
textarea {{ width: 100%; height: 110px; background: #1c1c1c; color: #eee;
  border: 1px solid #444; border-radius: 8px; padding: 10px; font-size: 15px;
  box-sizing: border-box; }}
#status {{ color: #7c7; font-size: 13px; height: 18px; }}
#meta {{ margin: 10px 0; }}
#meta span {{ display: inline-block; background: #2a2a2a; border-radius: 10px;
  padding: 2px 10px; margin: 0 6px 6px 0; font-size: 12px; color: #aaa; }}
#queue {{ margin-top: 14px; font-size: 13px; }}
#qlist div {{ padding: 4px 8px; margin: 4px 0; background: #1c1c1c;
  border-left: 3px solid #e66; border-radius: 4px; cursor: pointer;
  color: #ccc; }}
#keys {{ margin-top: 14px; color: #666; font-size: 12px; }}
#apply {{ margin-top: 10px; padding: 9px 18px; border-radius: 8px;
  border: none; background: #ffd54a; color: #111; font-weight: 700;
  font-size: 14px; cursor: pointer; }}
#apply:disabled {{ background: #555; color: #999; cursor: wait; }}
#applystatus {{ margin-left: 10px; font-size: 13px; color: #7c7; }}
#strip {{ display: flex; overflow-x: auto; gap: 8px; padding: 10px 16px 20px; }}
.shot {{ flex: 0 0 130px; cursor: pointer; border: 2px solid #333;
  border-radius: 8px; background: #1a1a1a; padding: 6px; }}
.shot img {{ width: 100%; height: 90px; object-fit: cover; border-radius: 4px; }}
.shot .t {{ font-size: 11px; color: #999; }}
.shot .w {{ font-size: 11px; color: #ccc; height: 26px; overflow: hidden; }}
.shot.noted {{ border-color: #e66; }}
.shot.current {{ border-color: #ffd54a; box-shadow: 0 0 0 2px #ffd54a; }}
.shot .dot {{ color: #e66; font-weight: 700; }}
#epnote {{ padding: 0 16px 30px; }}
#epnote textarea {{ height: 60px; }}
</style>
<div id="top">
  <video src="/media/renders/{ep}.mp4" controls></video>
  <div id="panel">
    <h2 id="sel">click a shot</h2>
    <div class="line" id="selline"></div>
    <div class="chips" id="chips">
      <button data-tag="reroll">Re-roll take</button>
      <button data-tag="slomo">Slomo</button>
      <button data-tag="cut">Cut shot</button>
    </div>
    <textarea id="note" placeholder="say what should change, in plain words"></textarea>
    <div id="status"></div>
    <div id="meta"></div>
    <div id="queue"><b>Queued changes</b><div id="qlist">none yet</div>
      <button id="apply">Apply &amp; re-render</button>
      <span id="applystatus"></span></div>
    <div id="keys">keys: &larr;/&rarr; shots &middot; R re-roll &middot; S slomo &middot; X cut &middot; N note &middot; 1/2 speed</div>
  </div>
</div>
<div id="strip"></div>
<div id="epnote"><b>Episode note</b> (applies to the whole video)<br>
  <textarea id="gnote" placeholder="e.g. the whole middle drags, tighten it"></textarea>
</div>
<script>
const SHOTS = {shots_json};
const EP = "{ep}";
let notes = {notes_json};
let sel = null;
const strip = document.getElementById("strip");
const video = document.querySelector("video");

SHOTS.forEach((s, i) => {{
  const d = document.createElement("div");
  d.className = "shot"; d.id = "shot" + i;
  d.innerHTML = `<img src="${{s.img}}" loading="lazy">
    <div class="t">#${{i}} · ${{s.start.toFixed(1)}}s <span class="dot"></span></div>
    <div class="w">${{s.line}}</div>`;
  d.onclick = () => {{ select(i); video.currentTime = s.start + 0.05; }};
  strip.appendChild(d);
}});

function noteFor(i) {{
  if (!notes[EP]) notes[EP] = {{}};
  if (!notes[EP][i]) notes[EP][i] = {{ tags: [], note: "" }};
  return notes[EP][i];
}}
function refresh() {{
  renderQueue();
  SHOTS.forEach((s, i) => {{
    const n = (notes[EP] || {{}})[i];
    const has = n && (n.tags.length || n.note.trim());
    document.getElementById("shot" + i).classList.toggle("noted", !!has);
    document.querySelector(`#shot${{i}} .dot`).textContent = has ? "●" : "";
  }});
  if (sel === null) return;
  const n = noteFor(sel);
  document.querySelectorAll("#chips button").forEach(b =>
    b.classList.toggle("on", n.tags.includes(b.dataset.tag)));
  document.getElementById("note").value = n.note;
}}
function select(i) {{
  sel = i;
  document.querySelectorAll(".shot").forEach(d => d.classList.remove("current"));
  const card = document.getElementById("shot" + i);
  card.classList.add("current");
  card.scrollIntoView({{ inline: "center", block: "nearest", behavior: "smooth" }});
  document.getElementById("sel").textContent = `shot ${{i}} — ${{SHOTS[i].panel}}`;
  document.getElementById("selline").textContent = SHOTS[i].line;
  const s = SHOTS[i];
  document.getElementById("meta").innerHTML =
    `<span>${{s.motion}}</span><span>sfx: ${{s.sfx}}</span>` +
    (s.hero ? "<span>hero</span>" : "") +
    `<span>${{s.emotion}}</span>`;
  refresh();
}}
function renderQueue() {{
  const q = document.getElementById("qlist");
  const items = Object.entries(notes[EP] || {{}}).filter(([k, n]) =>
    (n.tags || []).length || (n.note || "").trim());
  if (!items.length) {{ q.innerHTML = "none yet"; return; }}
  q.innerHTML = items.map(([k, n]) => {{
    const what = [...(n.tags || []), (n.note || "").trim()].filter(Boolean).join(" · ");
    const label = k === "_episode" ? "episode" : "shot " + k;
    return `<div data-k="${{k}}">${{label}}: ${{what}}</div>`;
  }}).join("");
  q.querySelectorAll("div").forEach(d => d.onclick = () => {{
    const k = d.dataset.k;
    if (k !== "_episode") {{ select(+k); video.currentTime = SHOTS[+k].start + 0.05; }}
  }});
}}
video.addEventListener("timeupdate", () => {{
  const t = video.currentTime;
  const i = SHOTS.findIndex(s => t >= s.start && t < s.end);
  if (i >= 0 && i !== sel) select(i);
}});
document.querySelectorAll("#chips button").forEach(b => b.onclick = () => {{
  if (sel === null) return;
  const n = noteFor(sel), tag = b.dataset.tag;
  n.tags = n.tags.includes(tag) ? n.tags.filter(t => t !== tag) : [...n.tags, tag];
  save();
}});
let timer = null;
document.getElementById("note").addEventListener("input", e => {{
  if (sel === null) return;
  noteFor(sel).note = e.target.value;
  clearTimeout(timer); timer = setTimeout(save, 600);
}});
document.getElementById("gnote").addEventListener("input", e => {{
  if (!notes[EP]) notes[EP] = {{}};
  notes[EP]["_episode"] = {{ note: e.target.value }};
  clearTimeout(timer); timer = setTimeout(save, 600);
}});
if ((notes[EP] || {{}})["_episode"])
  document.getElementById("gnote").value = notes[EP]["_episode"].note || "";
async function save() {{
  await fetch("/notes", {{ method: "POST", body: JSON.stringify(notes) }});
  document.getElementById("status").textContent =
    "saved — apply with: python run.py --stage notes 6 7";
  refresh();
}}
function toggleTag(tag) {{
  if (sel === null) return;
  const n = noteFor(sel);
  n.tags = n.tags.includes(tag) ? n.tags.filter(t => t !== tag) : [...n.tags, tag];
  save();
}}
document.addEventListener("keydown", e => {{
  if (["TEXTAREA", "INPUT"].includes(document.activeElement.tagName)) {{
    if (e.key === "Escape") document.activeElement.blur();
    return;
  }}
  if (e.key === "ArrowRight") {{ const i = Math.min((sel ?? -1) + 1, SHOTS.length - 1); select(i); video.currentTime = SHOTS[i].start + 0.05; }}
  else if (e.key === "ArrowLeft") {{ const i = Math.max((sel ?? 1) - 1, 0); select(i); video.currentTime = SHOTS[i].start + 0.05; }}
  else if (e.key === "r" || e.key === "R") toggleTag("reroll");
  else if (e.key === "s" || e.key === "S") toggleTag("slomo");
  else if (e.key === "x" || e.key === "X") toggleTag("cut");
  else if (e.key === "n" || e.key === "N") {{ e.preventDefault(); document.getElementById("note").focus(); }}
  else if (e.key === "1") video.playbackRate = 1;
  else if (e.key === "2") video.playbackRate = 1.5;
}});
const applyBtn = document.getElementById("apply");
const applyStatus = document.getElementById("applystatus");
applyBtn.onclick = async () => {{
  if (!confirm("Apply queued changes and re-render? Changed shots re-synthesize (a few minutes).")) return;
  applyBtn.disabled = true;
  applyStatus.textContent = "applying notes, re-synthesizing, rendering...";
  await fetch("/apply", {{ method: "POST" }});
  const poll = setInterval(async () => {{
    const s = await (await fetch("/apply/status")).json();
    if (!s.running) {{
      clearInterval(poll);
      applyStatus.textContent = s.ok ? "done — reloading" : ("FAILED: " + s.tail);
      if (s.ok) setTimeout(() => location.reload(), 800);
      else applyBtn.disabled = false;
    }}
  }}, 2000);
}};
refresh();
</script>"""


def _shots_payload(workdir, ep):
    manifest = json.loads((workdir / "manifests" / f"{ep}.json").read_text())
    clean = {p["id"]: p for p in
             json.loads((workdir / "clean.json").read_text())}
    shots, clock = [], 0
    for s in manifest["shots"]:
        dur = (s.get("duration_frames") or 36) / FPS
        panel = clean.get(s.get("panel"), {})
        shots.append({
            "panel": s.get("panel"), "line": s.get("line", ""),
            "start": round(clock, 3), "end": round(clock + dur, 3),
            "img": "/media/" + panel.get("clean_file", ""),
            "motion": s.get("motion", ""), "sfx": s.get("sfx", "none"),
            "emotion": s.get("emotion", ""), "hero": bool(s.get("hero")),
        })
        clock += dur
    return shots


APPLY = {"running": False, "ok": None, "tail": ""}


def _apply_worker(workdir, config):
    log = workdir / "apply.log"
    with open(log, "w") as f:
        r = subprocess.run(
            [sys.executable, "run.py", "--stage", "notes", "6", "7",
             "--config", config, "--workdir", str(workdir)],
            stdout=f, stderr=subprocess.STDOUT)
    APPLY["ok"] = r.returncode == 0
    APPLY["tail"] = "\n".join(log.read_text().splitlines()[-3:])
    APPLY["running"] = False


class Handler(BaseHTTPRequestHandler):
    workdir: Path
    config: str

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/apply":
            if APPLY["running"]:
                return self._send(409, b"already running", "text/plain")
            APPLY.update(running=True, ok=None, tail="")
            threading.Thread(target=_apply_worker,
                             args=(self.workdir, self.config),
                             daemon=True).start()
            return self._send(200, b"started", "text/plain")
        if self.path != "/notes":
            return self._send(404, b"nope")
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        json.loads(raw)  # validate before persisting
        (self.workdir / "review_notes.json").write_bytes(raw)
        self._send(200, b"ok", "text/plain")

    def do_GET(self):
        if self.path == "/apply/status":
            return self._send(200, json.dumps(APPLY).encode(),
                              "application/json")
        if self.path.startswith("/media/"):
            return self._media(self.workdir / self.path[len("/media/"):])
        eps = sorted(p.stem for p in (self.workdir / "manifests").glob("ep*.json"))
        m = re.match(r"^/(?:\?ep=(\w+))?$", self.path)
        if not m:
            return self._send(404, b"nope")
        ep = m.group(1) or eps[0]
        notes_file = self.workdir / "review_notes.json"
        notes = notes_file.read_text() if notes_file.exists() else "{}"
        page = PAGE.format(ep=ep, notes_json=notes,
                           shots_json=json.dumps(_shots_payload(self.workdir, ep)))
        self._send(200, page.encode())

    def _media(self, path):
        path = path.resolve()
        if not (path.is_file() and self.workdir.resolve() in path.parents):
            return self._send(404, b"nope")
        data = path.read_bytes()
        ctype = ("video/mp4" if path.suffix == ".mp4" else "image/png")
        rng = self.headers.get("Range")
        if rng:  # video seeking needs Range support
            m = re.match(r"bytes=(\d+)-(\d*)", rng)
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else len(data) - 1
            chunk = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
        else:
            self._send(200, data, ctype)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--config", default="config/comic.yaml")
    args = ap.parse_args()
    Handler.workdir = Path(args.workdir)
    Handler.config = args.config
    print(f"review gate: http://127.0.0.1:{args.port}/  "
          f"(workdir={args.workdir})")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
