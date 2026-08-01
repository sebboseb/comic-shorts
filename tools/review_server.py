"""Interactive review gate: annotate shots, don't edit JSON.

Serves one page per episode: the rendered video on top, the shot timeline
under it. Click a shot (or just pause the video - the current shot is
auto-selected), hit a quick tag (Re-roll / Slomo / Cut) or write a note in
plain words ("remove the laugh", "say motherfucking hulk here"). Notes are
saved to work_*/review_notes.json as data; `python run.py --stage notes`
compiles them into manifest edits (pipeline/apply_notes.py) and the next
`--stage 6 7` picks them up.

Usage:
  ./review                # newest-rendered workdir, config inferred, browser opens
  ./review issue          # fuzzy-match a workdir ("issue" -> work_jeff_issue)
  ./review --workdir work_x --config config/x.yaml --port 8420   # explicit
"""
import argparse
import html
import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FPS = 30

PAGE = """<!doctype html><meta charset="utf-8">
<title>Review — {wd} {ep}</title>
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
#nav {{ padding: 10px 16px 0; font-size: 14px; }}
#nav a {{ color: #8ab4f8; text-decoration: none; margin-right: 14px; }}
#nav a.ep {{ padding: 2px 10px; background: #222; border-radius: 10px; }}
#nav a.ep.on {{ background: #ffd54a; color: #111; }}
</style>
<div id="nav"><a href="/">&larr; all videos</a> <b>{wd}</b> {ep_tabs}</div>
<div id="top">
  <video src="/media/{wd}/renders/{ep}.mp4" controls></video>
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
const WD = "{wd}";
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
  await fetch(`/notes?wd=${{WD}}`, {{ method: "POST", body: JSON.stringify(notes) }});
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
  await fetch(`/apply?wd=${{WD}}`, {{ method: "POST" }});
  const poll = setInterval(async () => {{
    const s = await (await fetch(`/apply/status?wd=${{WD}}`)).json();
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



GALLERY = """<!doctype html><meta charset="utf-8">
<title>comic-shorts — videos</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 0; background: #111;
       color: #eee; padding: 20px 24px; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 15px; color: #999; margin: 26px 0 10px; }}
#grid {{ display: flex; flex-wrap: wrap; gap: 14px; }}
.card {{ width: 240px; background: #1a1a1a; border: 1px solid #333;
  border-radius: 10px; overflow: hidden; cursor: pointer; text-decoration:
  none; color: #eee; display: block; }}
.card:hover {{ border-color: #ffd54a; }}
.card img {{ width: 100%; height: 150px; object-fit: cover; display: block;
  background: #000; }}
.card .body {{ padding: 10px 12px 12px; }}
.card .title {{ font-size: 13px; color: #ddd; height: 34px; overflow: hidden; }}
.card .id {{ font-size: 12px; color: #888; }}
.badge {{ display: inline-block; font-size: 11px; padding: 2px 9px;
  border-radius: 9px; margin-top: 8px; }}
.rendered {{ background: #1d3a1d; color: #8fdc8f; }}
.voicing {{ background: #3a2f14; color: #ffd54a; }}
.scripted {{ background: #222; color: #aaa; }}
.notes {{ background: #3a1919; color: #f09090; margin-left: 6px; }}
#finals {{ font-size: 13px; color: #999; line-height: 1.7; }}
</style>
<h1>Videos</h1>
{sections}
<h2>finals/ (shipped snapshots)</h2>
<div id="finals">{finals}</div>
"""


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
            "img": f"/media/{workdir.name}/" + panel.get("clean_file", ""),
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


def _workdirs():
    return sorted((d for d in Path(".").glob("work*") if (d / "manifests").is_dir()),
                  key=lambda d: d.name)


def _ep_status(wd, ep):
    manifest = json.loads((wd / "manifests" / f"{ep}.json").read_text())
    shots = manifest.get("shots", [])
    wavs = len(list((wd / "audio" / ep).glob("*.wav"))) if (wd / "audio" / ep).is_dir() else 0
    render = wd / "renders" / f"{ep}.mp4"
    if render.exists():
        dur = sum(s.get("duration_frames") or 0 for s in shots) / FPS
        state = ("rendered", f"rendered · {dur:.0f}s · "
                 + time.strftime("%H:%M", time.localtime(render.stat().st_mtime)))
    elif wavs:
        state = ("voicing", f"voicing {wavs}/{len(shots)}")
    else:
        state = ("scripted", f"scripted · {len(shots)} shots")
    thumb = ""
    try:
        clean = {c["id"]: c for c in json.loads((wd / "clean.json").read_text())}
        first = clean.get(shots[0]["panel"]) if shots else None
        if first:
            thumb = f"/media/{wd.name}/" + first["clean_file"]
    except FileNotFoundError:
        pass
    notes_file = wd / "review_notes.json"
    n_notes = 0
    if notes_file.exists():
        n_notes = len(json.loads(notes_file.read_text()).get(ep, {}))
    return manifest.get("title", ""), state, thumb, n_notes


def _gallery():
    sections = []
    for wd in _workdirs():
        cards = []
        for mpath in sorted((wd / "manifests").glob("ep*.json")):
            ep = mpath.stem
            title, (cls, label), thumb, n_notes = _ep_status(wd, ep)
            notes_badge = (f"<span class='badge notes'>{n_notes} note"
                           f"{'s' if n_notes != 1 else ''}</span>" if n_notes else "")
            img = f"<img src='{thumb}' loading='lazy'>" if thumb else "<img>"
            cards.append(
                f"<a class='card' href='/review?wd={wd.name}&ep={ep}'>{img}"
                f"<div class='body'><div class='id'>{wd.name} · {ep}</div>"
                f"<div class='title'>{html.escape(title)}</div>"
                f"<span class='badge {cls}'>{label}</span>{notes_badge}"
                f"</div></a>")
        if cards:
            sections.append(f"<h2>{wd.name}</h2><div id='grid'>"
                            + "".join(cards) + "</div>")
    finals = Path.home() / "Dev/comicops/finals"
    flist = ""
    if finals.is_dir():
        rows = sorted(finals.glob("*.mp4"), key=lambda f: -f.stat().st_mtime)
        flist = "<br>".join(
            f"{f.name} — {time.strftime('%b %d %H:%M', time.localtime(f.stat().st_mtime))}"
            for f in rows[:12])
    return GALLERY.format(sections="".join(sections), finals=flist)


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

    def _wd(self):
        """Workdir from ?wd=, validated against the real set; falls back to
        the startup default so pre-gallery URLs keep working."""
        q = re.search(r"[?&]wd=([\w-]+)", self.path)
        if q:
            wd = Path(q.group(1))
            if wd in _workdirs():
                return wd
            return None
        return self.workdir

    def do_POST(self):
        wd = self._wd()
        if wd is None:
            return self._send(404, b"unknown workdir")
        if self.path.startswith("/apply"):
            if APPLY["running"]:
                return self._send(409, b"already running", "text/plain")
            APPLY.update(running=True, ok=None, tail="")
            threading.Thread(target=_apply_worker,
                             args=(wd, _pick_config(wd)),
                             daemon=True).start()
            return self._send(200, b"started", "text/plain")
        if not self.path.startswith("/notes"):
            return self._send(404, b"nope")
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        json.loads(raw)  # validate before persisting
        (wd / "review_notes.json").write_bytes(raw)
        self._send(200, b"ok", "text/plain")

    def do_GET(self):
        if self.path.startswith("/apply/status"):
            return self._send(200, json.dumps(APPLY).encode(),
                              "application/json")
        m = re.match(r"^/media/([\w-]+)/(.+)$", self.path)
        if m:
            wd = Path(m.group(1))
            if wd not in _workdirs():
                return self._send(404, b"unknown workdir")
            return self._media(wd, wd / m.group(2))
        if self.path == "/" or self.path.startswith("/?"):
            return self._send(200, _gallery().encode())
        if self.path.startswith("/review"):
            wd = self._wd()
            if wd is None:
                return self._send(404, b"unknown workdir")
            eps = sorted(p.stem for p in (wd / "manifests").glob("ep*.json"))
            q = re.search(r"[?&]ep=(\w+)", self.path)
            ep = q.group(1) if q and q.group(1) in eps else eps[0]
            tabs = " ".join(
                f"<a class='ep{' on' if e == ep else ''}' "
                f"href='/review?wd={wd.name}&ep={e}'>{e}</a>" for e in eps)
            notes_file = wd / "review_notes.json"
            notes = notes_file.read_text() if notes_file.exists() else "{}"
            page = PAGE.format(ep=ep, wd=wd.name, ep_tabs=tabs,
                               notes_json=notes,
                               shots_json=json.dumps(_shots_payload(wd, ep)))
            return self._send(200, page.encode())
        return self._send(404, b"nope")

    def _media(self, wd, path):
        path = path.resolve()
        if not (path.is_file() and wd.resolve() in path.parents):
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


def _pick_workdir(hint):
    """No flags to remember: any workdir with manifests qualifies; the hint
    fuzzy-matches the name; ties go to the most recently rendered."""
    cands = [d for d in Path(".").glob("work*")
             if (d / "manifests").is_dir()]
    if hint:
        cands = [d for d in cands if hint in d.name]
    if not cands:
        raise SystemExit(f"no workdir matches {hint!r} "
                         "(need a work*/ dir with manifests/)")
    def freshness(d):
        r = d / "renders"
        files = list(r.glob("*.mp4")) if r.is_dir() else []
        return max((f.stat().st_mtime for f in files), default=0)
    return max(cands, key=freshness)


def _pick_config(workdir):
    """Convention: work_jeff_issue -> config/jeff_issue.yaml."""
    guess = Path("config") / (workdir.name.removeprefix("work_") + ".yaml")
    return str(guess) if guess.exists() else "config/comic.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hint", nargs="?", default=None,
                    help="part of a workdir name, e.g. 'issue'")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    workdir = Path(args.workdir) if args.workdir else _pick_workdir(args.hint)
    Handler.workdir = workdir
    Handler.config = args.config or _pick_config(workdir)
    base = f"http://127.0.0.1:{args.port}"
    url = (f"{base}/review?wd={workdir.name}"
           if (args.hint or args.workdir) else f"{base}/")
    print(f"review gate: {url}")
    if not args.no_open:
        import webbrowser
        threading.Timer(0.6, webbrowser.open, [url]).start()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
