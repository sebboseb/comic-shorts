"""Face-tagging UI for the character roster.

Run:  python tools/tag_ui.py        (from the repo root)

Opens a local page where you drag a box around a face on any comic page,
assign it to a character, and the tool:
  - saves the crop to config/characters/<slug>.png
  - updates (or appends) the character's entry in config/comic.yaml,
    preserving comments and formatting via targeted text edits.

No dependencies beyond Pillow + PyYAML (already in requirements.txt).
"""

import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = ROOT / "pages"
CROPS_DIR = ROOT / "config" / "characters"
CONFIG_PATH = ROOT / "config" / "comic.yaml"
OVERRIDES_PATH = ROOT / "config" / "panel_overrides.json"
WORK_PANELS = ROOT / "work" / "panels.json"
PORT = 8765

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp"}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unnamed"


def list_pages():
    return sorted(p.name for p in PAGES_DIR.iterdir()
                  if p.suffix.lower() in IMAGE_EXTS)


def load_state():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    characters = []
    for ch in config.get("characters", []):
        ref = ch.get("ref_image")
        has_crop = bool(ref) and (CROPS_DIR / ref).exists()
        characters.append({
            "name": ch["name"],
            "description": ch.get("description", ""),
            "ref_image": ref,
            "has_crop": has_crop,
        })
    return {"pages": list_pages(), "characters": characters,
            "title": config.get("comic", {}).get("title", "Comic")}


def _entry_block_pattern(name: str):
    # Matches one roster entry starting at `- name: "<name>"` up to the next
    # entry or the next top-level key.
    return re.compile(
        r'([ \t]*-[ \t]+name:[ \t]*"?' + re.escape(name) + r'"?[ \t]*\n'
        r'(?:(?![ \t]*-[ \t]+name:)[ \t]+\S.*\n?)*)',
    )


def set_ref_image(name: str, ref_filename: str) -> bool:
    """Point an existing roster entry's ref_image at ref_filename."""
    text = CONFIG_PATH.read_text()
    m = _entry_block_pattern(name).search(text)
    if not m:
        return False
    block = m.group(1)
    if re.search(r"^[ \t]+ref_image:", block, re.M):
        new_block = re.sub(r"^([ \t]+ref_image:).*$",
                           rf'\1 "{ref_filename}"', block, count=1, flags=re.M)
    else:
        new_block = block.rstrip("\n") + f'\n    ref_image: "{ref_filename}"\n'
    CONFIG_PATH.write_text(text[:m.start(1)] + new_block + text[m.end(1):])
    return True


def append_character(name: str, ref_filename: str):
    """Insert a new roster entry at the end of the characters: block."""
    text = CONFIG_PATH.read_text()
    lines = text.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines)
                  if re.match(r"^characters:\s*$", ln)), None)
    if start is None:
        raise ValueError("no 'characters:' block in comic.yaml")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if re.match(r"^[A-Za-z_]", lines[i]):
            end = i
            break
    # back up over trailing blank/comment lines so the entry joins the list
    while end > start + 1 and lines[end - 1].strip() in ("",) :
        end -= 1
    entry = (f'  - name: "{name}"\n'
             f'    description: "TODO: describe {name}"\n'
             f'    ref_image: "{ref_filename}"\n'
             f'    voice_id: "ELEVENLABS_VOICE_ID_HERE"\n'
             f'    speaking_style: "neutral"\n')
    lines.insert(end, entry)
    CONFIG_PATH.write_text("".join(lines))


def save_crop(page: str, box: dict, name: str) -> dict:
    src = PAGES_DIR / page
    if not src.exists():
        raise ValueError(f"unknown page {page}")
    img = Image.open(src).convert("RGB")
    x, y = int(box["x"]), int(box["y"])
    w, h = int(box["w"]), int(box["h"])
    x = max(0, min(x, img.width - 1))
    y = max(0, min(y, img.height - 1))
    w = max(1, min(w, img.width - x))
    h = max(1, min(h, img.height - y))
    crop = img.crop((x, y, x + w, y + h))
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(CONFIG_PATH.read_text())
    existing = next((c for c in config.get("characters", [])
                     if c["name"].lower() == name.lower()), None)
    if existing:
        name = existing["name"]  # canonical casing
        filename = existing.get("ref_image") or f"{slugify(name)}.png"
        crop.save(CROPS_DIR / filename)
        set_ref_image(name, filename)
    else:
        filename = f"{slugify(name)}.png"
        crop.save(CROPS_DIR / filename)
        append_character(name, filename)
    return {"name": name, "ref_image": filename}


def load_overrides() -> dict:
    if OVERRIDES_PATH.exists():
        return json.loads(OVERRIDES_PATH.read_text())
    return {}


def save_overrides(overrides: dict):
    OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2))


def auto_boxes_for(page: str):
    """Boxes stage 1 detected for this page (for reference display)."""
    if not WORK_PANELS.exists():
        return []
    data = json.loads(WORK_PANELS.read_text())
    pages = list_pages()
    try:
        page_no = pages.index(page) + 1
    except ValueError:
        return []
    return [p["bbox_on_page"] for p in data.get("panels", [])
            if p["page"] == page_no]


def delete_crop(name: str) -> bool:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    ch = next((c for c in config.get("characters", [])
               if c["name"] == name), None)
    if not ch or not ch.get("ref_image"):
        return False
    path = CROPS_DIR / ch["ref_image"]
    if path.exists():
        path.unlink()
    return True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path):
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type",
                         MIME.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import parse_qs
        self.path, _, query = self.path.partition("?")
        params = parse_qs(query)
        if self.path == "/api/panels":
            page = params.get("page", [""])[0]
            self._json({"manual": load_overrides().get(page, []),
                        "auto": auto_boxes_for(page)})
            return
        if self.path == "/" or self.path.startswith("/index"):
            body = PAGE_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            self._json(load_state())
        elif self.path.startswith("/page/"):
            self._file(PAGES_DIR / Path(self.path[len("/page/"):]).name)
        elif self.path.startswith("/crop/"):
            self._file(CROPS_DIR / Path(self.path[len("/crop/"):]).name)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length))
            if self.path == "/api/tag":
                result = save_crop(data["page"], data["box"],
                                   data["name"].strip())
                self._json({"ok": True, **result})
            elif self.path == "/api/delete_crop":
                self._json({"ok": delete_crop(data["name"])})
            elif self.path == "/api/panels":
                overrides = load_overrides()
                boxes = data.get("boxes", [])
                if boxes:
                    overrides[data["page"]] = boxes
                else:
                    overrides.pop(data["page"], None)
                save_overrides(overrides)
                self._json({"ok": True, "count": len(boxes)})
            else:
                self.send_error(404)
        except Exception as exc:  # surface errors to the UI
            self._json({"ok": False, "error": str(exc)}, status=400)


PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face Tagger</title>
<style>
  :root {
    --bg: #101014; --panel: #17171d; --line: #26262e;
    --text: #e8e8ec; --dim: #85858f; --accent: #6ea8fe; --ok: #4ade80;
    --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
  }
  * { box-sizing: border-box; margin: 0; }
  html, body { height: 100%; }
  body {
    background: var(--bg); color: var(--text); overflow: hidden;
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    display: grid; grid-template-columns: 260px 1fr; grid-template-rows: 48px 1fr;
    grid-template-areas: "side head" "side main";
  }

  header {
    grid-area: head; display: flex; align-items: center; gap: 12px;
    padding: 0 14px; border-bottom: 1px solid var(--line); background: var(--panel);
  }
  header .pageno { font-variant-numeric: tabular-nums; color: var(--dim); }
  .strip { display: flex; gap: 3px; overflow-x: auto; scrollbar-width: none; }
  .strip::-webkit-scrollbar { display: none; }
  .strip button {
    min-width: 26px; height: 26px; padding: 0 4px; border-radius: 6px;
    border: 1px solid transparent; background: transparent; color: var(--dim);
    font: inherit; font-size: 12px; cursor: pointer;
    transition: transform 120ms var(--ease-out);
  }
  .strip button:active { transform: scale(0.94); }
  .strip button.active { background: var(--accent); color: #0b1220; font-weight: 600; }
  @media (hover: hover) and (pointer: fine) {
    .strip button:hover:not(.active) { border-color: var(--line); color: var(--text); }
  }
  header .hint { margin-left: auto; color: var(--dim); font-size: 12px; white-space: nowrap; }
  header kbd {
    background: var(--line); border-radius: 4px; padding: 1px 5px;
    font: 11px ui-monospace, monospace;
  }

  aside {
    grid-area: side; border-right: 1px solid var(--line); background: var(--panel);
    display: flex; flex-direction: column; min-height: 0;
  }
  aside h1 { font-size: 13px; font-weight: 600; padding: 14px 14px 4px; }
  aside .sub { color: var(--dim); font-size: 12px; padding: 0 14px 10px; }
  .roster { overflow-y: auto; flex: 1; padding: 0 8px 12px; }
  .char {
    display: flex; align-items: center; gap: 10px; width: 100%;
    padding: 7px 8px; border-radius: 8px; border: 1px solid transparent;
    background: transparent; color: inherit; font: inherit; text-align: left;
    cursor: pointer; position: relative;
    transition: transform 120ms var(--ease-out);
  }
  .char:active { transform: scale(0.98); }
  @media (hover: hover) and (pointer: fine) {
    .char:hover { background: #1e1e26; }
  }
  .char.armed { border-color: var(--accent); background: #1a2233; }
  .char .thumb {
    width: 40px; height: 40px; border-radius: 6px; object-fit: cover;
    background: var(--line); flex: none;
  }
  .char .thumb.empty {
    display: flex; align-items: center; justify-content: center;
    border: 1px dashed #3a3a44; color: var(--dim); font-size: 16px;
  }
  .char .meta { min-width: 0; }
  .char .name { font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .char .status { font-size: 11px; color: var(--dim); }
  .char .status.done { color: var(--ok); }
  .char .del {
    position: absolute; right: 6px; top: 50%; translate: 0 -50%;
    width: 20px; height: 20px; border-radius: 5px; border: none;
    background: var(--line); color: var(--dim); font-size: 11px; cursor: pointer;
    opacity: 0; transition: opacity 120ms var(--ease-out);
  }
  @media (hover: hover) and (pointer: fine) {
    .char:hover .del { opacity: 1; }
    .char .del:hover { color: #f87171; }
  }
  .char .thumb.flash { animation: flash 600ms var(--ease-out); }
  @keyframes flash {
    0% { outline: 2px solid var(--ok); outline-offset: 1px; }
    100% { outline: 2px solid transparent; outline-offset: 4px; }
  }
  aside .foot {
    padding: 10px 14px; border-top: 1px solid var(--line);
    color: var(--dim); font-size: 11.5px;
  }

  main { grid-area: main; overflow: auto; position: relative; }
  #stage { position: relative; width: max-content; margin: 24px auto; }
  #pageimg { display: block; cursor: crosshair; user-select: none; -webkit-user-drag: none; }
  #box {
    position: absolute; border: 1.5px solid var(--accent); border-radius: 2px;
    box-shadow: 0 0 0 9999px rgba(8, 8, 12, 0.45); pointer-events: none;
    display: none;
  }

  #pop {
    position: absolute; z-index: 10; width: 230px;
    background: #1c1c24; border: 1px solid #32323c; border-radius: 10px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.5); padding: 8px; display: none;
    opacity: 1; transform: scale(1);
    transition: opacity 140ms var(--ease-out), transform 140ms var(--ease-out);
  }
  #pop.entering { opacity: 0; transform: scale(0.97); }
  #pop input {
    width: 100%; padding: 7px 9px; border-radius: 7px;
    border: 1px solid #32323c; background: #12121a; color: var(--text);
    font: inherit; outline: none;
  }
  #pop input:focus { border-color: var(--accent); }
  #chips { margin-top: 6px; display: flex; flex-direction: column; gap: 2px; }
  .chip {
    display: flex; align-items: center; gap: 8px; padding: 5px 7px;
    border-radius: 6px; border: none; background: transparent; color: var(--text);
    font: inherit; font-size: 13px; cursor: pointer; text-align: left;
    transition: transform 100ms var(--ease-out);
  }
  .chip:active { transform: scale(0.97); }
  .chip.sel { background: #26304a; }
  .chip .n {
    width: 16px; height: 16px; border-radius: 4px; background: var(--line);
    color: var(--dim); font: 10px ui-monospace, monospace;
    display: flex; align-items: center; justify-content: center; flex: none;
  }
  .chip .new { color: var(--accent); }

  .seg { display: flex; background: var(--line); border-radius: 8px; padding: 2px; }
  .seg button {
    padding: 4px 12px; border: none; border-radius: 6px; background: transparent;
    color: var(--dim); font: inherit; font-size: 12.5px; cursor: pointer;
    transition: transform 120ms var(--ease-out);
  }
  .seg button:active { transform: scale(0.96); }
  .seg button.active { background: var(--panel); color: var(--text); font-weight: 600; }

  #panellayer { position: absolute; inset: 0; pointer-events: none; display: none; }
  body.panels #panellayer { display: block; }
  .pbox { position: absolute; border: 2px solid #f59e0b; border-radius: 2px;
          background: rgba(245, 158, 11, 0.08); }
  .pbox.auto { border: 1px dashed rgba(255,255,255,0.35); background: none; }
  .pbox .badge {
    position: absolute; top: -1px; left: -1px; min-width: 22px; height: 22px;
    display: flex; align-items: center; justify-content: center;
    background: #f59e0b; color: #1a1206; font-size: 12px; font-weight: 700;
    border-radius: 2px 0 6px 0; cursor: pointer; pointer-events: auto;
    transition: transform 100ms var(--ease-out);
  }
  @media (hover: hover) and (pointer: fine) {
    .pbox .badge:hover { background: #f87171; color: #200; }
  }
  .pbox .badge:active { transform: scale(0.9); }

  #toast {
    position: fixed; bottom: 18px; left: 50%; z-index: 20;
    background: #1c1c24; border: 1px solid #32323c; border-radius: 9px;
    padding: 8px 14px; font-size: 13px; pointer-events: none;
    opacity: 0; transform: translate(-50%, 6px);
    transition: opacity 160ms var(--ease-out), transform 160ms var(--ease-out);
  }
  #toast.show { opacity: 1; transform: translate(-50%, 0); }
  #toast.err { border-color: #7f1d1d; color: #fca5a5; }

  @media (prefers-reduced-motion: reduce) {
    * { transition-duration: 0ms !important; animation: none !important; }
  }
</style>
</head>
<body>
<aside>
  <h1 id="title">Face Tagger</h1>
  <div class="sub">Click a character to arm it, then drag a box around their face. Or just drag and pick a name.</div>
  <div class="roster" id="roster"></div>
  <div class="foot">
    <kbd>a</kbd><kbd>d</kbd> or <kbd>←</kbd><kbd>→</kbd> pages &nbsp; <kbd>⌘</kbd>+scroll zoom &nbsp; <kbd>esc</kbd> cancel
  </div>
</aside>
<header>
  <div class="seg">
    <button id="modeface" class="active">Faces</button>
    <button id="modepanel">Panels</button>
  </div>
  <span class="pageno" id="pageno"></span>
  <div class="strip" id="strip"></div>
  <span class="hint" id="hint">drag on the page to crop a face</span>
</header>
<main id="main">
  <div id="stage">
    <img id="pageimg" draggable="false" alt="">
    <div id="panellayer"></div>
    <div id="box"></div>
    <div id="pop">
      <input id="nameinput" type="text" placeholder="Character name…" autocomplete="off" spellcheck="false">
      <div id="chips"></div>
    </div>
  </div>
</main>
<div id="toast"></div>

<script>
const $ = (id) => document.getElementById(id);
let state = { pages: [], characters: [] };
let cur = 0, zoom = 1, natW = 0, natH = 0;
let drag = null, pendingBox = null, armed = null, sel = -1;
let mode = 'face', manualBoxes = [], autoBoxes = [];

function setMode(m) {
  mode = m;
  document.body.classList.toggle('panels', m === 'panel');
  $('modeface').className = m === 'face' ? 'active' : '';
  $('modepanel').className = m === 'panel' ? 'active' : '';
  $('hint').textContent = m === 'face'
    ? 'drag on the page to crop a face'
    : 'draw panels in reading order — click a number to delete';
  cancelBox();
  if (m === 'panel') fetchPanels();
}
$('modeface').onclick = () => setMode('face');
$('modepanel').onclick = () => setMode('panel');

async function fetchPanels() {
  const r = await (await fetch('/api/panels?page=' + encodeURIComponent(state.pages[cur]))).json();
  manualBoxes = r.manual; autoBoxes = r.auto;
  renderPanelLayer();
}

function renderPanelLayer() {
  const el = $('panellayer'); el.innerHTML = '';
  if (mode !== 'panel') return;
  const showAuto = manualBoxes.length === 0;
  if (showAuto) for (const b of autoBoxes) {
    const d = document.createElement('div');
    d.className = 'pbox auto';
    d.style.cssText = `left:${b[0]*zoom}px;top:${b[1]*zoom}px;width:${b[2]*zoom}px;height:${b[3]*zoom}px`;
    el.appendChild(d);
  }
  manualBoxes.forEach((b, i) => {
    const d = document.createElement('div');
    d.className = 'pbox';
    d.style.cssText = `left:${b[0]*zoom}px;top:${b[1]*zoom}px;width:${b[2]*zoom}px;height:${b[3]*zoom}px`;
    const badge = document.createElement('div');
    badge.className = 'badge'; badge.textContent = i + 1;
    badge.title = 'delete panel ' + (i + 1);
    badge.onclick = () => { manualBoxes.splice(i, 1); postPanels(); };
    d.appendChild(badge);
    el.appendChild(d);
  });
}

async function postPanels() {
  await fetch('/api/panels', { method: 'POST', body: JSON.stringify({
    page: state.pages[cur], boxes: manualBoxes }) });
  renderPanelLayer();
  toast(manualBoxes.length
    ? `${manualBoxes.length} panels saved for this page`
    : 'override cleared — auto-detection applies');
}

async function fetchState() {
  state = await (await fetch('/api/state')).json();
  $('title').textContent = state.title + ' — Face Tagger';
  renderRoster(); renderStrip();
}

function renderRoster(flashName) {
  const el = $('roster'); el.innerHTML = '';
  for (const ch of state.characters) {
    const b = document.createElement('button');
    b.className = 'char' + (armed === ch.name ? ' armed' : '');
    const t = ch.has_crop
      ? `<img class="thumb${ch.name === flashName ? ' flash' : ''}" src="/crop/${ch.ref_image}?${Date.now()}">`
      : `<div class="thumb empty">+</div>`;
    b.innerHTML = t + `<div class="meta"><div class="name">${ch.name}</div>
      <div class="status ${ch.has_crop ? 'done' : ''}">${ch.has_crop ? 'tagged' : 'needs face'}</div></div>`
      + (ch.has_crop ? '<button class="del" title="remove crop">✕</button>' : '');
    b.onclick = (e) => {
      if (e.target.classList.contains('del')) { removeCrop(ch.name); return; }
      armed = armed === ch.name ? null : ch.name;
      renderRoster();
      toast(armed ? `Armed: ${armed} — draw a box` : 'Disarmed');
    };
    el.appendChild(b);
  }
}

function renderStrip() {
  const el = $('strip'); el.innerHTML = '';
  state.pages.forEach((p, i) => {
    const b = document.createElement('button');
    b.textContent = i + 1;
    b.className = i === cur ? 'active' : '';
    b.onclick = () => showPage(i);
    el.appendChild(b);
  });
  $('pageno').textContent = state.pages.length ? `${cur + 1}/${state.pages.length}` : 'no pages';
}

function showPage(i) {
  if (!state.pages.length) return;
  cur = (i + state.pages.length) % state.pages.length;
  cancelBox();
  const img = $('pageimg');
  img.onload = () => {
    natW = img.naturalWidth; natH = img.naturalHeight;
    fitZoom(); applyZoom();
  };
  img.src = '/page/' + state.pages[cur];
  renderStrip();
  if (mode === 'panel') fetchPanels();
}

function fitZoom() {
  const m = $('main');
  zoom = Math.min(1, (m.clientWidth - 48) / natW, (m.clientHeight - 48) / natH);
}
function applyZoom() {
  const img = $('pageimg');
  img.style.width = (natW * zoom) + 'px';
  img.style.height = (natH * zoom) + 'px';
  renderPanelLayer();
}

// --- drag to draw box ---
const stage = $('stage');
function evtPos(e) {
  const r = $('pageimg').getBoundingClientRect();
  return { x: Math.min(Math.max(e.clientX - r.left, 0), r.width),
           y: Math.min(Math.max(e.clientY - r.top, 0), r.height) };
}
$('pageimg').addEventListener('pointerdown', (e) => {
  if (e.button !== 0 || drag) return;
  e.preventDefault(); hidePop();
  drag = { start: evtPos(e) };
  $('pageimg').setPointerCapture(e.pointerId);
});
$('pageimg').addEventListener('pointermove', (e) => {
  if (!drag) return;
  const p = evtPos(e), s = drag.start;
  const b = { x: Math.min(s.x, p.x), y: Math.min(s.y, p.y),
              w: Math.abs(p.x - s.x), h: Math.abs(p.y - s.y) };
  drag.box = b; drawBox(b);
});
$('pageimg').addEventListener('pointerup', () => {
  if (!drag) return;
  const b = drag.box; drag = null;
  if (!b || b.w < 8 || b.h < 8) { cancelBox(); return; }
  if (mode === 'panel') {
    manualBoxes.push([Math.round(b.x / zoom), Math.round(b.y / zoom),
                      Math.round(b.w / zoom), Math.round(b.h / zoom)]);
    $('box').style.display = 'none';
    postPanels();
    return;
  }
  pendingBox = b;
  if (armed) { submit(armed); } else { showPop(b); }
});

function drawBox(b) {
  const el = $('box');
  el.style.display = 'block';
  el.style.left = b.x + 'px'; el.style.top = b.y + 'px';
  el.style.width = b.w + 'px'; el.style.height = b.h + 'px';
}
function cancelBox() {
  $('box').style.display = 'none'; pendingBox = null; hidePop();
}

// --- naming popover ---
function showPop(b) {
  const pop = $('pop');
  const stageW = natW * zoom;
  pop.style.left = Math.min(b.x, stageW - 240) + 'px';
  pop.style.top = (b.y + b.h + 10) + 'px';
  pop.style.display = 'block';
  pop.classList.add('entering');
  requestAnimationFrame(() => requestAnimationFrame(() => pop.classList.remove('entering')));
  $('nameinput').value = ''; sel = -1; renderChips('');
  $('nameinput').focus();
}
function hidePop() { $('pop').style.display = 'none'; }

function chipMatches(q) {
  q = q.toLowerCase();
  return state.characters.filter(c => c.name.toLowerCase().includes(q));
}
function renderChips(q) {
  const el = $('chips'); el.innerHTML = '';
  const matches = chipMatches(q).slice(0, 9);
  matches.forEach((c, i) => {
    const b = document.createElement('button');
    b.className = 'chip' + (i === sel ? ' sel' : '');
    b.innerHTML = `<span class="n">${i + 1}</span>${c.name}` +
                  (c.has_crop ? ' <span style="color:var(--ok)">•</span>' : '');
    b.onclick = () => submit(c.name);
    el.appendChild(b);
  });
  if (q && !matches.some(c => c.name.toLowerCase() === q.toLowerCase())) {
    const b = document.createElement('button');
    b.className = 'chip';
    b.innerHTML = `<span class="n">↵</span><span class="new">Add “${q}”</span>`;
    b.onclick = () => submit(q);
    el.appendChild(b);
  }
}

$('nameinput').addEventListener('input', (e) => { sel = -1; renderChips(e.target.value); });
$('nameinput').addEventListener('keydown', (e) => {
  const matches = chipMatches(e.target.value).slice(0, 9);
  if (e.key === 'Enter') {
    e.preventDefault();
    if (sel >= 0 && matches[sel]) submit(matches[sel].name);
    else if (e.target.value.trim()) submit(e.target.value.trim());
    else if (matches.length === 1) submit(matches[0].name);
  } else if (e.key === 'ArrowDown') { e.preventDefault(); sel = Math.min(sel + 1, matches.length - 1); renderChips(e.target.value); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); sel = Math.max(sel - 1, -1); renderChips(e.target.value); }
  else if (/^[1-9]$/.test(e.key) && !e.target.value) {
    const c = matches[Number(e.key) - 1]; if (c) { e.preventDefault(); submit(c.name); }
  } else if (e.key === 'Escape') { cancelBox(); }
  e.stopPropagation();
});

async function submit(name) {
  if (!pendingBox) return;
  const b = pendingBox;
  const payload = {
    page: state.pages[cur], name,
    box: { x: Math.round(b.x / zoom), y: Math.round(b.y / zoom),
           w: Math.round(b.w / zoom), h: Math.round(b.h / zoom) },
  };
  cancelBox();
  const res = await (await fetch('/api/tag', {
    method: 'POST', body: JSON.stringify(payload),
  })).json();
  if (!res.ok) { toast(res.error || 'failed', true); return; }
  const wasArmed = armed === res.name;
  armed = null;
  await fetchState();
  renderRoster(res.name);
  toast(`Saved face for ${res.name}` + (wasArmed ? '' : ''));
}

async function removeCrop(name) {
  await fetch('/api/delete_crop', { method: 'POST', body: JSON.stringify({ name }) });
  await fetchState();
  toast(`Removed crop for ${name}`);
}

// --- toast ---
let toastTimer = null;
function toast(msg, err) {
  const t = $('toast');
  t.textContent = msg; t.className = 'show' + (err ? ' err' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.className = '', 1800);
}

// --- keyboard + zoom ---
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight' || e.key === 'd') showPage(cur + 1);
  else if (e.key === 'ArrowLeft' || e.key === 'a') showPage(cur - 1);
  else if (e.key === 'Escape') { armed = null; renderRoster(); cancelBox(); }
  else if (e.key === 'm') setMode(mode === 'face' ? 'panel' : 'face');
  else if (e.key === '+' || e.key === '=') { zoom = Math.min(zoom * 1.2, 4); applyZoom(); }
  else if (e.key === '-') { zoom = Math.max(zoom / 1.2, 0.1); applyZoom(); }
  else if (e.key === '0') { fitZoom(); applyZoom(); }
});
$('main').addEventListener('wheel', (e) => {
  if (!e.ctrlKey && !e.metaKey) return;
  e.preventDefault();
  const m = $('main'), r = m.getBoundingClientRect();
  const px = m.scrollLeft + (e.clientX - r.left);
  const py = m.scrollTop + (e.clientY - r.top);
  const old = zoom;
  // delta-proportional: gentle for trackpad pinch (small deltas), capped for
  // mouse wheels (|deltaY| ~100 per notch)
  const factor = Math.min(1.15, Math.max(1 / 1.15, Math.exp(-e.deltaY * 0.004)));
  zoom = Math.min(4, Math.max(0.1, zoom * factor));
  applyZoom();
  m.scrollLeft = px * (zoom / old) - (e.clientX - r.left);
  m.scrollTop = py * (zoom / old) - (e.clientY - r.top);
}, { passive: false });

fetchState().then(() => showPage(0));
</script>
</body>
</html>
"""


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Face tagger running at {url}  (ctrl-c to stop)")
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
