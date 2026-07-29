# Plan 001: Establish a verification baseline (git + pytest)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: This repo is NOT under version control yet
> (this plan fixes that), so there is no SHA to diff against. Instead,
> compare every excerpt in "Current state" against the live files before
> starting; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW (adds tests and VCS; touches no pipeline logic)
- **Depends on**: none
- **Category**: tests / dx
- **Planned at**: no VCS (repo not under git when planned), 2026-07-17

## Why this matters

This repo has zero tests, no lint, and — most importantly — is not a git
repository, even though its documented workflow is "hand-edit generated
files": the README tells the user to fix `work/panels.json`,
`work/manifests/epNN.json`, and `tools/tag_ui.py` performs regex-based
surgery on `config/comic.yaml`. One bad edit (human or tool) is currently
unrecoverable. Every other plan in `plans/` assumes it can verify changes
with a test command and commit incrementally; this plan creates that
foundation. It changes no runtime behavior.

## Current state

- Repo root: `/Users/seb/Downloads/comic-shorts` (referred to as `.` below).
- `git status` fails with "not a git repository".
- No `tests/` directory, no pytest in the venv:
  `.venv/bin/python -m pytest --version` → "No module named pytest".
- `.venv/bin/python --version` → `Python 3.14.6`.
- Installed in `.venv`: cv2 (opencv), numpy, PIL, yaml, anthropic.
  **torch is NOT installed on this machine** (stage 3 runs on a separate GPU
  box), so any test that imports `pipeline/stage3_cleanup.py` must be
  skipped gracefully when torch is missing.
- `requirements.txt` (complete current content):

  ```
  opencv-python
  numpy
  pillow
  pyyaml
  anthropic
  torch
  simple-lama-inpainting
  spandrel
  ```

- Pure functions worth testing (excerpts so you can confirm you're looking
  at the right code):

  `pipeline/stage1_panels.py:72` —

  ```python
  def reading_order(boxes, direction="ltr"):
      """Cluster boxes into rows by vertical center, then sort within rows."""
  ```

  `pipeline/stage1_panels.py:54` —

  ```python
  def detect_panels(img, white_threshold=235, dark_threshold=48,
                    min_area_ratio=0.02):
  ```

  `pipeline/stage2_understand.py:66-72` —

  ```python
  def _parse_json(text: str):
      text = text.strip()
      if text.startswith("```"):
          text = text.split("```")[1]
          if text.startswith("json"):
              text = text[4:]
      return json.loads(text)
  ```

  (an identical copy exists at `pipeline/stage4_story.py:67` — test both,
  do NOT deduplicate them in this plan)

  `pipeline/stage2_understand.py:75-78` —

  ```python
  def _norm_to_px(box, width, height):
      x, y, w, h = box
      return [round(x / 1000 * width), round(y / 1000 * height),
              round(w / 1000 * width), round(h / 1000 * height)]
  ```

  `tools/tag_ui.py:37-38` —

  ```python
  def slugify(name: str) -> str:
      return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unnamed"
  ```

  `tools/tag_ui.py:71` — `def set_ref_image(name, ref_filename)` and
  `tools/tag_ui.py:87` — `def append_character(name, ref_filename)`: both do
  targeted text edits on the YAML at the module constant
  `CONFIG_PATH = ROOT / "config" / "comic.yaml"` (`tools/tag_ui.py:27`).
  Tests must monkeypatch `tag_ui.CONFIG_PATH` to a temp file — never let a
  test touch the real `config/comic.yaml`.

- Repo conventions: plain functions, no classes in pipeline code, stdlib +
  minimal deps, 4-space indent, `_private` helper naming. Match this in
  tests: plain pytest functions, no test classes, no fixtures beyond
  `tmp_path`/`monkeypatch`.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Install dev deps | `.venv/bin/pip install -r requirements-dev.txt` | exit 0 |
| Tests | `.venv/bin/python -m pytest` | exit 0, all pass |
| Import sanity | `.venv/bin/python -c "import pipeline.stage1_panels"` | exit 0 |

There is no build, lint, or typecheck step in this repo. Do not add one in
this plan.

## Scope

**In scope** (the only files you should create/modify):
- `.gitignore` (create)
- `requirements-dev.txt` (create)
- `tests/conftest.py` (create)
- `tests/test_stage1_panels.py` (create)
- `tests/test_json_parsing.py` (create)
- `tests/test_tag_ui_yaml.py` (create)
- `tests/test_stage3_mask.py` (create — skips without torch)
- git repository initialization (`git init` + initial commit)

**Out of scope** (do NOT touch, even though they look related):
- Any file under `pipeline/`, `tools/`, `config/`, `pages/`, `work/` —
  this plan adds tests around existing behavior, it never changes it.
- `requirements.txt` — runtime deps stay as-is; dev deps go in the new
  `requirements-dev.txt`.
- No CI config — there is no remote; a local pytest run is the baseline.

## Git workflow

- `git init` on the default branch it creates; work directly on it (solo
  project, no remote).
- Commit messages: short imperative subject line (e.g. "Add pytest baseline
  for stage 1 helpers").
- Do NOT add a remote or push anywhere.

## Steps

### Step 1: Initialize git with a correct .gitignore

Create `.gitignore` at the repo root:

```
.venv/
__pycache__/
*.pyc
.DS_Store
work/
weights/
```

Rationale (do not deviate): `work/` is fully regenerable pipeline output,
`weights/` is a 17MB+ model download, `.venv/` is the environment. `pages/`
and `config/` (including `config/characters/*.png`) ARE tracked — they are
hand-made inputs the pipeline cannot regenerate.

Then:

```bash
git init
git add -A
git commit -m "Initial commit: stages 1-5 of comic-shorts pipeline"
```

**Verify**: `git status --short` → empty output; `git log --oneline` →
exactly 1 commit; `git ls-files | grep -c '^work/'` → `0`.

### Step 2: Add pytest as a dev dependency

Create `requirements-dev.txt`:

```
pytest
```

Install: `.venv/bin/pip install -r requirements-dev.txt`

**Verify**: `.venv/bin/python -m pytest --version` → prints a pytest
version, exit 0.

### Step 3: Make the repo importable from tests

Create `tests/conftest.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

(`pipeline` has an `__init__.py`; `tools` does not, but namespace-package
import of `tools.tag_ui` works once the root is on `sys.path`.)

**Verify**: `.venv/bin/python -m pytest --collect-only` → exit 0 (collects
0 tests, no errors).

### Step 4: Tests for stage-1 geometry helpers

Create `tests/test_stage1_panels.py` covering, with synthetic numpy images
(no fixture image files):

- `reading_order`: (a) a 2×2 grid of boxes returns top-left, top-right,
  bottom-left, bottom-right for `ltr`; (b) the same grid with `rtl`
  reverses within rows; (c) boxes with slightly different heights on the
  same visual row still cluster into one row (vertical centers within half
  the max height); (d) empty list returns empty list.
- `detect_panels`: build a 400×400 white uint8 BGR image
  (`np.full((400, 400, 3), 255, np.uint8)`), draw two dark filled
  rectangles separated by a white gutter (e.g. `img[20:180, 20:380] = 30`
  and `img[220:380, 20:380] = 30`), assert exactly 2 boxes come back and
  their y-order is correct.
- `_layout_score`: empty boxes → `-1.0`; a single box covering >90% of the
  page scores lower than two half-page boxes.

**Verify**: `.venv/bin/python -m pytest tests/test_stage1_panels.py -q` →
all pass.

### Step 5: Tests for the JSON parsing + coordinate helpers

Create `tests/test_json_parsing.py`. Parameterize over BOTH copies of
`_parse_json` (`pipeline.stage2_understand._parse_json` and
`pipeline.stage4_story._parse_json` — import both; they are intentionally
duplicated today):

- plain JSON object string → parsed dict
- ` ```json\n{...}\n``` ` fenced → parsed dict
- ` ```\n{...}\n``` ` fenced without the `json` tag → parsed dict
- garbage string → raises `json.JSONDecodeError`

And for `pipeline.stage2_understand._norm_to_px`:

- `[0, 0, 1000, 1000]` on a 800×600 image → `[0, 0, 800, 600]`
- `[500, 500, 100, 100]` on 1000×2000 → `[500, 1000, 100, 200]`
- rounding: `[333, 0, 334, 0]` on 3×3 → `[1, 0, 1, 0]`

Note: importing `pipeline.stage2_understand` imports the `anthropic`
package at module level but creates no client — safe without an API key.

**Verify**: `.venv/bin/python -m pytest tests/test_json_parsing.py -q` →
all pass.

### Step 6: Tests for tag_ui's YAML text surgery

Create `tests/test_tag_ui_yaml.py`. Use `monkeypatch.setattr` on
`tools.tag_ui.CONFIG_PATH` pointing at a `tmp_path / "comic.yaml"` seeded
with a realistic miniature config (copy the real file's shape):

```yaml
# comment that must survive
comic:
  title: "T"

characters:
  - name: "nova"
    description: "d1"
    ref_image: "nova.png"
    voice_id: "V1"
    speaking_style: "neutral"
  - name: "Thanos"
    description: "d2"
    voice_id: "V2"
    speaking_style: "neutral"

models:
  vision_model: "m"
```

Cases:

- `slugify`: `"Nova (no helmet)!"` → `"nova_no_helmet"`; `"日本語"` →
  `"unnamed"`.
- `set_ref_image("nova", "new.png")` returns True, rewrites only nova's
  `ref_image` line, leaves the `# comment` and Thanos untouched, and the
  file still parses with `yaml.safe_load` to the same structure plus the
  change.
- `set_ref_image("Thanos", "t.png")` on an entry that has NO `ref_image`
  line inserts one inside Thanos's block (verify via `yaml.safe_load`).
- `set_ref_image("nobody", "x.png")` returns False and the file is
  byte-identical.
- `append_character("Gamora", "gamora.png")` inserts a new entry that
  `yaml.safe_load` sees as the LAST element of `characters` (and `models`
  is still a top-level key, not swallowed into the list).

**Verify**: `.venv/bin/python -m pytest tests/test_tag_ui_yaml.py -q` →
all pass.

### Step 7: Torch-guarded test for the bubble mask refiner

Create `tests/test_stage3_mask.py` starting with:

```python
import pytest

torch = pytest.importorskip("torch")
from pipeline.stage3_cleanup import refine_bubble_mask
```

One test: a 200×200 gray (value 128) uint8 image with a white (255) filled
circle at a known location; call `refine_bubble_mask(img, bbox)` with a
bbox loosely around the circle; assert the returned mask is nonzero at the
circle center and zero far outside the padded bbox. On this machine the
whole file will report as SKIPPED — that is the expected result here; it
will run on the GPU box.

**Verify**: `.venv/bin/python -m pytest tests/test_stage3_mask.py -q` →
`1 skipped`, exit 0.

### Step 8: Full run and commit

```bash
.venv/bin/python -m pytest -q
git add -A && git commit -m "Add pytest baseline: stage1 geometry, JSON parsing, tag_ui YAML surgery"
```

**Verify**: pytest exit 0 with ≥12 passed, 1 skipped; `git status --short`
→ empty.

## Test plan

This plan IS the test plan. No existing tests exist to use as a pattern —
these become the pattern for later plans.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `git log --oneline | wc -l` ≥ 2 and `git status --short` is empty
- [ ] `git ls-files | grep -c '^work/'` → 0 (work/ not tracked)
- [ ] `.venv/bin/python -m pytest -q` exits 0; ≥ 12 passed, 1 skipped
- [ ] No files outside the in-scope list are modified
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Any "Current state" excerpt doesn't match the live file.
- `git init` reveals the directory already IS a git repo (plan is stale).
- A test you wrote fails because it exposes a real bug in the helper under
  test (e.g. `reading_order` misorders the 2×2 grid). Do NOT change
  pipeline code to make tests pass — report the bug; the test suite must
  document current behavior.
- `pip install` cannot reach the network or pytest won't install on
  Python 3.14.

## Maintenance notes

- Later plans (002–005) add tests to this suite; they assume
  `.venv/bin/python -m pytest` is the one-command verification gate.
- The duplicated `_parse_json` is tested in both locations on purpose;
  deduplication was considered and deferred (see plans/README.md).
- When the GPU box pulls this repo, run the full suite there once — the
  torch-guarded stage-3 test only executes in that environment.
