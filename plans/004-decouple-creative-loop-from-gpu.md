# Plan 004: Let stages 4–5 run before the GPU stage, and surface all known problems at the review gate

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: if plan 001 landed, run
> `git diff --stat -- pipeline/stage4_story.py pipeline/review.py` against
> the commit where the most recent plan finished; also compare the
> "Current state" excerpts against the live code. On a mismatch, treat it
> as a STOP condition. Note plan 002 also edits `stage4_story.py` — apply
> this plan AFTER 002 and expect 002's changes (raw-response persistence)
> to be present around the API call; they do not overlap with the lines
> this plan touches.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED (changes which input files two stages read; mitigated by
  strict fallback ordering + tests)
- **Depends on**: plans/001-verification-baseline.md,
  plans/002-stage4-response-robustness.md (same-file edits; land 002 first)
- **Category**: tech-debt / dx
- **Planned at**: no VCS at planning time (see plan 001), 2026-07-17

## Why this matters

Stage 4 (story compile, API, cheap) and stage 5 (review page) both hard-
require `work/clean.json`, which only exists after stage 3 — the local
GPU inpaint/upscale stage that needs a machine with CUDA/MPS and
downloaded weights. But stage 4 only uses fields (`id`, `scene`,
`dialogue`, `sfx_text`) that already exist in `work/understanding.json`
after stage 2. The coupling forces the user to run the GPU stage before
they can iterate on the creative output at all. (Concretely: this repo
currently has `understanding.json` for 121 panels but no GPU run yet — the
user is blocked from story iteration today for no reason.)

Separately, the review gate — whose entire job is "catch problems before
stages that cost money" — never shows two classes of problems the pipeline
already knows about: panels whose stage-2 response failed to parse
(`parse_error` fields in understanding.json) and pages stage 1 flagged as
unreliable (`flagged_pages` in panels.json). This plan makes the review
page the single place where every known problem is visible, and fixes its
handful of unescaped HTML interpolations while touching those lines.

## Current state

- `pipeline/stage4_story.py:83` —

  ```python
  panels = json.loads((workdir / "clean.json").read_text())
  ```

  Everything stage 4 uses from `panels`: `p["id"]`, `p.get("scene")`,
  `p.get("dialogue")`, `p.get("sfx_text")` (lines 86–94, 117). All of
  these exist identically in `understanding.json` — `clean.json` entries
  are understanding entries plus `clean_file`/`clean_size` (see
  `pipeline/stage3_cleanup.py:123-124`).

- `pipeline/review.py:41-42` —

  ```python
  clean = {p["id"]: p for p in
           json.loads((workdir / "clean.json").read_text())}
  ```

  and line 61:

  ```python
  img_src = f"../{panel['clean_file']}" if panel else ""
  ```

  Raw panel images exist at `work/panels/<id>.png` and each understanding
  entry carries `"file": "panels/<id>.png"`, so `f"../{panel['file']}"` is
  the correct fallback image path (the review page lives at
  `work/review/index.html`; `../` resolves to `work/`).

- Unescaped interpolations in `pipeline/review.py` (all other fields go
  through `html.escape` already — match that):
  - line 57: `music: {short.get('music_mood', '')}`
  - line 72: `<img src='{img_src}' ...>`
  - lines 75–76: `{shot.get('motion', '')}` and `{shot.get('sfx', 'none')}`

- Stage-2 parse failures are recorded per panel as a `"parse_error"` key
  (see `pipeline/stage2_understand.py:154-156, 164-166, 174-177`). The
  current `work/understanding.json` has zero of them, but batch errors
  and JSON failures produce them.

- Stage-1 flags live in `work/panels.json` under `"flagged_pages"`
  (list of `{"page", "file", "reason"}` — see
  `pipeline/stage1_panels.py:164`).

- README run block (`README.md:44-50`) documents the stage order 1→5;
  step 6 below updates it.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Tests | `.venv/bin/python -m pytest -q` | exit 0, all pass |
| Real run of stage 5 | `.venv/bin/python run.py --stage 5` | see Step 7 — only valid AFTER a real stage-4 output exists; skip if `work/manifests/` is absent |

Stage 4 itself is never run for verification (API cost). Stage 5 is free
and local — run it only if manifests already exist.

## Scope

**In scope** (the only files you should modify):
- `pipeline/stage4_story.py` (input-loading only)
- `pipeline/review.py`
- `tests/test_review_page.py` (create)
- `README.md` (run-order note)

**Out of scope** (do NOT touch):
- `pipeline/stage3_cleanup.py` — its outputs stay the preferred input;
  nothing about stage 3 changes.
- `SYSTEM_PROMPT` in stage4_story.py, the API call, and plan 002's
  raw-response code around it.
- `run.py` — stage numbering and CLI stay as-is.

## Git workflow

- Default branch, commit per step, short imperative messages
  (e.g. "Fall back to understanding.json when clean.json is absent").

## Steps

### Step 1: Shared input-fallback helper in stage 4

In `pipeline/stage4_story.py`, replace line 83 with:

```python
panels = _load_panels(workdir)
```

and add the helper (module level, near `_parse_json`):

```python
def _load_panels(workdir: Path):
    """Prefer stage-3 output; fall back to stage-2 output so story
    compilation doesn't require the GPU stage."""
    for name in ("clean.json", "understanding.json"):
        path = workdir / name
        if path.exists():
            print(f"panels from {name}")
            return json.loads(path.read_text())
    raise SystemExit("Run stage 2 first (no understanding.json)")
```

**Verify**: `.venv/bin/python -c "import pipeline.stage4_story"` → exit 0.

### Step 2: Same fallback in review.py, including image paths

In `pipeline/review.py` `run()`:

1. Replace the `clean = {...}` load (lines 41–42) with the same
   two-file fallback (either import `_load_panels` from
   `pipeline.stage4_story` or duplicate the 7-line helper — prefer the
   import; it's not private-by-contract, and review already depends on
   stage 4's outputs conceptually).
2. Replace line 61's image source with a fallback:

   ```python
   img_src = f"../{panel.get('clean_file', panel['file'])}" if panel else ""
   ```

3. Update the "panel not found" warn text (line 63) from
   `panel not found in clean.json` to `panel not found in pipeline output`.

**Verify**: `.venv/bin/python -c "import pipeline.review"` → exit 0.

### Step 3: Surface stage-1 flags and stage-2 parse errors at the top of the review page

In `pipeline/review.py` `run()`, after loading the panel data and before
the manifest loop, build a problems block:

- If `workdir / "panels.json"` exists and its `"flagged_pages"` list is
  non-empty: one `<div class='warn'>` line per flag —
  `stage 1 flagged {file}: {reason}` (both `html.escape`d).
- For every loaded panel entry containing a `"parse_error"` key: one warn
  line — `stage 2 parse error on {id} — its scene/dialogue are empty;
  fix understanding.json or re-run` (escape the id; do NOT print the
  parse_error payload itself, it can be an entire raw LLM response).
- If neither produced output, emit nothing (no empty section header).
- Insert these lines into `parts` right after the existing intro
  `<p class='meta'>` block, under a heading like
  `<h2>Pipeline warnings</h2>`.

Also add a banner note when the page was built from `understanding.json`
(no `clean.json`): a single meta line — `showing RAW panels (stage 3 has
not run); bubbles are still in the art`.

**Verify**: `.venv/bin/python -m pytest -q` → exit 0 (new tests in Step 5
will pin this behavior; suite must stay green now).

### Step 4: Escape the remaining interpolations

Still in `pipeline/review.py`:

- line 57 area: wrap `short.get('music_mood', '')` in `html.escape(...)`.
- lines 75–76: wrap `shot.get('motion', '')` and `shot.get('sfx', 'none')`
  in `html.escape(...)`.
- line 72: switch the img tag to double-quoted attribute with
  `html.escape(img_src, quote=True)`:
  `f'<img src="{html.escape(img_src, quote=True)}" loading="lazy">'`

These values come from LLM-generated manifests; the review page should
render whatever they contain as text, never as markup.

**Verify**: `grep -n "html.escape" pipeline/review.py` → ≥ 8 matches.

### Step 5: Tests

Create `tests/test_review_page.py` (pattern: plan 001's tests; import
`pipeline.review` and call `run(config, workdir)` with `tmp_path` as
workdir). Build a minimal synthetic workdir:

- `tmp_path/"understanding.json"`: two panel entries —
  `{"id": "p01_01", "file": "panels/p01_01.png", "scene": "s", "dialogue": []}`
  and one with `"parse_error": "boom"`.
- `tmp_path/"panels.json"`:
  `{"panels": [], "flagged_pages": [{"page": 1, "file": "x.jpg", "reason": "low coverage"}]}`
- `tmp_path/"manifests/ep01.json"`: one short with one shot referencing
  `p01_01`, including a hostile value to pin escaping:
  `"motion": "<script>alert(1)</script>"`.
- NO `clean.json` — this is the fallback path under test.

Assert on the generated `tmp_path/"review/index.html"` text:

1. it exists (fallback worked without clean.json);
2. contains `panels/p01_01.png` (raw-image fallback path);
3. contains the flagged-page reason and the parse-error panel id;
4. does NOT contain `<script>alert` (escaped), but DOES contain
   `&lt;script&gt;`;
5. contains the "RAW panels" banner.

Add one more test where `clean.json` IS present (same entries plus
`"clean_file": "clean/p01_01.png"`) and assert the img src uses
`clean/p01_01.png` and the RAW banner is absent.

Also test `pipeline.stage4_story._load_panels` directly: prefers
clean.json when both exist; falls back to understanding.json; raises
`SystemExit` when neither exists.

**Verify**: `.venv/bin/python -m pytest tests/test_review_page.py -q` →
all pass; full suite exit 0.

### Step 6: README

In `README.md`'s Run section, note that stages 4 and 5 can run before
stage 3 (review then shows raw panels with bubbles still in the art), so
story iteration doesn't wait for the GPU box.

**Verify**: `grep -ni "before stage 3\|without the gpu\|raw panels" README.md`
→ ≥ 1 match.

### Step 7: Real-artifact smoke test (conditional)

Only if `work/manifests/` exists with ep*.json files:
`.venv/bin/python run.py --stage 5` → prints `Review page -> work/review/index.html`,
exit 0. If `work/manifests/` does not exist, skip this step and note that
in your report — do NOT run stage 4 to create manifests.

## Test plan

Covered in Step 5 — the synthetic-workdir test is the first integration-
level test in the repo; keep it dependency-free (json + tmp_path only,
no images needed since review never opens image files).

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `grep -n "clean.json" pipeline/stage4_story.py` → only inside `_load_panels`'s fallback tuple
- [ ] `grep -c "html.escape" pipeline/review.py` → ≥ 8
- [ ] `.venv/bin/python -m pytest -q` exits 0 including the new tests
- [ ] README updated (grep from Step 6 passes)
- [ ] No files outside the in-scope list are modified (`git status`)
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Plan 002 has not landed (stage4_story.py still parses
  `resp.content[0].text` directly) — ordering matters for same-file edits.
- `review.py` or `stage4_story.py` don't match the "Current state"
  excerpts.
- The understanding.json entries turn out NOT to carry a `file` key
  (check `work/understanding.json` — they do at planning time), which
  would break the raw-image fallback.
- You find yourself modifying stage 3 or run.py.

## Maintenance notes

- Stage 4 compiled from raw understanding data and stage 4 compiled from
  clean data produce identical manifests (it never reads clean_file), so
  no re-run is needed after stage 3 — but the review page should be
  regenerated after stage 3 to eyeball the inpainted art.
- If a future stage 6 (TTS) also wants panel data, reuse `_load_panels`
  — and at that point move it to a small shared module
  (e.g. `pipeline/io.py`) rather than importing across stages twice.
- Reviewer should scrutinize: the fallback ORDER (clean first) and that
  the parse_error payload is never rendered into the page.
