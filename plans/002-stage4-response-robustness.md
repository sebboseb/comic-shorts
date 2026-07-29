# Plan 002: Never lose a paid stage-4 (Opus) response

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: if plan 001 landed, run
> `git diff --stat -- pipeline/stage4_story.py` against the commit where
> 001 finished; also compare the "Current state" excerpts against the live
> code. On a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW (error-path hardening; happy path unchanged)
- **Depends on**: plans/001-verification-baseline.md (for pytest + git)
- **Category**: bug
- **Planned at**: no VCS at planning time (see plan 001), 2026-07-17

## Why this matters

Stage 4 is the one expensive, quality-critical LLM call in the pipeline
(Claude Opus over all ~121 panels; the config comment calls it
"quality-critical"). Today the response is parsed straight from
`resp.content[0].text` with no persistence: if the model's JSON is
truncated (hit `max_tokens=16000`), wrapped in unexpected content blocks,
or malformed in any way, the run dies with a traceback and the paid
response is gone — the user must pay for the call again. Saving the raw
text before parsing and failing with an actionable message makes every
failure mode recoverable for free.

## Current state

- `pipeline/stage4_story.py` — the whole stage; ~135 lines. The fragile
  section is `run()`, lines 105–112:

  ```python
  print("compiling story...")
  resp = client.messages.create(
      model=model,
      max_tokens=16000,
      system=SYSTEM_PROMPT,
      messages=[{"role": "user", "content": user_msg}],
  )
  data = _parse_json(resp.content[0].text)
  ```

  Failure modes, concretely:
  1. `resp.stop_reason == "max_tokens"` → truncated JSON →
     `json.JSONDecodeError` traceback, response lost.
  2. `resp.content[0]` is not a text block (e.g. model families that emit
     a thinking block first) → `AttributeError`, response lost.
  3. Any other malformed JSON → traceback, response lost.

- `_parse_json` at `pipeline/stage4_story.py:67-73` (strips markdown
  fences, then `json.loads`) — keep it as-is; tests for it exist from
  plan 001 in `tests/test_json_parsing.py`.

- Error-handling convention in this repo: fatal user-facing errors are
  `raise SystemExit("message")` — see `pipeline/stage4_story.py:78`
  (`raise SystemExit("Set ANTHROPIC_API_KEY")`) and
  `pipeline/stage3_cleanup.py:93`. Match it.

- Stage 4 reads its input from `workdir / "clean.json"` (line 83) and
  writes manifests to `workdir / "manifests"`. `workdir` is a `Path`
  passed in by `run.py` (default `work`).

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Tests | `.venv/bin/python -m pytest -q` | exit 0, all pass |
| Import sanity | `.venv/bin/python -c "import pipeline.stage4_story"` | exit 0 |

Do NOT run `python run.py --stage 4` to verify — it costs real money and
needs an API key. Verification is tests + import only.

## Scope

**In scope** (the only files you should modify):
- `pipeline/stage4_story.py`
- `tests/test_stage4_response.py` (create)

**Out of scope** (do NOT touch, even though they look related):
- `pipeline/stage2_understand.py` — has its own copy of `_parse_json`;
  deduplication is explicitly deferred (see plans/README.md).
- `SYSTEM_PROMPT` in stage4_story.py — prompt changes alter output
  quality; not this plan's business.
- `run.py`, `pipeline/review.py`.

## Git workflow

- Work on the default branch (solo repo, no remote), commit per step.
- Message style: short imperative subject, e.g.
  "Persist raw stage-4 response before parsing".

## Steps

### Step 1: Extract response text via a testable helper

In `pipeline/stage4_story.py`, add a module-level helper next to
`_parse_json`:

```python
def _response_text(resp):
    """All text blocks joined; tolerates non-text blocks in content."""
    return "".join(b.text for b in resp.content if b.type == "text")
```

**Verify**: `.venv/bin/python -c "import pipeline.stage4_story"` → exit 0.

### Step 2: Save raw text before parsing, fail with actionable errors

Replace the single line `data = _parse_json(resp.content[0].text)` with
logic that does, in order:

1. `text = _response_text(resp)`
2. `raw_path = workdir / "story_raw.txt"`; write `text` to it
   unconditionally, and print that it was saved.
3. If `resp.stop_reason == "max_tokens"`:
   `raise SystemExit(...)` with a message stating the response was
   truncated at the token limit, the raw text is at `work/story_raw.txt`,
   and the fix is to raise `max_tokens` or ask for fewer/shorter shorts —
   do not silently parse a truncated response.
4. Wrap `data = _parse_json(text)` in `try/except json.JSONDecodeError`
   and on failure `raise SystemExit(...)` naming the raw file path and the
   decode error, so the user can repair the JSON by hand and knows nothing
   was lost. (Add `json` usage as needed — it is already imported.)
5. After parsing, if `"shorts" not in data or not data["shorts"]`:
   `raise SystemExit(...)` naming the raw file (defends against a valid-
   JSON-wrong-shape response before the `data["shorts"]` access at the
   loop on the following lines).

Keep everything after this point (manifest writing loop) unchanged.

**Verify**: `.venv/bin/python -m pytest -q` → exit 0 (existing suite still
green; new tests come next).

### Step 3: Tests

Create `tests/test_stage4_response.py` (follow the import pattern from
`tests/test_json_parsing.py`, plan 001). Build tiny stand-in objects — no
network, no anthropic client:

```python
class _Block:
    def __init__(self, type, text=""):
        self.type = type
        self.text = text

class _Resp:
    def __init__(self, blocks, stop_reason="end_turn"):
        self.content = blocks
        self.stop_reason = stop_reason
```

Cases for `_response_text`:
- single text block → its text
- a non-text block (`_Block("thinking")`) before a text block → only the
  text block's text
- multiple text blocks → concatenated

Do not test `run()` end-to-end (it would need an API client); the helper
plus the SystemExit branches being simple straight-line code is the agreed
depth. If you refactor step 2's logic into a second pure helper
(e.g. `_parse_story(text, stop_reason, raw_path)`), you may — then test
its three SystemExit branches with `pytest.raises(SystemExit)` and assert
the message mentions `story_raw.txt`.

**Verify**: `.venv/bin/python -m pytest tests/test_stage4_response.py -q`
→ all pass; `.venv/bin/python -m pytest -q` → exit 0 overall.

## Test plan

Covered in Step 3. Pattern to follow: `tests/test_json_parsing.py` from
plan 001 (plain pytest functions, no classes).

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `grep -n "story_raw" pipeline/stage4_story.py` → at least one match
- [ ] `grep -n "resp.content\[0\]" pipeline/stage4_story.py` → no matches
- [ ] `grep -n "stop_reason" pipeline/stage4_story.py` → at least one match
- [ ] `.venv/bin/python -m pytest -q` exits 0, including new tests
- [ ] No files outside the in-scope list are modified (`git status`)
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- `pipeline/stage4_story.py` no longer matches the "Current state"
  excerpts (someone changed stage 4 since planning).
- Plan 001 has not landed (no `tests/` directory or pytest missing) —
  report; don't inline a partial test setup.
- You find yourself wanting to modify the API call parameters (model,
  max_tokens) or the system prompt — out of scope; report instead.

## Maintenance notes

- `work/story_raw.txt` is overwritten on every stage-4 run; if someone
  later adds multi-run comparison, timestamp the filename then.
- Reviewer should scrutinize: the truncation branch must fire BEFORE the
  parse attempt (a truncated response can still be valid JSON prefix-wise
  in rare cases — the stop_reason check is the authoritative signal).
- Deferred: a `--resume-from-raw` flag to re-parse a hand-repaired
  `story_raw.txt` without a new API call. Cheap to add later; the raw file
  this plan introduces is the prerequisite.
