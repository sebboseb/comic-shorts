# Plan 003: Persist and resume stage-2 batches (no orphaned paid batches)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: if plan 001 landed, run
> `git diff --stat -- pipeline/stage2_understand.py` against the commit
> where 001 finished; also compare the "Current state" excerpts against the
> live code. On a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW (adds persistence around an existing flow; parsing and
  output format unchanged)
- **Depends on**: plans/001-verification-baseline.md
- **Category**: bug
- **Planned at**: no VCS at planning time (see plan 001), 2026-07-17

## Why this matters

Stage 2 submits one Anthropic Batches-API request per panel (121 on the
current comic) and then polls in a loop, sleeping 20s between checks. The
batch id exists only in a `print()`. If the process dies mid-poll —
Ctrl-C, laptop sleep, network blip after the retries are exhausted — the
paid batch is orphaned: results are retrievable from the API for days, but
the pipeline has no record of the id, and re-running stage 2 submits and
pays for a brand-new batch. Persisting the batch id to the work dir and
resuming it on the next run makes the stage crash-safe for free.

## Current state

- `pipeline/stage2_understand.py` — the whole stage (~194 lines).
  The submit-and-poll section, lines 136–146:

  ```python
  batch = client.messages.batches.create(requests=requests)
  print(f"batch {batch.id} submitted ({len(requests)} panels), polling...")

  while True:
      batch = client.messages.batches.retrieve(batch.id)
      if batch.processing_status == "ended":
          break
      c = batch.request_counts
      print(f"  processing... {c.succeeded} done, {c.errored} errored, "
            f"{c.processing} in flight")
      time.sleep(20)
  ```

- Results are consumed by custom_id (`custom_id=panel["id"]`, line 127;
  results loop at lines 148–166). Panels missing from the results already
  get a placeholder entry (lines 174–177: `"parse_error": "missing from
  batch results"`), so a resumed batch whose request set no longer matches
  a re-generated `panels.json` degrades gracefully rather than crashing.
- The stage reads `workdir / "panels.json"` and writes
  `workdir / "understanding.json"` at the end (line 192). `workdir` is a
  `Path` (default `work`).
- Anthropic SDK facts (verify against the installed SDK, do not trust
  blindly): `client.messages.batches.retrieve(id)` returns a batch whose
  `processing_status` is `"in_progress"`, `"canceling"`, or `"ended"`;
  retrieving an unknown/expired id raises `anthropic.NotFoundError`.
- Error convention: fatal errors are `raise SystemExit("msg")` — see
  line 83. Match it.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Tests | `.venv/bin/python -m pytest -q` | exit 0, all pass |
| Import sanity | `.venv/bin/python -c "import pipeline.stage2_understand"` | exit 0 |

Do NOT run `python run.py --stage 2` to verify — it costs money and needs
an API key.

## Scope

**In scope** (the only files you should modify):
- `pipeline/stage2_understand.py`
- `tests/test_stage2_batch_state.py` (create)
- `README.md` — one short paragraph in "What to check at each gate" or the
  Run section noting that an interrupted stage 2 resumes automatically.

**Out of scope** (do NOT touch):
- The prompt, model choice, image sizing, roster block construction, or
  result parsing in stage 2 — resume logic only.
- `run.py` — no new CLI flags; resume must be automatic and transparent.
- `pipeline/stage4_story.py` (batch-like concerns there are plan 002).

## Git workflow

- Default branch, commit per step, short imperative messages
  (e.g. "Persist stage-2 batch id for crash-safe resume").

## Steps

### Step 1: Extract submit-or-resume into a helper

In `pipeline/stage2_understand.py`, add a module-level constant and
helper:

```python
BATCH_STATE_FILE = "batch_id.txt"


def _get_or_create_batch(client, requests, workdir: Path):
    """Resume a previously submitted batch if its id is on disk and still
    retrievable; otherwise submit a new one and persist its id."""
    state = workdir / BATCH_STATE_FILE
    if state.exists():
        batch_id = state.read_text().strip()
        try:
            batch = client.messages.batches.retrieve(batch_id)
            print(f"resuming batch {batch_id} "
                  f"({batch.processing_status})")
            return batch
        except anthropic.NotFoundError:
            print(f"stale batch id {batch_id} (not found on API), "
                  "submitting fresh")
    batch = client.messages.batches.create(requests=requests)
    state.write_text(batch.id)
    print(f"batch {batch.id} submitted ({len(requests)} panels), polling...")
    return batch
```

Ordering matters: `state.write_text(batch.id)` must happen immediately
after `create()` returns, before any polling.

**Verify**: `.venv/bin/python -c "import pipeline.stage2_understand"` →
exit 0.

### Step 2: Use the helper and clear state on success

In `run()`:

1. Replace lines 136–137 (`batch = client.messages.batches.create(...)`
   and the print) with `batch = _get_or_create_batch(client, requests, workdir)`.
2. The polling loop stays as-is.
3. After `understanding.json` is successfully written (currently line
   192), delete the state file:
   `(workdir / BATCH_STATE_FILE).unlink(missing_ok=True)`.

Net behavior: a fresh run submits and records; a crashed run re-attaches
to the in-flight or ended batch instead of paying again; a completed run
leaves no state behind, so the next invocation is a fresh submit.

**Verify**: `.venv/bin/python -m pytest -q` → exit 0 (suite from plan 001
still green).

### Step 3: Tests

Create `tests/test_stage2_batch_state.py` (pattern:
`tests/test_json_parsing.py` from plan 001). Use `tmp_path` as workdir and
a fake client — no network:

```python
import anthropic


class _FakeBatches:
    def __init__(self, known=None, fail_retrieve=False):
        self.known = known or {}
        self.created = []
        self.fail_retrieve = fail_retrieve

    def retrieve(self, batch_id):
        if self.fail_retrieve or batch_id not in self.known:
            raise anthropic.NotFoundError(...)   # construct minimally; see note
        return self.known[batch_id]

    def create(self, requests):
        b = type("B", (), {"id": "batch_new", "processing_status": "in_progress"})()
        self.created.append(requests)
        return b
```

Note: `anthropic.NotFoundError` may require constructor args
(`message`, `response`, `body`); if constructing it is awkward, raise it
via `NotFoundError.__new__(NotFoundError)` or build the minimal httpx
response it wants — whichever is least code. The production code only
needs `except anthropic.NotFoundError` to catch it.

Cases:

1. No state file → `create` called once, `batch_id.txt` now contains
   `batch_new`.
2. State file with a known id → `retrieve` returns that batch, `create`
   never called, state file untouched.
3. State file with an unknown id (retrieve raises NotFoundError) →
   falls through to `create`, state file overwritten with the new id.

**Verify**:
`.venv/bin/python -m pytest tests/test_stage2_batch_state.py -q` → 3
passed; `.venv/bin/python -m pytest -q` → exit 0 overall.

### Step 4: README note

Add one sentence to `README.md` under the stage-2 bullet in "Run" or the
gates section: an interrupted stage-2 run can simply be re-run — it
re-attaches to the submitted batch via `work/batch_id.txt` instead of
paying for a new one; delete that file to force a fresh submit.

**Verify**: `grep -n "batch_id.txt" README.md` → one match.

## Test plan

Covered in Step 3. All tests offline; the fake client is the only test
double — do not mock `time.sleep` or the polling loop (untested by
agreement; it's unchanged code).

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `grep -n "BATCH_STATE_FILE" pipeline/stage2_understand.py` → ≥ 3 matches (constant, helper, cleanup)
- [ ] `grep -n "unlink" pipeline/stage2_understand.py` → 1 match, after the understanding.json write
- [ ] `.venv/bin/python -m pytest -q` exits 0 including 3 new tests
- [ ] `grep -n "batch_id.txt" README.md` → 1 match
- [ ] No files outside the in-scope list are modified (`git status`)
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- The submit/poll section of `stage2_understand.py` doesn't match the
  "Current state" excerpt.
- The installed anthropic SDK has no `NotFoundError` or its
  `messages.batches` surface differs from what "Current state" describes
  (check with `.venv/bin/python -c "import anthropic; print(anthropic.NotFoundError)"`).
- You find yourself adding CLI flags or touching `run.py` — out of scope.

## Maintenance notes

- If `panels.json` is regenerated between submit and resume, resumed
  results are matched by custom_id and any mismatch degrades to the
  existing `"missing from batch results"` placeholder — acceptable, but a
  reviewer should know that a resumed batch reflects the panel set at
  submit time.
- If a future change adds parallel/multi-comic runs against the same
  workdir, one `batch_id.txt` per workdir is still correct (workdir is
  per-comic by design — see `run.py --workdir`).
- Deferred: honoring `batch.request_counts.errored` with an automatic
  partial resubmit. Today errored requests get placeholder entries fixed
  at the review gate, which is the documented workflow.
