# Plan 005: One character, many appearances — stop splitting a character across roster identities

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: if plan 001 landed, diff
> `pipeline/stage2_understand.py`, `pipeline/stage4_story.py`, and
> `config/comic.yaml` against the commit where the most recent plan
> finished; also compare the "Current state" excerpts against the live
> code. On a mismatch, treat it as a STOP condition. Plans 002/004 edit
> `stage4_story.py` — land them first.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED (touches the stage-2 prompt context; attribution quality
  must not regress — mitigations below avoid any paid re-run)
- **Depends on**: plans/001-verification-baseline.md,
  plans/002-stage4-response-robustness.md,
  plans/004-decouple-creative-loop-from-gpu.md (same-file edits)
- **Category**: bug (data model)
- **Planned at**: no VCS at planning time (see plan 001), 2026-07-17

## Why this matters

The roster conflates "a character" with "an appearance of a character".
The current comic has both `nova` and `Nova_nohelmet` as separate roster
entries with separate `voice_id`s, and the real stage-2 output split one
character's dialogue across them: 51 lines attributed to `Nova_nohelmet`
vs 20 to `nova` (measured in `work/understanding.json`). When TTS lands
(stage 6, README-stated), the protagonist would speak with two different
voices depending on whether he's wearing his helmet. The fix is an
`variant_of` field: variant entries contribute an extra reference image
to the vision prompt but resolve to the canonical character everywhere —
speaker names, characters_present, and eventually voice selection. A
one-off remap of the existing `understanding.json` avoids paying for a
stage-2 re-run.

## Current state

- `config/comic.yaml:18-42` — roster entries (4-space-indented mapping
  under `characters:`). The two entries that motivate this plan:

  ```yaml
    - name: "nova"
      description: "cocky but earnest, wisecracks under pressure"
      ref_image: "nova.png"
      voice_id: "ELEVENLABS_VOICE_ID_HERE"
      speaking_style: "neutral"
  ```

  ```yaml
    - name: "Nova_nohelmet"
      description: "tired, unguarded, more human without the mask"
      ref_image: "nova_nohelmet.png"
      voice_id: "ELEVENLABS_VOICE_ID_HERE"
      speaking_style: "neutral"
  ```

- `pipeline/stage2_understand.py:90-110` — roster prompt blocks: for each
  character with an existing ref image it emits
  `"Reference image of {name} ({description}):"` + the image block;
  the LAST block gets `cache_control` (line 108-110) — that must remain
  the last block after your changes:

  ```python
  if roster_blocks:
      # identical prefix across all requests -> cache the roster
      roster_blocks[-1]["cache_control"] = {"type": "ephemeral"}
  ```

- `pipeline/stage2_understand.py:168-169` — the canonical-name map used
  to fix casing drift in speaker names:

  ```python
  canon = {ch["name"].lower(): ch["name"] for ch in roster}
  canon.update({"narrator": "narrator", "unknown": "unknown"})
  ```

- `pipeline/stage4_story.py:88-91` (post-plan-004 the load line differs,
  but this loop is unchanged) — builds the panel lines sent to the story
  model, passing `d.get("speaker", "unknown")` through untouched.

- `work/understanding.json` — existing paid output; 51 dialogue lines
  have `"speaker": "Nova_nohelmet"`. Also `characters_present` arrays
  contain `"Nova_nohelmet"`.

- `tools/tag_ui.py` — reads roster entries by `name`/`ref_image` only
  (`load_state`, line 46) and appends new entries without `variant_of`
  (`append_character`, line 87). It tolerates unknown keys in entries; no
  change needed there in this plan.

- Repo conventions: plain functions, `_private` helpers,
  `raise SystemExit` for fatal user errors, tests in `tests/` as plain
  pytest functions (see `tests/test_json_parsing.py` from plan 001).

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Tests | `.venv/bin/python -m pytest -q` | exit 0, all pass |
| Remap dry run | see Step 4 | prints per-name remap counts |

Never run stages 2 or 4 for verification (API cost).

## Scope

**In scope** (the only files you should create/modify):
- `pipeline/roster.py` (create — canonicalization helpers)
- `pipeline/stage2_understand.py`
- `pipeline/stage4_story.py` (speaker canonicalization on load only)
- `config/comic.yaml` (mark `Nova_nohelmet` as a variant)
- `tools/remap_speakers.py` (create — one-off migration script)
- `tests/test_roster.py` (create)
- `README.md` (document `variant_of`)

**Out of scope** (do NOT touch):
- `work/understanding.json` by hand — only via the Step 4 script, run by
  the operator (see Step 4's note).
- `tools/tag_ui.py` — variant-aware tagging UI is deferred (see
  Maintenance notes).
- Voice/TTS logic — stage 6 doesn't exist yet; this plan only guarantees
  it will see one name per character.
- `SYSTEM_PROMPT` texts in stage 2 and stage 4.

## Git workflow

- Default branch, commit per step, short imperative messages
  (e.g. "Add variant_of roster field and canonical speaker mapping").

## Steps

### Step 1: Canonicalization helpers

Create `pipeline/roster.py`:

```python
"""Roster helpers: appearance variants resolve to canonical characters.

A roster entry with `variant_of: "<canonical name>"` is not a character;
it contributes an extra reference image for the vision pass and maps any
attribution of its name back to the canonical entry.
"""


def canonical_names(characters):
    """Case-insensitive map of every roster/variant name (plus narrator/
    unknown) to its canonical display name."""
    canon = {}
    for ch in characters:
        target = ch.get("variant_of") or ch["name"]
        canon[ch["name"].lower()] = target
    # variants may point at names in any casing; normalize targets too
    for k, v in list(canon.items()):
        canon[k] = canon.get(v.lower(), v)
    canon["narrator"] = "narrator"
    canon["unknown"] = "unknown"
    return canon


def canonicalize(name, canon):
    """Map a speaker/character name through canon; unknown names pass
    through unchanged (they're surfaced at the review gate)."""
    return canon.get(str(name).lower(), name)
```

Note the two-pass loop: `variant_of: "nova"` must resolve through nova's
own canonical casing.

**Verify**: `.venv/bin/python -c "from pipeline.roster import canonical_names; print(canonical_names([{'name':'nova'},{'name':'Nova_nohelmet','variant_of':'nova'}]))"`
→ maps both `nova` and `nova_nohelmet` keys to `nova`.

### Step 2: Stage 2 — variant ref images under the canonical name; canonical output names

In `pipeline/stage2_understand.py`:

1. Import the helpers: `from pipeline.roster import canonical_names, canonicalize`.
2. Roster block construction (lines 90–107): for a variant entry
   (`ch.get("variant_of")`), label its image with the CANONICAL name so
   the model never learns the variant name:
   `f"Reference image of {canonical} ({ch['description']}):"` where
   `canonical = canonicalize(ch['variant_of'], canon)`. Non-variant
   entries unchanged. Build `canon = canonical_names(roster)` once,
   before the loop. Keep the `cache_control` assignment on the last
   block exactly as-is.
3. Replace the inline canon construction (lines 168–169) with the
   prebuilt `canon`, and replace the casing-fix branch (lines 179–181)
   with `line["speaker"] = canonicalize(s, canon)`.
4. Also canonicalize `characters_present`:
   `parsed["characters_present"] = [canonicalize(n, canon) for n in parsed.get("characters_present", [])]`
   (place it next to the focus_box conversion around line 184).

**Verify**: `.venv/bin/python -m pytest -q` → exit 0;
`.venv/bin/python -c "import pipeline.stage2_understand"` → exit 0.

### Step 3: Stage 4 — canonicalize speakers on load (defense in depth)

In `pipeline/stage4_story.py`, when building `panel_lines` (the loop at
~lines 86–94), map each dialogue speaker through the roster:
build `canon = canonical_names(config.get("characters", []))` once and
use `canonicalize(d.get("speaker", "unknown"), canon)`. Also exclude
variant entries from the character list sent in `user_msg`
(line ~101): skip entries with `variant_of` so the story model never
sees variant names.

This makes stage 4 correct even against a pre-migration
`understanding.json`.

**Verify**: `.venv/bin/python -c "import pipeline.stage4_story"` → exit 0;
full pytest green.

### Step 4: Config migration + one-off remap script

1. Edit `config/comic.yaml`: in the `Nova_nohelmet` entry, add
   `variant_of: "nova"` (keep its `ref_image` and `description`; its
   `voice_id` line may remain — it is now ignored — or be deleted;
   prefer deleting to avoid confusion).
2. Create `tools/remap_speakers.py` — a small script (argparse,
   `--workdir work` default, `--dry-run` flag) that loads the config and
   `workdir/understanding.json`, maps every `dialogue[].speaker` and
   `characters_present[]` entry through `canonical_names`, prints a
   count per remapped name (e.g. `Nova_nohelmet -> nova: 51 lines`), and
   writes the file back only without `--dry-run`. Reuse
   `pipeline.roster`; add the repo root to `sys.path` the same way
   `tests/conftest.py` does, or run it as
   `.venv/bin/python -m tools.remap_speakers`.

**Verify**:
`.venv/bin/python tools/remap_speakers.py --dry-run` →
prints `Nova_nohelmet -> nova: 51` (and a characters_present count),
exits 0, and `grep -c '"Nova_nohelmet"' work/understanding.json` is
unchanged. Then run it for real:
`.venv/bin/python tools/remap_speakers.py` →
`grep -c '"speaker": "Nova_nohelmet"' work/understanding.json` → 0.
(Running the real remap is in-scope here: it edits regenerable pipeline
output, is what the script exists for, and git — from plan 001 — can
revert it. Commit `work/` is NOT tracked, so note the remap in your
report.)

### Step 5: Tests

Create `tests/test_roster.py` (pattern: plan 001 tests):

- `canonical_names`: roster with a variant → both lowercase keys map to
  the canonical name; narrator/unknown present; variant pointing at a
  differently-cased target (`variant_of: "NOVA"` with entry name
  `nova`) still resolves to `nova`.
- `canonicalize`: known name any casing → canonical; unknown name →
  passed through unchanged; non-string (None) → returned as-is without
  raising (the `str(name)` guard).
- Chain safety: a variant whose target doesn't exist in the roster maps
  to the target string as written (no KeyError).

**Verify**: `.venv/bin/python -m pytest tests/test_roster.py -q` → all
pass; full suite exit 0.

### Step 6: README

Document `variant_of` in the "Prepare your comic" section: one character
= one entry with a voice; extra appearances (helmet off, disguise, young
version) get their own entry with `variant_of: "<name>"` and a ref image,
tagged the same way in the UI.

**Verify**: `grep -n "variant_of" README.md` → ≥ 1 match.

## Test plan

Covered in Step 5. No stage-2/4 end-to-end tests (API); the prompt-block
change is verified by reading + the import check, and the speaker mapping
by unit tests plus the real remap counts in Step 4.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `grep -n "variant_of" config/comic.yaml` → 1 match (Nova_nohelmet)
- [ ] `grep -c '"speaker": "Nova_nohelmet"' work/understanding.json` → 0
- [ ] `grep -n "canonical_names" pipeline/stage2_understand.py pipeline/stage4_story.py` → ≥ 1 match in each
- [ ] `.venv/bin/python -m pytest -q` exits 0 including the new tests
- [ ] `grep -n "variant_of" README.md` → ≥ 1 match
- [ ] No files outside the in-scope list are modified (`git status`; work/ is untracked — report the remap separately)
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Plans 002/004 haven't landed (stage4_story.py doesn't contain
  `_load_panels` / `story_raw`) — same-file ordering.
- The `cache_control` block in stage 2 isn't where "Current state" says —
  your change must not move which block carries it.
- The dry-run remap count is not 51 for Nova_nohelmet speaker lines
  (the data changed since planning — re-verify with
  `grep -c '"speaker": "Nova_nohelmet"' work/understanding.json` and
  report the actual number before proceeding).
- You find yourself editing tag_ui.py or inventing TTS/voice logic.

## Maintenance notes

- Stage 6 (TTS) must select voices by CANONICAL name only and should
  assert no variant names appear in manifests.
- `tools/tag_ui.py` currently appends new names as full character
  entries; a follow-up could add a "variant of…" choice in the naming
  popover (deferred — UI change, and manual `variant_of` edits are
  cheap).
- If the user re-runs stage 2 from scratch later, attribution may improve
  further (the model now sees both Nova images under one name — 2 ref
  images beats 1); the 81 `unknown` lines in the current data are
  unaffected by this plan and remain review-gate work.
- Reviewer should scrutinize: the two-pass resolution in
  `canonical_names` and that variant descriptions still reach the prompt
  (they carry useful "no helmet" context for face matching).
