# Local open-weights VLMs vs the Claude Sonnet API for stage 2 (panel understanding)

*Researched 2026-07-12. Question: can any open-weights vision-language model, running on the
24GB-VRAM GPU box or a 36GB M5 Max MacBook, replace the `claude-sonnet-4-6` Batches-API call in
`pipeline/stage2_understand.py` with near-zero quality regression?*

Workload being judged (not generic benchmarks): per panel, one strict-JSON response containing
verbatim speech-bubble transcription (stylized comic lettering), speaker attribution against ~8
reference face crops sent as multi-image input, bubble bounding boxes (normalized 0-1000, later
refined into pixel masks by stage 3's near-white blob detection) plus a focus box, emotion/delivery
labels, and a one-sentence scene description.

---

## TL;DR verdict

**No local model is *verified* good enough to swap in today with near-zero regression — stay on the
API for production, but the gap is narrow enough that a cheap A/B test of Qwen3.6-27B is justified.**

- The smartest local candidate is **Qwen3.6-27B** (April 2026, Apache-2.0, natively multimodal,
  Q4 fits both machines). On official tables it *beats* Claude-tier models on exactly the axes this
  pipeline needs: OCRBench 89.4 vs Claude Sonnet 4.5's 76.6, and RefCOCO grounding 92.5 avg — a
  trained capability Claude's own docs describe as "approximate" on their side.
- But **no primary source measures the two make-or-break fields on comics**: verbatim transcription
  of stylized Western comic lettering, and speaker attribution from reference face crops. The comic
  literature that does exist (CoMix, MangaOCR) shows frontier API models transcribing comic dialogue
  extremely well (GPT-4: 93.1% dialog score) while *every* general VLM — open and API alike —
  collapses on end-to-end text localization in comic pages, and zero-shot speaker ID is unsolved
  across the board.
- The API baseline costs ~$1 per 112-panel run (Batches API, 50% discount). The saving is trivially
  small, so the only reasons to go local are independence/offline/scale — in which case the
  evidence-backed shape is a **hybrid** (see Recommendation).

---

## Comparison table (models that fit)

"OCR" = official DocVQA / OCRBench where published. "Boxes" = is box output a *trained, documented*
capability. JSON reliability is effectively solved for **all** local models via engine-level
constrained decoding (vLLM guided decoding, llama.cpp GBNF/json_schema, Ollama structured outputs,
mlx-vlm json_schema), so the column reflects model-level claims only.

| Model | Size / quant that fits | OCR (official) | Boxes / grounding | Multi-image | JSON | 24GB CUDA | 36GB M5 Max | License |
|---|---|---|---|---|---|---|---|---|
| **Qwen3.6-27B** (Apr 2026) | 27B dense; Q4 ≈ 16GB | OCRBench **89.4**, CC-OCR 81.2 | **Yes** — RefCOCO avg **92.5** | Family-documented (Qwen3-VL lineage); not spelled out on card | Engine-level | vLLM, llama.cpp (text+vision) | mlx-vlm (official), llama.cpp/LM Studio | Apache-2.0 |
| **Qwen3.5-27B** (Feb 2026) | 27B dense; Q4 ≈ 16GB | OCRBench **89.4** (vs Sonnet 4.5 76.6) | **Yes** — RefCOCO avg 90.9 | Same as above | Engine-level | vLLM, llama.cpp, Ollama (`qwen3.5:27b`) | mlx-vlm, Ollama, LM Studio | Apache-2.0 |
| **Qwen3-VL-32B / 30B-A3B / 8B** (Sep-Oct 2025) | Q4 ≈ 19 / 17 / 5.5GB; official GGUFs | 32-language OCR (card claim; charts not machine-readable) | **Yes** — "Precise Object Grounding" cookbook, relative coords, boxes + points | Yes — multi-image inference examples in repo | Engine-level | vLLM ≥0.11 (official path) | MLX + GGUF in LM Studio (2B-32B) | Apache-2.0 |
| **Gemma 4 31B** (Mar-Apr 2026) | 31B dense; official QAT q4_0 GGUF ≈ 15-17GB | OmniDocBench 0.131 edit-dist; OCR/handwriting listed as trained capabilities (no DocVQA/OCRBench published) | **Yes** — native JSON `{"box_2d":[y1,x1,y2,x2]}`, **normalized 0-1000** (pipeline's exact convention) | **Yes** — "multiple images in a single prompt" (official docs) | "Native structured JSON output" (Google blog) + engine-level | llama.cpp day-0, vLLM (w4a16 QAT) | mlx-vlm (official), llama.cpp, LM Studio | Apache-2.0 |
| **Gemma 4 26B-A4B** (MoE, 3.8B active) | Q4 ≈ 13GB | OmniDocBench 0.149 | Same trained detection/pointing as 31B | Yes | Same | Yes, fastest option | Yes, fastest option | Apache-2.0 |
| **MiniCPM-V 4.5** (8.7B) | Official GGUF Q4_K_M 5.0GB + 1.1GB mmproj; F16 fits 24GB | OCRBench **89.0** (> GPT-4o 82.2), DocVQA **94.7**, TextVQA 82.2 | **Yes** — official grounding cookbook (`<ref>/<box>`, 0-1000 coords, JSON multi-object) | Supported, but no official example beyond 2 images — 9-image prompt **unverified** | Engine-level | vLLM ≥0.10.2, llama.cpp ≥b6282, official Ollama | llama.cpp/Ollama/LM Studio | Apache-2.0 (registration requirement dropped) |
| **Mistral Small 3.2 24B** | Q4_K_M ≈ 15GB | **DocVQA 94.86**, ChartQA 87.4 — best published doc-OCR that fits | **No** — no grounding claim anywhere on card/docs | Yes — vLLM guidance shows 10 images/prompt | Engine-level | vLLM ≥0.9.1, Ollama | llama.cpp (3.1 listed; 3.2 same arch — minor caveat), mlx-vlm | Apache-2.0 |
| **InternVL3.5-38B / 30B-A3B** (Aug 2025) | 38B Q4 ≈ 21-22GB (**marginal** on 24GB with vision tower + KV); 30B-A3B ≈ 17GB fits both | Strong per paper; card tables are chart images (numbers unverified here) | **Yes** — RefCOCO avg 89.1 (38B, paper) | Yes — multi-image multi-round examples on card | Engine-level | vLLM, LMDeploy | llama.cpp lists InternVL "2.5 and 3" only — 3.5 support **unverified** | Apache-2.0 |
| **Molmo2-8B** (Ai2, Dec 2025) | Q4 ≈ 5GB | No official DocVQA/OCRBench on card | **Points only** (`<points>` XML, 0-1000) — no boxes; would need point→SAM/blob step | **Yes** — multi-image QA is a headline feature | Engine-level | vLLM, transformers | GGUF quants listed | Apache-2.0 |
| **GLM-4.6V-Flash 9B** (Dec 2025) | Q4 ≈ 6GB | Not verified from primary source in this pass | Grounding claimed for family (not verified in detail) | Not verified | Engine-level | vLLM | Unverified | Free commercial (per announcement) |

### Ruled out on arithmetic (do not revisit)

| Model | Why |
|---|---|
| Llama 4 Scout (109B/17B-active MoE) | All 109B must be resident: 4-bit ≈ 55GB. Meta's own "single H100" claim is an 80GB card. Also Llama-4 license + only "tested up to 5 input images" — the 9-image prompt is out of tested envelope anyway. [Card](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct), [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| Llama 4 Maverick (400B) | ~200GB at 4-bit. |
| Pixtral Large (124B) | ~62GB at 4-bit; Mistral Research License (non-commercial). [Card](https://huggingface.co/mistralai/Pixtral-Large-Instruct-2411) |
| Mistral Small 4 (119B MoE, Mar 2026) | ~60GB at 4-bit despite the name. [Announcement](https://mistral.ai/news/mistral-small-4/) |
| GLM-4.6V (106B-A12B) | ~53GB at 4-bit. [Announcement coverage](https://news.aibase.com/news/23480) |
| GLM-5V-Turbo | API-only, not open weights. |
| Pixtral 12B, Gemma 3 27B | Fit, but strictly dominated by successors in the same license family (Mistral Small 3.2; Gemma 4 — which also swaps the restrictive Gemma Terms for Apache-2.0). Gemma 3 has **no** documented trained grounding. [Gemma 3 card](https://ai.google.dev/gemma/docs/core/model_card_3) |

---

## Per-candidate notes

### Qwen3.5-27B / Qwen3.6-27B — the smartest fit, and the one to test

- **What they are:** Qwen went natively multimodal in Feb 2026: every Qwen3.5/3.6 checkpoint is a
  VLM ("Causal Language Model with Vision Encoder"), superseding the separate Qwen3-VL line.
  Lineup: 0.8B/2B/4B/9B/27B dense + 35B-A3B/122B-A10B/397B-A17B MoE (3.5); 27B + 35B-A3B (3.6).
  All Apache-2.0. Sources: [Qwen3.5-27B card](https://huggingface.co/Qwen/Qwen3.5-27B),
  [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B),
  [QwenLM/Qwen3.5 → Qwen3.6 repo](https://github.com/QwenLM/Qwen3.5).
- **Head-to-head vs Claude, from Qwen's own card tables** (vendor-published — treat with the usual
  skepticism, but it is the only primary-source Claude comparison that exists for any fitting model):
  - Qwen3.5-27B: OCRBench **89.4** vs Claude Sonnet 4.5 **76.6**, GPT-5-mini 82.1; MMMU-Pro **75.0**
    vs Sonnet 4.5 68.4; CharXiv-RQ 79.5 vs 67.2; RefCOCO avg **90.9** (no Claude number — Anthropic
    doesn't do RefCOCO). ([card](https://huggingface.co/Qwen/Qwen3.5-27B))
  - Qwen3.6-27B: OCRBench 89.4, RefCOCO avg **92.5**, MMMU 82.9 vs Claude Opus 4.5 80.7, MMMU-Pro
    75.8 vs Opus 4.5 70.6. ([card](https://huggingface.co/Qwen/Qwen3.6-27B))
- **Grounding:** first-class trained capability with published RefCOCO scores. Exact coordinate
  convention for 3.5/3.6 not spelled out on the cards (Qwen3-VL's cookbook says "relative position
  coordinates… both boxes and points", [repo](https://github.com/qwenlm/qwen3-vl)) — **verify the
  output convention empirically before wiring into `_norm_to_px`.**
- **Multi-image:** documented for the Qwen3-VL lineage (multi-image inference examples in the
  official repo); the 3.5/3.6 cards don't restate it explicitly — flagged unverified for the exact
  9-image case, but no documented limit either.
- **Fit & runtimes:** 27B dense → Q4_K_M ≈ 16GB weights, comfortable on 24GB CUDA with KV headroom;
  Q4-Q6 comfortable on 36GB unified (macOS wires ~75% of RAM to GPU, so budget ~27GB usable —
  rule-of-thumb, unverified for M5 Max). Official support: vLLM/SGLang; "llama.cpp supports Qwen3.6
  (text & vision)"; "mlx-vlm (vision + text) support Qwen3.6" ([repo](https://github.com/QwenLM/Qwen3.5));
  [Ollama `qwen3.5:27b`](https://ollama.com/library/qwen3.5:27b);
  [unsloth GGUFs](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF); 262K context.
- **Cheaper sibling:** Qwen3.5-35B-A3B (MoE, ~3B active, Q4 ≈ 18-19GB) would be markedly faster on
  the Mac; no per-size RefCOCO/OCRBench verified for it in this pass.

### Gemma 4 31B / 26B-A4B — best box-format match, second pick

- Released Mar 31/Apr 2, 2026, **Apache-2.0** (Gemma dropped its bespoke license).
  [HF launch blog](https://huggingface.co/blog/gemma4),
  [model card](https://ai.google.dev/gemma/docs/core/model_card_4),
  [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/).
- **Boxes are native and use the pipeline's exact convention:** official docs specify JSON
  `{"box_2d": [y1, x1, y2, x2]}` with "normalized values relative to a 1000x1000 grid" — note the
  **y-first ordering** differs from stage 2's `[x, y, w, h]`, so a transform is still needed.
  Detection/pointing/OCR/handwriting are listed as trained capabilities.
  ([vision docs](https://ai.google.dev/gemma/docs/capabilities/vision/image),
  [model card](https://ai.google.dev/gemma/docs/core/model_card_4))
- **Multi-image is explicit:** "You can provide multiple images in a single prompt" (same docs page).
  Visual token budget is configurable per image (70-1120 tokens) — refs at low budget, panel at high.
- **OCR evidence is the weak spot:** Google publishes OmniDocBench (31B: 0.131 edit distance,
  lower=better) and MMMU-Pro 76.9, but no DocVQA/OCRBench — no direct number against Qwen or Claude.
- **Fit & runtimes:** official QAT q4_0 GGUF ([google/gemma-4-31B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf))
  ≈ 15-17GB; day-0 llama.cpp, mlx-vlm, vLLM (w4a16), LM Studio, Ollama. The 26B-A4B MoE (3.8B
  active, Q4 ≈ 13GB) trades a benchmark notch for ~4B-class decode speed on the Mac.

### MiniCPM-V 4.5 — the efficiency pick

- 8.7B (Qwen3-8B + SigLIP2-400M). OCRBench **89.0** (reported > GPT-4o-latest 82.2), DocVQA **94.7**,
  TextVQA 82.2. First-class grounding via official cookbook: `<ref>/<box>` tags, 0-1000 normalized
  coords, JSON multi-object output. Apache-2.0 (previous commercial-registration requirement
  dropped). Official GGUF (Q4_K_M 5.0GB), int4, AWQ; vLLM ≥0.10.2, llama.cpp ≥b6282, official
  Ollama. [Card](https://huggingface.co/openbmb/MiniCPM-V-4_5),
  [paper](https://arxiv.org/abs/2509.18154),
  [grounding cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/inference/minicpm-v4_5_grounding.md),
  [GGUF](https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf).
- Caveats: no official RefCOCO score; multi-image documented only up to 2 images (the 9-image roster
  prompt is **unverified**); at 8.7B, expect weaker instruction-following on the composite task than
  a 27B. Siblings: MiniCPM-o 4.5 (9B omni, Feb 2026) has the family's best OCR (OCRBench 87.6,
  DocVQA 94.7) but **no documented grounding**; MiniCPM-V 4.6 (May 2026) is a 1.3B phone-class model
  — not a candidate despite the higher version number.

### Molmo2-8B — pointing specialist, wrong output primitive

- Ai2, Apache-2.0, Qwen3-8B backbone. Outputs **points, not boxes** (`<points coords=.../>`,
  0-1000 normalized). Multi-image QA is a first-class documented feature. On Ai2's 15-benchmark
  average it scores 63.1 vs Claude Sonnet 4.5's 59.6 and Qwen3-VL-8B's 59.5 (vendor table).
  [Card](https://huggingface.co/allenai/Molmo2-8B), [blog](https://allenai.org/blog/molmo2).
- For this pipeline a point inside each bubble could seed stage 3's blob detector (it already grows
  a mask from a near-white region), but the focus *box* and bubble extents would need extra
  engineering. No official OCR benchmark published. Keep as fallback, not primary.

### InternVL3.5 — strong on paper, weakest runtime story for these targets

- Apache-2.0; 38B = InternViT-6B + 32.8B LM. Grounding published: RefCOCO avg 89.1 (38B).
  [Card](https://huggingface.co/OpenGVLab/InternVL3_5-38B), [paper](https://arxiv.org/abs/2508.18265).
- 38B Q4 ≈ 21-22GB — marginal on 24GB once the 6B vision tower and KV cache are counted; 30B-A3B is
  the realistic size. llama.cpp's official support list names InternVL "2.5 and 3" only; no MLX
  support verified. vLLM/LMDeploy are the supported paths (CUDA box only). No InternVL4 exists as of
  this writing.

---

## Comic-domain evidence (the part generic benchmarks hide)

This is where the replace-vs-stay decision actually lives. Three primary sources:

1. **CoMix** (NeurIPS 2024 D&B, [paper](https://arxiv.org/abs/2407.03550)) — multi-task comic
   benchmark (Western + manga):
   - **Dialogue transcription: frontier APIs are genuinely good.** GPT-4(V) hybrid dialog score
     **93.1%** vs the manga-specialized Magi's 43.6% on Western comics. This is the best available
     proxy for Sonnet-tier verbatim bubble OCR — and no stock open 7-32B model has a published
     equivalent number on Western comics.
   - **Speaker identification is unsolved by everyone.** A trivial closest-character heuristic
     (38.4% recall) beat Magi (27.9%); no VLM did well. Stage 2's design — face refs + a human
     review gate at stage 5 — is well matched to this reality, but it means speaker attribution is
     the field most likely to regress silently if the model changes.
2. **MangaOCR / MangaVQA** ([paper](https://arxiv.org/abs/2505.20298)) — end-to-end text
   detection+recognition on manga pages: GPT-4o, Gemini 2.5 Flash, and Qwen2.5-VL all scored
   **0.0-0.9% Hmean** ("likely due to unfamiliarity with manga data and weak detection
   capabilities"), while a Qwen2.5-VL-7B *fine-tune* (MangaLMM) hit 71.5%. Two lessons: (a) asking
   any general VLM — API or local — to *localize* text in comic pages is the failure mode, which is
   why stage 3's blob-refinement of approximate boxes is the right design; (b) a small open model
   fine-tuned on domain data can leapfrog frontier APIs here, so the ceiling for local is high if
   fine-tuning is ever on the table.
3. **Magi v2** ("Tails Tell Tales", ACCV 2024, [paper](https://arxiv.org/abs/2408.00298),
   [weights](https://huggingface.co/ragavsachdeva/magiv2)) — the only published system doing
   reference-crop speaker attribution well, via a character bank (names + exemplar crops) plus
   **speech-bubble-tail geometry**, not prompting. Already noted in the README as the stage-1
   upgrade path. Two catches: **non-commercial license** (commercial use requires contacting the
   author), and CoMix showed its speaker head transfers poorly from manga to Western art. Magi v3
   ([paper](https://arxiv.org/abs/2503.23344)) unifies detection+OCR+speaker matching and
   demonstrates pairing with a VLM.
   - Supporting tools: [comic-text-detector](https://github.com/dmMaze/comic-text-detector)
     (YOLOv5-based bubble/text boxes + masks, trained ⅓ on Western comics, no published metrics);
     [manga-ocr](https://github.com/kha-white/manga-ocr) (Japanese-only).

**No primary source benchmarks any model — open or API — on speaker attribution from reference face
crops via prompting.** That field's current quality on Sonnet is unmeasured anywhere except this
repo's own stage-5 review experience; any local swap must be A/B'd against it directly.

---

## The Claude baseline, precisely

- **Multi-image:** officially supported and documented (up to 100 images/request on 200k-context
  models; labeling each image is the recommended pattern — stage 2 already does this).
  [Vision docs](https://platform.claude.com/docs/en/build-with-claude/vision)
- **Boxes:** Anthropic now publishes a
  [coordinates guide](https://platform.claude.com/docs/en/build-with-claude/vision-coordinates) —
  Claude returns **absolute pixel coordinates** on the resized image, and the docs explicitly warn:
  "Claude's coordinate and localization outputs are approximate… verify outputs before relying on
  them." Grounding is not a trained, benchmarked capability the way it is for Qwen/Gemma/MiniCPM.
  (Stage 2 prompts for 0-1000 normalized boxes and gets usable-but-approximate results that stage 3
  repairs — consistent with the docs.) Note: **Sonnet 4.6 is standard-resolution tier (1568px long
  edge)**; at stage 2's 1400px sends this is fine, but small lettering is near the margin.
- **Structured output:** now GA on the API (`output_config.format` json_schema), which would remove
  the `_parse_json` failure branch *without* leaving the API.
  [Docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
- **Cost:** Batches API = official 50% discount ([docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing));
  measured cost in this repo ≈ **$1 per 112-panel run** (README). A local 27B at Q4 on the 24GB box
  runs at roughly 20-40 tok/s (order-of-magnitude, unverified for these exact models) — a full run
  is minutes-to-tens-of-minutes of GPU time. The economics are a rounding error either way; this
  decision is about quality, iteration speed, and offline capability, not money.
- One documented policy edge: Claude "cannot be used to name people in images." Fictional drawn
  characters don't normally trigger this, but it is a standing risk for a face-crop→name pipeline
  that pure-local models don't have (behavior on drawn characters: unverified).

---

## Recommendation

**Stay on the API for production output today; run a bounded experiment with Qwen3.6-27B before the
next big batch; adopt hybrid only if going local for non-cost reasons.**

1. **Stay (default).** At ~$1/run the API saving is irrelevant, Sonnet-tier transcription of comic
   lettering is the best-evidenced in class (CoMix 93.1% for GPT-4-class models; no fitting open
   model has a published Western-comics number), and stage 5's human gate exists precisely because
   speaker attribution is fragile everywhere. Cheap immediate win regardless: switch stage 2 to the
   API's structured outputs to eliminate JSON parse failures.
2. **Experiment (recommended, ~1 hour of GPU time).** Qwen3.6-27B Q4 via vLLM (CUDA box) or mlx-vlm
   (Mac) with json_schema-constrained decoding, on 20 already-processed panels; diff against
   `work/understanding.json` field by field. Decision fields, in order of risk: verbatim text
   (character-exact), speaker (vs stage-5-corrected truth), bubble_box IoU-after-stage-3-refinement,
   focus_box plausibility. The official numbers (OCRBench 89.4 > Sonnet 4.5's 76.6; RefCOCO 92.5 —
   a capability Claude documents as approximate) make it genuinely plausible that boxes come back
   *better* locally, with unknown-direction deltas on transcription and speakers. Backups if it
   disappoints: Gemma 4 31B QAT (native 0-1000 `box_2d`, explicit multi-image) and MiniCPM-V 4.5
   (fastest, strongest small-model OCR).
3. **Hybrid (if local matters for scale/offline/ToS-comfort).** Don't split fields between local
   and API — split *stages*: keep a detector for geometry and a VLM for language. Concretely:
   comic-text-detector (or licensed Magi heads) for bubble boxes/masks feeding stage 3, plus the
   best local VLM running on bubble *crops* for verbatim text and on the panel for scene/emotion,
   with speaker attribution remaining the review-gate field it already is. MangaOCR's numbers say
   whole-page text localization is the thing to take away from the VLM — whichever VLM it is —
   not transcription.

### What would change this verdict

- A published Western-comics dialogue-transcription score for Qwen3.5/3.6-class models at or above
  the GPT-4 CoMix mark (watch the [ICDAR COMICS challenge](https://rrc.cvc.uab.es/?ch=31) and the
  [awesome-comics-understanding](https://github.com/emanuelevivoli/awesome-comics-understanding) tracker).
- The stage-2 A/B in (2) coming back clean on verbatim text and no worse on speakers.
- Run volume growing to where API latency/cost or per-panel iteration (re-prompting single panels
  during review) makes local turnaround materially nicer.

---

## Source index

Primary sources cited inline throughout. Vendor benchmark tables (Qwen, OpenBMB, Ai2, Google,
Mistral) are self-reported; the CoMix and MangaOCR/MangaVQA numbers are peer-reviewed third-party
measurements; Anthropic capability statements are from official platform docs. Claims marked
**unverified** could not be traced to a primary source during this pass — the notable ones:
9-image prompts on MiniCPM-V and Qwen3.5/3.6 (documented for neither beyond small counts),
llama.cpp support for InternVL3.5, exact grounding coordinate conventions for Qwen3.5/3.6, the
~75% macOS GPU-wired-memory ceiling on M5 Max, and local decode-speed estimates.
