# TTS narrator options — replacing kokoro (research, 2026-07-31)

Scope: replacement narration voice for comic-shorts stage 6 (~110 s / ~2,000–3,000
chars per episode, a few episodes/week). Current setup: kokoro preset via
jamiepine/voicebox REST API. All claims below were checked against primary
sources (GitHub repos, HF model cards, official pricing/docs pages) unless
explicitly flagged as thin/secondary.

## TL;DR — recommendation path

1. **Best local option: Qwen3-TTS 1.7B, already inside voicebox.**
   Voicebox's *default* engine (`engine: "qwen"`) is now Qwen3-TTS — Apache 2.0,
   released Jan 22 2026, 3-second zero-shot voice cloning, and it runs natively
   on Apple Silicon via mlx-audio (voicebox loads
   `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16`, ~4.5 GB — trivial for 36 GB).
   The `qwen_custom_voice` sibling engine accepts the `instruct` param for
   natural-language delivery direction ("dramatic movie-trailer narrator,
   building tension") — the exact knob kokoro ignored. **Concrete move: update
   voicebox to v0.5.x, create/clone a narrator profile on the qwen engine, and
   stage6_tts.py barely changes** (same `/generate` contract; add
   `engine`, pass manifest `emotion` through `instruct`).
2. **Second local option: Chatterbox family (MIT).** Original 0.5B has
   `exaggeration`/`cfg_weight` expressiveness knobs; new **Chatterbox Turbo**
   (350M, Dec 2025) adds paralinguistic tags (`[laugh]`, `[sigh]`). Caveats on
   this Mac: voicebox forces Chatterbox to **CPU on macOS** ("known MPS tensor
   issues" per its source), and voicebox hardcodes exaggeration=0.5 — the knobs
   are only reachable by running Chatterbox yourself (PyTorch or the
   mlx-community MLX ports).
3. **Best paid option: ElevenLabs Creator plan.** `eleven_multilingual_v2`
   (most stable long-form) or `eleven_v3` (most expressive, audio tags).
   ~2,500 credits/episode → ~13 eps/month ≈ 26k–39k credits. Starter ($6/mo,
   30k credits, commercial license) is borderline; **Creator ($22/mo, $11 first
   month, 121k credits)** is comfortable at roughly **$0.20–0.45 per episode**.
   Python SDK `elevenlabs`, endpoint `POST /v1/text-to-speech/{voice_id}`.
4. **What the reference channels use:** no channel-level primary evidence
   exists (creators don't disclose), but ElevenLabs is the default in every
   faceless-shorts tutorial, and ElevenLabs' own help center names **Brian,
   Liam, Adam** (male) as the voices "creators use every day" on
   TikTok/YouTube — Adam is the viral deep-authoritative shorts staple.

---

## 1. What comic-recap channels actually use

**Honest status: evidence is thin.** No creator interview, video description,
or community thread was found that names the exact tool/voice for the
Injustice-recap TikTok accounts or Comicfy (`youtube.com/@Comicfy_`, "The GOAT
of comic summary shorts" — its About page discloses nothing about narration
tooling). Multiple searches (reddit/r/NewTubers, tutorial articles, TikTok
discover pages) produced no channel-specific attribution. Treat everything
below as ecosystem-level evidence, not channel-level proof.

What the primary and near-primary sources do support:

- **ElevenLabs is the presumed default and claims it outright**: "If you've
  seen a clip using AI voices on TikTok, YouTube, Instagram, X… there's a good
  chance it was created with ElevenLabs"
  ([ElevenLabs help center](https://help.elevenlabs.io/hc/en-us/articles/19012226601233-What-voices-are-popular-on-TikTok-YouTube-and-social-media)).
- **The recap-staple voices, per ElevenLabs itself**: that same help article
  names **Brian, Liam, Adam** (male) and **Jessica, Matilda, Sarah** (female)
  as the popular social-media voices. It does not break them down by niche.
- **Adam** is repeatedly described (secondary sources) as the viral
  deep/calm/authoritative male voice behind motivational Shorts, history-fact
  videos and "stoic" TikToks — the same register comic recaps use
  ([theaianalystlab.com guide](https://theaianalystlab.com/how-to-get-the-viral-elevenlabs-adam-voice-for-shorts-tiktok-2025-guide/),
  [TikTok discover: "Adam voice ElevenLabs"](https://www.tiktok.com/discover/adam-voice-elevenlabs)).
  **Antoni**: no evidence found tying it to recap content — unverified.
- Faceless-channel tutorials uniformly assume ElevenLabs and recommend
  settings in the narrator range: stability 0.35–0.45, similarity 0.75–0.85,
  style 0.15–0.25
  ([aiproductivity.ai workflow guide](https://aiproductivity.ai/guides/elevenlabs-youtube-voiceover-workflow/) — secondary).
- **Alternatives in the wild**: TikTok's built-in TTS is a recognizable
  distinct style (robotic "TikTok voice") used by lower-effort accounts; no
  evidence surfaced of OpenAI TTS or Speechify being a comic-recap staple.
  The big *long-form* comic channels (Comicstorian, Comics Explained) are
  human-narrated, which is why they don't appear in AI-voice discussions —
  the AI-voice pattern is specifically a Shorts/TikTok recap phenomenon.
  (Human-narration claim: common knowledge, not verified against a primary
  source.)
- Compliance note: YouTube requires an altered/synthetic-content disclosure
  toggle for realistic AI voices (mentioned across 2026 tutorials; YouTube's
  own policy page not fetched — verify before publishing).

## 2. ElevenLabs, mid-2026

Primary sources: [pricing page](https://elevenlabs.io/pricing),
[docs overview](https://elevenlabs.io/docs/overview),
[TTS API reference](https://elevenlabs.io/docs/api-reference/text-to-speech/convert).

**Pricing (per elevenlabs.io/pricing, July 2026):**

| Plan | $/mo | Credits/mo | Commercial license |
|---|---|---|---|
| Free | 0 | 10,000 | **No** |
| Starter | 6 | 30,000 | Yes |
| Creator | 22 (first month 50% off) | 121,000 | Yes |
| Pro | 99 | 600,000 | Yes (+44.1kHz PCM via API) |

TTS costs ~1 credit per character. (A search-level claim says Flash/Turbo bill
0.5 credits/char on API and v3 is flat 1 credit/char; the pricing page fetch
only confirmed "1 credit per character" generally — **the 0.5x Flash rate is
unverified against the pricing page itself**.)

**Models** ([docs](https://elevenlabs.io/docs/overview)):
- `eleven_v3` — "most emotionally rich, expressive" model, 70+ languages,
  5,000-char limit per request, supports inline audio tags (e.g. `[whispers]`,
  `[excited]`).
- `eleven_multilingual_v2` — "most stable on long-form generations", 29
  languages, 10,000-char limit. The API's default `model_id`.
- `eleven_flash_v2_5` — fast/cheap (~75 ms latency), 40,000-char limit, "50%
  lower price per character for API generations".

**API** ([reference](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)):
`POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}` with
`{text, model_id, voice_settings:{stability, similarity_boost, style, speed,
use_speaker_boost}, seed, output_format}` (default `mp3_44100_128`; PCM/WAV
variants exist). Python SDK is the **`elevenlabs`** package:
`ElevenLabs(api_key=...).text_to_speech.convert(voice_id=..., text=..., model_id=...)`.

**Narrator voices**: ElevenLabs' own popularity list (help center, above):
Brian, Liam, Adam. Their blog post on faceless YouTube voices compares
platforms rather than naming preset voices, so "best narrator preset" beyond
that list is community lore, not primary-sourced. Practical approach: audition
Adam and Brian at stability ~0.4 / style ~0.2 on multilingual_v2, and the same
script on v3 with audio tags.

**Cost estimate for this pipeline**: 2,000–3,000 chars/episode ≈ 2–3k credits.
At 3 episodes/week (~13/month): 26k–39k credits/month.
- Free (10k): ~3–4 episodes, **no commercial license** — fine for tests only.
- Starter (30k, $6): covers ~2 eps/week reliably, 3/week only in light months.
- **Creator (121k, $22)**: ~40–60 episodes of headroom, ≈ **$0.20–0.45/episode**
  (at the $11 first-month price, half that). This is the realistic tier.

## 3. Open/local expressive TTS for M5 Max 36 GB

Primary-source matrix. "Mac path" = documented way to run on Apple Silicon.
[mlx-audio](https://github.com/Blaizzy/mlx-audio) (Apple-Silicon-native
inference lib) currently lists support for Kokoro, Qwen3-TTS, Higgs Audio
v2/v3, CSM, Dia, Chatterbox, Spark, OuteTTS and others (per README fetch).

| Model | License (weights) | Zero-shot clone | Expressiveness | Mac path | Verdict for narration |
|---|---|---|---|---|---|
| **Qwen3-TTS** 0.6B/1.7B ([GitHub](https://github.com/QwenLM/Qwen3-TTS), [HF](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)) | Apache 2.0 | Yes, **3 s** ref audio + transcript | `instruct` natural-language style control (CustomVoice), free-form **VoiceDesign** persona description | **Yes — mlx-audio; official-quality `mlx-community` conversions (bf16 ~4.5 GB, 4/8-bit smaller); already voicebox's default engine** | **Top pick.** Trained on 5M+ hours, 10 languages; the instruct/voice-design combo is exactly a "directable narrator" |
| **Chatterbox** 0.5B / **Multilingual v3** / **Turbo** 350M / Nano 110M ([GitHub](https://github.com/resemble-ai/chatterbox), [HF](https://huggingface.co/ResembleAI/chatterbox)) | MIT (all variants incl. [Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo), Dec 2025) | Yes, ~10 s ref clip | `exaggeration` (0.5 default, 0.7+ dramatic) + `cfg_weight` (lower ≈ slower/more expressive pacing) on original; `[laugh]`/`[cough]`/`[chuckle]` tags on Turbo/Nano; Nano runs 3x realtime on 8 CPU cores | PyTorch MPS claimed upstream but **voicebox forces CPU on macOS ("known MPS tensor issues")**; mlx-community ports exist (`chatterbox-turbo-fp16`, `Chatterbox-TTS-4bit` — community, low download counts) | **Second pick.** Resemble cites a 63.75% blind-test preference over ElevenLabs; strong cloning; knobs only usable outside voicebox |
| **Hume TADA** 1B/3B ([GitHub](https://github.com/HumeAI/tada), MIT code) | **Llama 3.2 Community License** (commercial OK under Meta's terms) | Yes, ref audio + transcript | Prosody handled dynamically; no explicit emotion tags documented | **Yes — official `HumeAI/mlx-tada-1b` / `mlx-tada-3b` MLX builds**; in voicebox as `tada` | **Third pick / dark horse.** Text-acoustic alignment design targets zero "off-script" hallucination and long coherent audio (voicebox cites 700+ s); Hume's pedigree is emotion research. Too new (Mar 2026) for a narration track record |
| Maya1 3B ([HF](https://huggingface.co/maya-research/maya1)) | Apache 2.0 | **No** — voices via natural-language description (`<description="40yo, low-pitch, warm">`) | 20+ inline emotion tags (`<laugh>`, `<sigh>`, `<whisper>`, `<angry>`…) | Not in mlx-audio's list; docs target 16 GB+ CUDA GPUs w/ vLLM | Interesting voice-design angle for character lines; no Mac story |
| IndexTTS-2 (Bilibili) ([GitHub](https://github.com/index-tts/index-tts)) | MIT code, but README says contact indexspeech@bilibili.com "for commercial usage" — ambiguous | Yes (`spk_audio_prompt`) | **Best-in-class emotion control**: emotion ref audio + `emo_alpha`, 8-dim emotion vectors, text-described emotion | **No** — requires CUDA 12.8+; no Apple Silicon support documented | Great controls, wrong hardware; also license ambiguity |
| VibeVoice 1.5B/7B (Microsoft) ([community fork](https://github.com/vibevoice-community/VibeVoice), [HF](https://huggingface.co/microsoft/VibeVoice-1.5B)) | MIT | Yes (speaker samples) | Long-form: up to 90 min, 4 speakers | Not in mlx-audio list; community-maintained since **Microsoft pulled the code** (Aug 2025) for misuse concerns | Multi-speaker long-form specialist; upstream abandonment is a real maintenance risk |
| Higgs Audio v2 / TTS 2 / v3 (Boson AI) ([GitHub](https://github.com/boson-ai/higgs-audio), [HF](https://huggingface.co/bosonai/higgs-tts-2-3b-base)) | **"Research and Non-Commercial License"** (custom; commercial requires separate license). v2-era weights now also listed `license:other` on HF | Yes, strong (75.7% win vs gpt-4o-mini-tts on EmergentTTS-Eval Emotions; "automatic prosody adaptation during narration") | Inline emotion/style/prosody control | mlx-audio lists Higgs v2/v3 support | **Quality-wise a top narrator, but license kills it** for a monetized channel |
| F5-TTS ([GitHub](https://github.com/SWivid/F5-TTS)) | Code MIT; **weights CC-BY-NC** (Emilia dataset) | Yes (~10 s convention) | Multi-style via ref audio; no tag system | Yes — pip install with MPS documented | Non-commercial weights → out |
| Fish Speech / OpenAudio **S1-mini** ([GitHub](https://github.com/fishaudio/fish-speech), [HF](https://huggingface.co/fishaudio/s1-mini)) | s1-mini **CC-BY-NC-SA**; newer S2 Pro under "Fish Audio Research License"; full S1/S2 are API products | Yes, 10–30 s | Rich: `[whisper]`, `[excited]`, `[angry]`, `(15,000+ tags)` claim | H200 benchmarks; no macOS docs | Expressive but non-commercial local weights → out |
| Orpheus 3B (Canopy Labs) ([GitHub](https://github.com/canopyai/Orpheus-TTS)) | Apache 2.0 | Yes (prompt-conditioned) | `<laugh>`, `<sigh>`, `<gasp>` etc. tags | vLLM-first; llama.cpp CPU path; not confirmed in current mlx-audio list | Solid 2025 model, largely superseded by 2026 entrants |
| Dia 1.6B / Dia2 (Nari Labs) ([GitHub](https://github.com/nari-labs/dia)) | Apache 2.0 | Yes, 5–10 s w/ speaker tags | `(laughs)`, `(sighs)` nonverbals | GPU-only per README (CUDA); mlx-audio lists Dia | **Dialogue** model; README warns >20 s inputs go unnaturally fast — wrong shape for 110 s narration |
| Sesame CSM-1B ([HF](https://huggingface.co/sesame/csm-1b)) | Apache 2.0 | Contextual (conversation-primed) | Conversational prosody, no direct controls | mlx-audio lists CSM | Built for dialogue agents, not directed narration |
| MegaTTS3 0.45B (ByteDance) ([GitHub](https://github.com/bytedance/MegaTTS3)) | Apache 2.0 | **Crippled**: WaveVAE encoder withheld — you must submit your ref audio to ByteDance's Google Drive queue for `.npy` latents | `p_w`/`t_w` intelligibility/expressiveness weights | CPU possible but slow | The cloning gate makes it unusable here |
| Kyutai TTS 1.6B ([HF](https://huggingface.co/kyutai/tts-1.6b-en_fr)) | CC-BY 4.0 | **No arbitrary cloning** — only pre-computed embeddings from their `tts-voices` repo (deliberate policy) | Streaming-focused | Rust/PyTorch server stack | Voice lock-in → out |

**Top 2–3 for "expressive storyteller narrator on this Mac":**

1. **Qwen3-TTS-12Hz-1.7B** (Base for cloning, CustomVoice/VoiceDesign for
   directed delivery). Apache 2.0, MLX-native, 3-s cloning, instruct control,
   and already wired into voicebox — lowest-friction upgrade by far.
2. **Chatterbox** (original for `exaggeration` knob, Turbo for tags). MIT,
   proven cloning quality, but on macOS you must run it yourself (CPU PyTorch
   or the mlx-community port) to touch the expressiveness knobs.
3. **Hume TADA-3B** (via voicebox `tada` engine or official MLX weights) as
   the long-form reliability option — evaluate against Qwen on a real script.

Suggested bake-off: one real ~2,500-char episode script through (a) voicebox
qwen engine + instruct, (b) voicebox tada engine, (c) standalone Chatterbox
with exaggeration 0.6–0.7 / cfg 0.35, (d) ElevenLabs Adam or Brian free tier.
Pick by ear; wire the winner into stage 6.

## 4. jamiepine/voicebox status (verified against source, main @ 2026-07-28)

Latest release **v0.5.0 "Capture" (2026-04-25)**; the big engine expansion was
**v0.4.0 (2026-04-16)** ([releases](https://github.com/jamiepine/voicebox/releases)).
Repo license MIT. Seven engines now (from `backend/models.py` `GenerationRequest.engine`):

```
qwen | qwen_custom_voice | luxtts | chatterbox | chatterbox_turbo | tada | kokoro
```

`POST /generate` schema (`backend/models.py`, `GenerationRequest`):
`profile_id`, `text` (≤50,000 chars), `language`, `seed`,
`model_size` (`1.7B|0.6B|1B|3B`), `instruct` (≤500 chars),
`engine` (default `"qwen"`), `personality` (LLM rewrite in-character),
`max_chunk_chars` (default 800), `crossfade_ms` (default 50), `normalize`,
`effects_chain`.

Key findings for our integration:

- **No exaggeration/pace/cfg params are exposed over the REST API.**
  `backend/backends/chatterbox_backend.py` hardcodes
  `exaggeration=0.5, cfg_weight=0.5, temperature=0.8` (globals; Hebrew gets a
  special preset). To use Chatterbox's expressiveness knobs you must bypass
  voicebox.
- **`instruct` is engine-dependent**: both chatterbox backends document it as
  "Unused (protocol compatibility)"; `qwen_custom_voice_backend.py` passes it
  through ("natural language instruction for tone/emotion/prosody control").
  Chatterbox **Turbo** instead takes paralinguistic tags inline in `text`
  (`[laugh]`, `[cough]`, …).
- **Apple Silicon**: Qwen3-TTS runs through **mlx-audio**
  (`backend/backends/mlx_backend.py`, models
  `mlx-community/Qwen3-TTS-12Hz-{1.7B,0.6B}-Base-bf16`); both chatterbox
  engines force CPU on macOS ("known MPS tensor issues"). So on this Mac,
  qwen is the fast path inside voicebox.
- Profiles remain engine-bound (v0.4.0 "enforced preset/profile engine
  compatibility") — a new narrator profile is needed per engine.
- Docs/README reference the API at `http://127.0.0.1:17493` (MCP server is
  pinned there); whether the REST port is now fixed rather than random wasn't
  verified in source — **keep the port-probe in stage6_tts.py until confirmed**.

## Evidence gaps / unverified claims

- **Q1 is inference, not proof**: no primary source ties any named comic-recap
  channel to a specific TTS tool or voice. Adam/Brian-for-recaps is
  extrapolated from ElevenLabs' own popularity list + viral-voice guides.
- ElevenLabs Flash "0.5 credits/char on API" and any v3 credit discount:
  search-result claims, not confirmed on the pricing page fetch.
- Chatterbox blind-test preference over ElevenLabs (65.3%/63.75% figures
  circulate): Resemble publishes Podonos evaluations, but the exact number
  wasn't verified against Resemble's own report.
- TADA 700+ s coherent audio: stated in voicebox's engine listing and Hume
  coverage; the HumeAI/tada README fetch didn't surface the number.
- mlx-community Chatterbox ports: community conversions with low download
  counts — quality/parity untested.
- Qwen3-TTS GitHub README's "96 GB RAM recommended" applies to its CUDA
  flash-attention build path, not MLX inference; MLX memory needs are just
  model size (~4.5 GB bf16). Reasonable inference, not a documented claim.
