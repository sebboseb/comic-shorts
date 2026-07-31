# What the reference scripts actually do (transcript analysis, 2026-07-31)

Four reference TikToks transcribed locally (whisper-base via Voicebox; proper
nouns garble, structure survives). Raw text in `~/Dev/comicops/tiktoks/transcripts/`,
videos in `~/Dev/comicops/tiktoks/`. Compared against our ep01/ep02 manifests.

## The one big finding

**Every reference channel writes in forward-rolling chains; we write in
staccato fragments.** Same 69-second Kingpin video is essentially three
enormous sentences:

> "…Peter decides to pretend he bumped into Fisk by accident so he can
> secretly plant a tracker on his shoulder, and the Kingpin gets super
> annoyed telling the kid to watch his hands, to which Peter irritably
> replies that he's not a kid, and even leaves Fisk somewhat impressed by
> his nerve, after which he simply apologizes and goes on his way."

Our ep01 opening, for contrast:

> "This is Richard Rider. Nova. The last living officer of the Nova Corps.
> The whole Corps. Gone. And it started when something punched through the
> Crunch. The Crunch being the literal edge of our universe. So yeah."

The fragment style *reads* punchy on paper but depends on a performer
selling each pause; a TTS voice turns it into a flat list. The chain style
carries its own momentum — every clause hands off to the next ("to which…",
"only for…", "after which…"), so there is never a full stop where the
viewer can leave. This is a retention device, not a grammar accident, and
all four references use it regardless of tone:

- Kingpin (69s, neutral-wry): one chain per scene, adverb-rich attribution.
- Comicfy Jeff (60s): "When X… After Y sees… only to find… before rushing
  him to the hospital, where…"
- Spidey karaoke (48s, maximum irreverence): same chains, slang register:
  "Right on cue, the first Scion appears and starts beating the [--] out of
  him. Peter immediately changes his mind."
- Injustice (7 min serial): chains plus relentless location anchors.

## The full pattern list

1. **Present tense, always.** All four references, every sentence.
2. **Rolling syntax.** Subordinate connectives chain 2-4 events per
   sentence: *when, after, to which, only for, only to realize, which
   causes, before, after which, however, right on cue*. Full stops land
   AFTER reversals, never at tension points.
3. **Reported dialogue with delivery adverbs.** "Peter *irritably* replies
   that he's not a kid", "*casually* remark that he must have overdone it".
   Verb variety: replies / scoffs / remarks / asks / shouts / deduces /
   mentions. Verbatim quotes appear inverted, attribution trailing:
   "'You must be a Robin, and you're the son I take it,' says Cobblepot."
4. **Setup → reversal as the engine** (our prompt already says this; the
   references execute it *inside* sentences): "asks if Peter isn't afraid
   of him, but Peter just scoffs back that he's not", "pretends to throw a
   punch… Spider-Man doesn't so much as twitch a muscle."
5. **Location/time anchors open sentences** in multi-scene stories:
   "Back in Australia", "On the other side of the battlefield",
   "Elsewhere", "Seconds later", "Now outside the bunker", "At that same
   time". (Already in our rules — references confirm.)
6. **Endings state a new goal, not a mood.** "…only for Peter to
   immediately throw the money away and start tailing Fisk inside **to
   uncover his plans**." The cliffhanger is an intention.
7. **Cold opens launch mid-action** with a "When/After + event" clause —
   never scene-setting, never a posed question. Serial episodes ("Part 33")
   open on the continuing action directly: "After Dr. Fate is impaled by
   the Spear of Destiny, John Constantine decides to…"
8. **Irreverence is a register, not a structure.** The karaoke channel
   proves our configured narrator persona works — epithet insults ("this
   bum", "Temu Dr. Strange knockoff"), slang ("pulls up on") — but it sits
   ON TOP of the same rolling chains. Our current output has the register
   and lacks the chains.
9. **Emotional beats are named plainly** when they land: "Kate yells at him
   in disappointment while Jeff is left embarrassed."

## What this means for the pipeline

- Stage 4 prompt: replace the "short sentences hit" fragment aesthetic with
  rolling-chain guidance; keep one beat per shot but require consecutive
  shot lines to chain grammatically (start on connectives) so the
  concatenated narration flows as one told story. Done 2026-07-31.
- Word-level karaoke captions (the highlight-word style) pair naturally
  with chained sentences — future stage 7 work.
- The 7-minute serial proves episode length is not the constraint;
  clarity + momentum is.

## Caveats

- whisper-base mis-hears proper nouns and slang; one transcript contains a
  garbled slur-like artifact that the burned-in captions show was "bum" —
  treat transcripts as structural evidence, not quotable text.
- Music beds are under all narration; loudness balance not analyzed here.
