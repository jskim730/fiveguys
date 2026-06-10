# R2 - aggressive style coaching (content fixed)

This file documents the actual R2 prompt implemented in `src/generation/generate.py`.

## Purpose

R2 keeps content fixed but aggressively coaches the writing style toward short, informal student-like responses.

## Fixed across all rounds

* Persona: `a data science undergraduate writing an in-class discussion answer`
* Content: the paired human answer's `keywords`
* Output format: answer text only
* Slides: not used
* Semantic embeddings: not used

## Style instruction

```text
Write like a real student answering quickly in class:
very short,
1-3 sentences,
around 40 words,
use first person,
plain vocabulary,
few commas,
"I think" / "maybe" allowed,
sentence fragments are okay,
don't define concepts,
don't sound polished,
give one small personal example if natural.
```

## Interpretation

R2 is designed to aggressively suppress common surface-level LLM cues such as verbosity, polished explanations, textbook-style definitions, and impersonal tone.

Because the keywords remain fixed, differences between R0, R1, and R2 are intended to reflect style coaching rather than content changes.
