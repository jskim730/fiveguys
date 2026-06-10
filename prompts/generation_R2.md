# R1 - light style coaching (content fixed)

This file documents the actual R1 prompt implemented in `src/generation/generate.py`.

## Purpose

R1 keeps the content fixed through the paired human response's keywords, but adds light style coaching intended to move the answer closer to the writing style of undergraduate discussion responses.

## Fixed across all rounds

* Persona: `a data science undergraduate writing an in-class discussion answer`
* Content: the paired human answer's `keywords`
* Output format: answer text only
* Slides: not used
* Semantic embeddings: not used

## Style instruction

```text
Write like an undergraduate quickly noting a thought:
keep it fairly short,
use first person,
plain everyday words,
don't over-explain,
don't sound like a textbook.
```

## Interpretation

R1 is intended to reduce obvious GPT-style formality while preserving the same content constraints as R0.
