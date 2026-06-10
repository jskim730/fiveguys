# R0 - naive baseline (content fixed, no style coaching)

This file documents the actual R0 prompt implemented in `src/generation/generate.py`.
The code, not this markdown file, is the source used for generation.

## Purpose

R0 is the naive baseline. It receives the same content control as every other round,
but it receives **no additional style coaching**. This captures the generator's default
voice when asked to answer with the paired human response's keywords.

## Fixed across all rounds

- Persona: `a data science undergraduate writing an in-class discussion answer`
- Content: the paired human answer's `keywords`
- Output format: answer text only
- Slides: not used
- Semantic embeddings: not used

## Prompt template

```text
You are {PERSONA}.
Write ONE answer to an in-class data-science discussion. Cover these points: {KEYWORDS}.
Return ONLY the answer text — no preamble, no quotation marks, no lists.
```

## Style instruction

None.

In code, this is:

```python
STYLE["R0"] = ""
```
