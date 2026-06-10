# R0 - naive baseline (naive prompt + slides, NO content control)

Role: simulate DIVERSE undergraduates who took this lecture (fixed persona, all rounds).
Grounding: include the lecture slides for this discussion -> {SLIDES}
Output: a JSON array of {N} answers to the question below; vary engagement (high/mid/low);
do NOT make them textbook-uniform; keep realistic messiness. NO keywords / no content control.

Fairness (fix across rounds): persona, temperature, top_p, use_slides.

Question context: {QUESTION_CONTEXT}

Return ONLY: [{"engagement": "...", "text": "..."}, ...]  -> later normalized to id/q/round="R0".
