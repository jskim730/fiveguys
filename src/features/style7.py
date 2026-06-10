"""STYLE7 — surface style features (the 'tells' that R1/R2 prompts try to erase).

Exact team-agreed definitions from the finalized design (fixed for reproducibility):
n_words, avg_word_len, avg_sent_len, lexical_diversity, first_person_ratio,
hedging_ratio, comma_density. Content-agnostic; interpretable via logistic coefficients.
This is OUR code, not an LLM — it just counts things.
"""
from __future__ import annotations

import re

import numpy as np


def tokens(text):
    """Word tokens: lowercased, letters + apostrophe."""
    return re.findall(r"[a-zA-Z']+", str(text).lower())


FIRST_PERSON = {"i", "my", "we", "me", "our", "mine", "us", "i'm", "i've", "i'd", "i'll"}
HEDGE = ["maybe", "perhaps", "probably", "i think", "not sure", "i guess",
         "kind of", "sort of", "it seems", "possibly", "i believe", "i feel like"]
STYLE7_NAMES = ["n_words", "avg_word_len", "avg_sent_len", "lexical_diversity",
                "first_person_ratio", "hedging_ratio", "comma_density"]


def style7(text):
    """7 surface features for one response -> np.array(float, shape=(7,))."""
    t = str(text)
    low = t.lower()
    w = tokens(t)
    n = len(w) or 1
    sents = [s for s in re.split(r"[.!?]+", t) if s.strip()] or [""]
    fp = sum(1 for tok in w if tok in FIRST_PERSON)
    hed = sum(low.count(h) for h in HEDGE)
    return np.array([
        len(w),                                          # 1 word count
        float(np.mean([len(x) for x in w])) if w else 0.0,  # 2 mean word length
        n / len(sents),                                  # 3 mean sentence length
        len(set(w)) / n,                                 # 4 lexical diversity = unique/total
        fp / n,                                          # 5 first-person ratio
        hed / n,                                         # 6 hedging ratio
        t.count(",") / n,                                # 7 comma density
    ], float)


def style7_matrix(texts):
    """Stack STYLE7 for many texts -> np.ndarray(n, 7)."""
    return np.vstack([style7(t) for t in texts])
