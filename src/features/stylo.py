"""STYLO — stylometric signature (independent judge; prompts do NOT target it).

Function-word relative frequencies (DISJOINT from STYLE7) + character 3-grams (top 200)
+ punctuation density (comma excluded — owned by STYLE7). Stateful: fit the n-gram
vocabulary on TRAIN only, then transform (prevents leakage). Disjointness from STYLE7 is
asserted so the two feature sets share no token — that is what keeps STYLO a genuinely
independent judge of "real humanization vs surface gaming". OUR code, not an LLM.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from .style7 import FIRST_PERSON, tokens

# Function-word list; STYLE7-owned tokens (1st person, hedge unigrams, comma) are removed
# so STYLO and STYLE7 stay disjoint.
_FW_RAW = """the a an of to in for on with as at by from that this these those and or but so if then
than because while although however therefore thus also not no very just more most some any all
can could would should will shall may might must do does did have has had been being be is are was
were am of into onto upon about over under above below between among through during before after
again further once here there when where why how what which who whom its their his her your you
they them he she it we us our your what whose where""".split()
HEDGE_UNIGRAMS = {"maybe", "perhaps", "probably", "possibly"}
STYLE7_OWNED = FIRST_PERSON | HEDGE_UNIGRAMS | {","}                  # excluded from STYLO
FUNCTION_WORDS = [w for w in dict.fromkeys(_FW_RAW) if w not in STYLE7_OWNED]
PUNCT = list(".!?;:'\"()-/")                                          # comma is STYLE7's


def assert_disjoint():
    """Guarantee STYLO shares no token with STYLE7 (true independent judge)."""
    overlap = set(FUNCTION_WORDS) & STYLE7_OWNED
    assert not overlap, f"STYLO/STYLE7 token overlap: {overlap}"
    assert "," not in PUNCT, "comma is STYLE7-owned; must be excluded from STYLO"


class Stylometry:
    """Function-word rel-freq + char 3-gram (top 200) + punctuation density.

    fit() fixes the char-ngram vocabulary (on TRAIN); transform() vectorizes.
    """

    def __init__(self, char_ngram=(3, 3), max_char=200, parts=("fw", "char", "punct")):
        self.parts = tuple(parts)   # subset for component ablation; default = all three
        self.fw = FUNCTION_WORDS
        self.cv = (CountVectorizer(analyzer="char_wb", ngram_range=char_ngram,
                                   max_features=max_char, lowercase=True)
                   if "char" in self.parts else None)

    def fit(self, texts):
        if self.cv is not None:
            self.cv.fit([str(t) for t in texts])
        return self

    def transform(self, texts):
        char = (self.cv.transform([str(t) for t in texts]).toarray().astype(float)
                if self.cv is not None else None)
        rows = []
        for i, t in enumerate(texts):
            w = tokens(t)
            n = len(w) or 1
            nchar = max(len(str(t)), 1)
            vec = []
            if "fw" in self.parts:
                vec += [sum(1 for tok in w if tok == f) / n for f in self.fw]   # function-word freq
            if "char" in self.parts:
                vec += list(char[i] / nchar)                                    # char 3-gram density
            if "punct" in self.parts:
                vec += [str(t).count(p) / nchar for p in PUNCT]                 # punctuation density
            rows.append(np.array(vec, dtype=float))
        return np.vstack(rows)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


def stylo_matrix(texts):
    """Descriptive convenience: fit+transform on the same texts.
    NOT for train/test eval (would leak the char-ngram vocab) — use Stylometry() there.
    """
    return Stylometry().fit_transform(texts)
