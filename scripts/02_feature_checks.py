"""Reconcile the matched-human set size, and re-measure the LENGTH confound.

(1) The finalized plan says "426 human answers" each get a 1:1 LLM pair. The corrected
    master may differ, so recompute the usable human set under candidate filters to fix N.
(2) Plan 3-c cites lexical_diversity vs length r=-0.62 (stale, from prior work) and asks
    whether to re-measure. We re-measure on the CURRENT data with the agreed STYLE7.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.data_io import load_master  # noqa: E402
from src.features.stylo import assert_disjoint  # noqa: E402
from src.features.style7 import STYLE7_NAMES, style7_matrix  # noqa: E402

assert_disjoint()
print("STYLE7 / STYLO disjointness: OK\n")

df = load_master(usable_only=False)


def has_text(s):
    return str(s).strip().lower() not in {"missing", "unclear", ""}


def kw_ok(ks):
    return len([k for k in ks if k not in ("missing", "unclear")]) > 0


A = df["text"].map(has_text)          # has usable english_text
B = df["quality"] != "low"            # not flagged low quality
C = df["keywords"].map(kw_ok)         # has real keywords (needed to drive 1:1 generation)

print("Matched-human candidate sizes (reconcile the plan's '426'):")
print(f"  total rows                       : {len(df)}")
print(f"  A: has english_text              : {int(A.sum())}")
print(f"  B: quality != low                : {int(B.sum())}")
print(f"  C: valid keywords                : {int(C.sum())}")
print(f"  A & C                            : {int((A & C).sum())}")
print(f"  A & B                            : {int((A & B).sum())}")
print(f"  A & B & C                        : {int((A & B & C).sum())}")
print(f"  A & C, per question:\n{df[A & C].groupby('question').size().to_string()}")

# Length confound: correlation of n_words with each other STYLE7 feature
use = df[A & B]
X = style7_matrix(use["text"].tolist())
sdf = pd.DataFrame(X, columns=STYLE7_NAMES)
length = sdf["n_words"]
print("\nLength (n_words) vs each STYLE7 feature  [re-measure of the -0.62]:")
for c in STYLE7_NAMES:
    if c == "n_words":
        continue
    r = np.corrcoef(length, sdf[c])[0, 1]
    print(f"  corr(n_words, {c:18}) = {r:+.3f}")
