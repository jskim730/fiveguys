"""Ablation Ⓐ — which STYLO component carries the irreducible signature?

Re-runs the leakage-proof trajectory (frozen R0 classifier, held-out humans) with the
STYLO judge restricted to: function words only / char-3gram only / punctuation only / full.
A component that keeps a high held-out R2 AUC is where the un-coachable signature lives.

    python scripts/07_ablation_stylo.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.round_eval import ROUNDS, run  # noqa: E402
from src.features.stylo import Stylometry  # noqa: E402


class StyloPartsF:
    def __init__(self, parts):
        self.parts = parts

    def fit(self, texts):
        self.s = Stylometry(parts=self.parts).fit(texts)
        return self

    def transform(self, texts):
        return self.s.transform(texts)


VARIANTS = {
    "full (fw+char+punct)": ("fw", "char", "punct"),
    "function words": ("fw",),
    "char 3-gram": ("char",),
    "punctuation": ("punct",),
}
# zero-arg factories with correct closure over parts
judges = {name: (lambda p: (lambda: StyloPartsF(p)))(parts) for name, parts in VARIANTS.items()}

_, _, summary = run(judges=judges, n_repeats=20)
piv = summary.pivot(index="judge", columns="round", values="auc_mean")[ROUNDS].reindex(list(VARIANTS))
print("STYLO component ablation — held-out AUC (frozen R0 classifier):")
print(piv.round(3).to_string())
piv.round(3).to_csv(ROOT / "results/tables/ablation_stylo.csv")

fig, ax = plt.subplots(figsize=(7, 4))
x = range(len(piv))
ax.bar([i - 0.2 for i in x], piv["R0"], 0.4, label="R0 (naive)", color="#9ecae1")
ax.bar([i + 0.2 for i in x], piv["R2"], 0.4, label="R2 (mimic)", color="#1b7837")
ax.axhline(0.5, ls="--", c="gray", lw=1)
ax.set_xticks(list(x))
ax.set_xticklabels(list(VARIANTS), fontsize=9)
ax.set(ylim=(0, 1.05), ylabel="held-out STYLO AUC",
       title="Where the signature lives: STYLO component ablation")
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "results/figures/fig9_ablation_stylo.png", dpi=140)
print("saved -> results/figures/fig9_ablation_stylo.png  &  tables/ablation_stylo.csv")
