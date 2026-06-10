"""Ablation Ⓑ — which STYLE7 feature is the strongest R0 'tell', and does length dominate?

Per-question human(0) vs naive R0(1), 5-fold AUC using ONE STYLE7 feature at a time
(macro-averaged over questions). Also: full STYLE7 vs STYLE7 with n_words removed.

    python scripts/08_ablation_style7.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.analysis.classify import load_pairs  # noqa: E402
from src.data_io import load_config  # noqa: E402
from src.features.style7 import STYLE7_NAMES, style7_matrix  # noqa: E402

cfg = load_config()
SEED = cfg.get("seed", 42)


def macro_auc(cols, pairs):
    aucs = []
    for q, (texts, y, nh, nl) in pairs.items():
        X = style7_matrix(texts)[:, cols]
        y = np.asarray(y)
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        oof = np.full(len(y), np.nan)
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs))


def single_dir_auc(i, pairs):
    """Directional separability of ONE raw feature (no model, so >0.5 / <0.5 shows direction)."""
    aucs = []
    for q, (texts, y, nh, nl) in pairs.items():
        x = style7_matrix(texts)[:, i]
        aucs.append(roc_auc_score(np.asarray(y), x))
    return float(np.mean(aucs))


pairs = load_pairs("R0")
rows = []
for i, name in enumerate(STYLE7_NAMES):
    a = single_dir_auc(i, pairs)
    rows.append({"feature": name, "single_AUC": round(a, 3), "power|AUC-0.5|": round(abs(a - 0.5), 3)})
res = pd.DataFrame(rows).sort_values("power|AUC-0.5|", ascending=False)

full = macro_auc(list(range(7)), pairs)
nolen = macro_auc([i for i in range(7) if STYLE7_NAMES[i] != "n_words"], pairs)

print("Single-feature R0 detectability (human vs naive LLM), per-question macro AUC:")
print(res.to_string(index=False))
print(f"\nfull STYLE7 AUC = {full:.3f}   |   STYLE7 without n_words = {nolen:.3f}")
print("(AUC<0.5 = feature points the HUMAN way; sort key is distance from 0.5)")
res.to_csv(ROOT / "results/tables/ablation_style7.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 4))
order = res.sort_values("single_AUC")
colors = ["#c0392b" if v > 0.5 else "#2c6fbb" for v in order["single_AUC"]]
ax.barh(order["feature"], order["single_AUC"], color=colors)
ax.axvline(0.5, ls="--", c="gray", lw=1)
ax.set(xlim=(0, 1.0), xlabel="single-feature R0 AUC (red=LLM-ward, blue=human-ward)",
       title="Which STYLE7 feature is the 'tell'?")
fig.tight_layout()
fig.savefig(ROOT / "results/figures/fig10_ablation_style7.png", dpi=140)
print("saved -> results/figures/fig10_ablation_style7.png  &  tables/ablation_style7.csv")
