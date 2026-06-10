"""Ablation Ⓒ — which single coaching instruction removes the most AI-ness?

Each C_* round isolates ONE component of the R2 prompt (short / first-person / hedge /
personal-example). We score each against the frozen R0 classifier on held-out humans
(same leakage-proof protocol). The STYLE7 AUC drop vs R0 = that instruction's effect;
STYLO should barely move (none target it). Answers the original RQ: which prompt closes
the gap.

    python scripts/09_ablation_coaching.py
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

from src.evaluation.round_eval import run  # noqa: E402

ROUND_ORDER = ["R0", "C_short", "C_firstperson", "C_hedge", "C_personal", "R2"]
LABEL = {"R0": "R0\n(naive)", "C_short": "short", "C_firstperson": "1st-person",
         "C_hedge": "hedge", "C_personal": "personal\nexample", "R2": "R2\n(all)"}

_, _, summary = run(rounds_list=ROUND_ORDER, n_repeats=20)
piv = summary.pivot(index="round", columns="judge", values="auc_mean").reindex(ROUND_ORDER)
piv["STYLE7_dAUC_vs_R0"] = (piv["STYLE7"] - piv.loc["R0", "STYLE7"]).round(3)
print("Coaching-condition ablation — held-out AUC (frozen R0 classifier):")
print(piv.round(3).to_string())
piv.round(3).to_csv(ROOT / "results/tables/ablation_coaching.csv")

# figure: single conditions only (exclude R0/R2 from bars, show them as reference lines)
conds = ["C_short", "C_firstperson", "C_hedge", "C_personal"]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
vals = [piv.loc[c, "STYLE7"] for c in conds]
ax.bar([LABEL[c] for c in conds], vals, color="#c0392b", width=0.6)
ax.axhline(piv.loc["R0", "STYLE7"], ls="--", c="#777", lw=1.2)
ax.text(3.4, piv.loc["R0", "STYLE7"] - 0.05, "R0 naive (0.99)", color="#777", fontsize=9, ha="right")
ax.axhline(piv.loc["R2", "STYLE7"], ls="--", c="#1b7837", lw=1.2)
ax.text(3.4, piv.loc["R2", "STYLE7"] + 0.02, "R2 all combined", color="#1b7837", fontsize=9, ha="right")
ax.axhline(0.5, ls=":", c="gray", lw=1)
ax.set(ylim=(0, 1.05), ylabel="STYLE7 held-out AUC",
       title="Which single coaching instruction lowers detectability most?")
fig.tight_layout()
fig.savefig(ROOT / "results/figures/fig11_ablation_coaching.png", dpi=140)
print("saved -> results/figures/fig11_ablation_coaching.png  &  tables/ablation_coaching.csv")
