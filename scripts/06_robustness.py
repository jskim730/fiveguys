"""Robustness of the headline result to the human-set definition.

Re-runs the leakage-proof round trajectory (frozen R0 classifier, held-out 30%) for several
human-set definitions and prints STYLE7 / STYLO AUC at R0 and R2. The question:

    Does "STYLE7 breaks, STYLO holds" survive when we remove the translation confound
    (English-original only) and vary the quality filter?

No regeneration needed — LLM answers exist for the 510 superset; every set is a subset.

    python scripts/06_robustness.py
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

from src.data_io import load_master, select_human_set  # noqa: E402
from src.evaluation.round_eval import run  # noqa: E402

SETS = ["non_low", "en_non_low", "en_with_text", "with_text"]
rows = []
for hs in SETS:
    n = len(select_human_set(load_master(), hs))
    _, _, summary = run(human_set=hs, n_repeats=20)
    piv = summary.pivot(index="round", columns="judge", values="auc_mean")
    rows.append({
        "human_set": hs, "n": n,
        "STYLE7_R0": round(piv.loc["R0", "STYLE7"], 3),
        "STYLE7_R2": round(piv.loc["R2", "STYLE7"], 3),
        "STYLO_R0": round(piv.loc["R0", "STYLO"], 3),
        "STYLO_R2": round(piv.loc["R2", "STYLO"], 3),
    })
    print(f"  done: {hs:14} (n={n})")

res = pd.DataFrame(rows)
print("\n── Headline robustness to human-set choice ──")
print(res.to_string(index=False))
print("\nRead: STYLE7_R2 ≈ 0 (surface fully broken) and STYLO_R2 well above 0.5 (signature holds)")
print("      across ALL sets  =>  conclusion is not an artifact of the translation confound.")
res.to_csv(ROOT / "results/tables/robustness_human_set.csv", index=False)
print(f"\nsaved -> results/tables/robustness_human_set.csv")
