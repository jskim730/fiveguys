"""[Technique ②] LOF in STYLO space — human answers farthest from the LLM distribution.

Per question (STYLO is topic-loaded, so control topic): build the STYLO space on
human+LLM texts, fit Local Outlier Factor (novelty mode) on the LLM 'manifold', then
score the human answers. The most-outlying humans = candidate 'human-only, hard-to-forge'
style signatures — the thing the generator never covers. Distance-based, so it is NOT a
re-run of the classifier (Technique ①), which keeps the two techniques methodologically
distinct (course requirement).

Run after a round is generated:
    python src/analysis/outlier_lof.py --round R0 --topk 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_io import load_config, load_master, select_human_set  # noqa: E402
from src.features.stylo import Stylometry, assert_disjoint  # noqa: E402

cfg = load_config()
NON_TEXT = {"missing", "unclear", ""}


def _load(round_name, human_set):
    hum = select_human_set(load_master(), human_set)
    keep = set(hum["id"])
    path = ROOT / cfg["paths"]["llm_answers_dir"] / f"{round_name}.json"
    llm = pd.DataFrame(json.load(open(path, encoding="utf-8")))
    llm = llm[llm["id"].isin(keep)]
    return hum, llm


def score_humans(round_name, human_set=None, n_neighbors=None):
    """Per question: LOF(novelty) on LLM STYLO; score humans (lower = more outlier)."""
    human_set = human_set or cfg.get("human_set", "non_low")
    n_neighbors = n_neighbors or cfg["outlier"]["n_neighbors"]
    hum, llm = _load(round_name, human_set)

    rows = []
    for q in cfg["questions"]:
        th = hum[hum["question"] == q]
        tl = llm[llm["q"] == q]
        if len(th) < 10 or len(tl) < 15:
            continue
        S = Stylometry().fit(th["text"].tolist() + tl["text"].tolist())
        Xh = S.transform(th["text"].tolist())
        Xl = S.transform(tl["text"].tolist())
        scaler = StandardScaler().fit(np.vstack([Xh, Xl]))
        Xh, Xl = scaler.transform(Xh), scaler.transform(Xl)

        k = min(n_neighbors, len(Xl) - 1)
        lof = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Xl)
        sh = lof.score_samples(Xh)          # lower = farther from the LLM manifold
        # LLM self-baseline (novelty=False on the LLM cloud) for a reference scale
        sl = LocalOutlierFactor(n_neighbors=k).fit(Xl).negative_outlier_factor_

        thr = np.percentile(sl, 5)          # 5th pct of LLM self-scores
        for (_, r), s in zip(th.iterrows(), sh):
            rows.append({"id": r["id"], "q": q, "lof_score": round(float(s), 3),
                         "below_llm_p5": bool(s < thr),
                         "keywords": r["keywords"], "text": r["text"]})
    df = pd.DataFrame(rows).sort_values("lof_score")  # most outlier first
    return df


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="R0")
    ap.add_argument("--human-set", default=None)
    ap.add_argument("--topk", type=int, default=15)
    a = ap.parse_args()
    assert_disjoint()

    df = score_humans(a.round, a.human_set)
    frac = df["below_llm_p5"].mean()
    print(f"{a.round}: scored {len(df)} human answers in STYLO space "
          f"(per question, vs LLM manifold).")
    print(f"{frac:.0%} of humans fall below the LLM 5th-percentile self-score "
          f"= clearly outside the LLM style cloud.\n")
    print(f"── Top {a.topk} 'human-only' style outliers (farthest from LLM) ──")
    for _, r in df.head(a.topk).iterrows():
        print(f"[{r['id']} | {r['q']} | LOF={r['lof_score']}] kw={r['keywords']}")
        print(f"   {r['text'][:160]}")

    out = ROOT / cfg["paths"]["tables_dir"] / f"outlier_lof_{a.round}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out.relative_to(ROOT)}")
