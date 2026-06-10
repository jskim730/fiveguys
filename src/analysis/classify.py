"""[Technique ①] Human(0) vs LLM(1) STYLE classifier — the core metric.

Per-question (NO pooling — topic confound). For each question: 5-fold CV AUC with logistic
regression and GBM on three feature sets — length-only (n_words), STYLE7 (the targeted
'tells'), and STYLO (independent judge). Plus bootstrap 95% CI, a label-shuffle permutation
null, a human-vs-human negative control (must give AUC≈0.5), and STYLE7 logistic coefficients
(which feature separates the classes).

STYLO is fit per TRAIN fold (its char-ngram vocabulary) to avoid leakage.

Smoke test (no R0 needed — validates the machine via the negative control):
    python src/analysis/classify.py --smoke
Full R0 analysis (after generation):
    python src/analysis/classify.py --round R0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_io import load_config, load_master, select_human_set  # noqa: E402
from src.features.style7 import STYLE7_NAMES, style7, style7_matrix  # noqa: E402
from src.features.stylo import Stylometry, assert_disjoint  # noqa: E402

cfg = load_config()
SEED = cfg.get("seed", 42)
QUESTIONS = cfg["questions"]


# ── featurizers: fit(train_texts) -> self ; transform(texts) -> X ──────────
class Style7F:
    def fit(self, texts):
        return self

    def transform(self, texts):
        return style7_matrix(texts)


class LengthF:
    def fit(self, texts):
        return self

    def transform(self, texts):
        return style7_matrix(texts)[:, [0]]  # n_words only


class StyloF:
    def __init__(self):
        self.s = Stylometry()

    def fit(self, texts):
        self.s.fit(texts)
        return self

    def transform(self, texts):
        return self.s.transform(texts)


FEATS = {"length": LengthF, "style7": Style7F, "stylo": StyloF}


def make_clf(name):
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if name == "gbm":
        return HistGradientBoostingClassifier(random_state=SEED)
    raise ValueError(name)


# ── out-of-fold CV scores (STYLO fit per train fold) ───────────────────────
def cv_scores(texts, y, feat_name, clf_name, seed=SEED, n_splits=5) -> np.ndarray:
    texts = list(texts)
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(texts, y):
        feat = FEATS[feat_name]().fit([texts[i] for i in tr])
        Xtr = feat.transform([texts[i] for i in tr])
        Xte = feat.transform([texts[i] for i in te])
        sc = StandardScaler().fit(Xtr)
        clf = make_clf(clf_name).fit(sc.transform(Xtr), y[tr])
        oof[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
    return oof


def bootstrap_ci(y, oof, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    oof = np.asarray(oof)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], oof[b]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def permutation_p(texts, y, feat_name, clf_name, observed, n=300, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    count = 0
    for _ in range(n):
        yp = rng.permutation(y)
        auc = roc_auc_score(yp, cv_scores(texts, yp, feat_name, clf_name, seed=seed))
        if auc >= observed:
            count += 1
    return (count + 1) / (n + 1)


# ── data assembly: per-question human(0) vs paired R0/LLM(1) ───────────────
def load_pairs(round_name, human_set=None):
    """Return {question: (texts, y)} with human(0) + 1:1-paired LLM(1)."""
    human_set = human_set or cfg.get("human_set", "non_low")
    hum = select_human_set(load_master(), human_set)
    keep_ids = set(hum["id"])

    path = ROOT / cfg["paths"]["llm_answers_dir"] / f"{round_name}.json"
    llm = pd.DataFrame(json.load(open(path, encoding="utf-8")))
    llm = llm[llm["id"].isin(keep_ids)]  # 1:1 with the analysis human set

    out = {}
    for q in QUESTIONS:
        ht = hum[hum["question"] == q]["text"].tolist()
        lt = llm[llm["q"] == q]["text"].tolist()
        if len(ht) >= 10 and len(lt) >= 10:
            texts = ht + lt
            y = np.array([0] * len(ht) + [1] * len(lt))
            out[q] = (texts, y, len(ht), len(lt))
    return out


# ── negative control: humans split into two random groups -> AUC≈0.5 ───────
def negative_control(human_set=None, feat_names=("style7", "stylo")):
    human_set = human_set or cfg.get("human_set", "non_low")
    hum = select_human_set(load_master(), human_set)
    rng = np.random.default_rng(SEED)
    rows = []
    for q in QUESTIONS:
        texts = hum[hum["question"] == q]["text"].tolist()
        if len(texts) < 20:
            continue
        y = rng.integers(0, 2, len(texts))  # random labels = no real signal
        row = {"question": q, "n": len(texts)}
        for f in feat_names:
            row[f"hvh_{f}"] = round(roc_auc_score(y, cv_scores(texts, y, f, "logreg")), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def style7_coefficients(pairs):
    """Mean standardized STYLE7 logistic coefficients across questions (+ = LLM-ward)."""
    coefs = []
    for q, (texts, y, _, _) in pairs.items():
        X = style7_matrix(texts)
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
        coefs.append(clf.coef_[0])
    mean = np.mean(coefs, axis=0)
    return pd.DataFrame({"feature": STYLE7_NAMES, "mean_coef(+=LLM)": np.round(mean, 3)}) \
        .sort_values("mean_coef(+=LLM)")


def run_full(round_name, human_set=None):
    assert_disjoint()
    pairs = load_pairs(round_name, human_set)
    rows = []
    for q, (texts, y, nh, nl) in pairs.items():
        row = {"question": q, "n_human": nh, "n_llm": nl}
        scores = {}
        for feat in ["length", "style7", "stylo"]:
            for clf in ["logreg", "gbm"]:
                oof = cv_scores(texts, y, feat, clf)
                auc = roc_auc_score(y, oof)
                row[f"{feat}_{clf}"] = round(auc, 3)
                scores[(feat, clf)] = oof
        # CI + permutation on the two key judges (logreg)
        lo, hi = bootstrap_ci(y, scores[("style7", "logreg")])
        row["style7_logreg_CI"] = f"[{lo:.2f},{hi:.2f}]"
        slo, shi = bootstrap_ci(y, scores[("stylo", "logreg")])
        row["stylo_logreg_CI"] = f"[{slo:.2f},{shi:.2f}]"
        row["style7_perm_p"] = round(
            permutation_p(texts, y, "style7", "logreg", row["style7_logreg"]), 3)
        rows.append(row)
    return pd.DataFrame(rows), pairs


def _save(df, name):
    out = ROOT / cfg["paths"]["tables_dir"] / name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="R0")
    ap.add_argument("--human-set", default=None, help="non_low (default) | with_text")
    ap.add_argument("--smoke", action="store_true", help="negative control only (no R0 needed)")
    a = ap.parse_args()
    assert_disjoint()
    print("STYLE7/STYLO disjoint: OK")

    if a.smoke:
        print("\n── Negative control: human vs human (random labels) — expect AUC≈0.5 ──")
        nc = negative_control(a.human_set)
        print(nc.to_string(index=False))
        print(f"\nmean hvh_style7={nc['hvh_style7'].mean():.3f}  "
              f"hvh_stylo={nc['hvh_stylo'].mean():.3f}  (both should be ~0.5)")
        _save(nc, "negative_control.csv")
        sys.exit(0)

    print(f"\n── {a.round}: per-question human(0) vs LLM(1) AUC ──")
    df, pairs = run_full(a.round, a.human_set)
    print(df.to_string(index=False))
    _save(df, f"classify_{a.round}.csv")
    print("\n── STYLE7 mean coefficients (+ = LLM-ward, − = human-ward) ──")
    coef = style7_coefficients(pairs)
    print(coef.to_string(index=False))
    _save(coef, f"style7_coef_{a.round}.csv")
