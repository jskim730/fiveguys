"""[Methodology validation] Does 'style' encode the TOPIC (question)?

The plan's key design choice is per-question evaluation (NEVER pool across questions),
because style features can predict WHICH QUESTION a response answers — i.e. the features
carry topic, not just authorship. This script tests that on the human data alone:

    Task: predict the question label (Q1..Q9) of a HUMAN response from style features only.
    High accuracy / macro-AUC  =>  style is topic-confounded  =>  pooling would leak topic
                                    =>  per-question human-vs-LLM evaluation is justified.

Feature sets compared: length-only (1 feat) < STYLE7 (7) < STYLO (~159).
Runs with zero external dependencies (no LLM, no embeddings).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:  # Windows consoles default to cp949; force UTF-8 so unicode prints don't crash.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.data_io import load_config, load_master, select_human_set  # noqa: E402
from src.features.style7 import style7_matrix  # noqa: E402
from src.features.stylo import assert_disjoint, stylo_matrix  # noqa: E402

cfg = load_config()
SEED = cfg.get("seed", 42)


def evaluate(X: np.ndarray, y: np.ndarray, name: str) -> dict:
    """5-fold multiclass (Q1..Q9) eval: accuracy + macro one-vs-rest AUC."""
    classes = np.unique(y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0),
    )
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=-1)
    preds = classes[proba.argmax(axis=1)]
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
    return {
        "feature_set": name,
        "n_features": X.shape[1],
        "accuracy": round(acc, 3),
        "macro_auc_ovr": round(auc, 3),
    }


def main() -> None:
    df = select_human_set(load_master(), cfg.get("human_set", "en_with_text"))
    texts = df["text"].tolist()
    y = df["question"].to_numpy()
    print(f"Human responses used: {len(df)}  | text_field={df['text_field'].iloc[0]}")
    print("Per-question counts:")
    print(df.groupby("question").size().to_string(), "\n")

    assert_disjoint()
    X_style7 = style7_matrix(texts)
    X_stylo = stylo_matrix(texts)
    X_len = X_style7[:, [0]]  # n_words only

    rows = [
        evaluate(X_len, y, "length_only(word_count)"),
        evaluate(X_style7, y, "STYLE7"),
        evaluate(X_stylo, y, "STYLO"),
    ]
    res = pd.DataFrame(rows)
    n_classes = df["question"].nunique()
    res["chance_acc"] = round(1 / n_classes, 3)
    res["chance_auc"] = 0.5

    print("=" * 64)
    print("TOPIC-CONFOUND CHECK — predict the QUESTION from style (humans only)")
    print("=" * 64)
    print(res.to_string(index=False))
    print(f"\nChance: accuracy={1/n_classes:.3f} (1/{n_classes}), AUC=0.5")

    interp = (
        "STYLO >> STYLE7 >> chance  =>  style features DO carry topic.\n"
        "  => Pooling questions would let a human-vs-LLM classifier cheat on topic.\n"
        "  => Per-question evaluation (plan 3-B/c) is justified."
    )
    print("\n" + interp)

    # persist
    tables = ROOT / cfg["paths"]["tables_dir"]
    tables.mkdir(parents=True, exist_ok=True)
    out_csv = tables / "topic_confound.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nsaved -> {out_csv.relative_to(ROOT)}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(res["feature_set"], res["macro_auc_ovr"], color="#3b7a8a")
        ax.axhline(0.5, ls="--", c="gray", label="chance (0.5)")
        ax.set_ylabel("macro OVR AUC")
        ax.set_title("Can style predict the question? (humans only)")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=15)
        ax.legend()
        fig.tight_layout()
        figs = ROOT / cfg["paths"]["figures_dir"]
        figs.mkdir(parents=True, exist_ok=True)
        out_png = figs / "topic_confound_auc.png"
        fig.savefig(out_png, dpi=130)
        print(f"saved -> {out_png.relative_to(ROOT)}")
    except Exception as e:  # plotting is optional
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
