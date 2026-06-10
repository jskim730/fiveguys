"""[step 5] Round comparison under a leakage-proof protocol.

Protocol (per question, repeated over random splits to stabilize small-n noise):
  - Split humans 70/30. Train a FROZEN classifier on  human-70% (0) + R0-70% (1).
  - Evaluate that frozen classifier on the unseen human-30% (0) vs each round's answers
    for those same held-out humans (1): R0-held / R1 / R2.
  - Do this for two judges: STYLE7 (the coached target) and STYLO (independent).

Read the AUC trajectory R0 -> R1 -> R2:
  - STYLE7 AUC dropping  = surface tells were successfully coached away.
  - STYLO AUC holding    = the unconscious signature is NOT coached away => real human
    signal persists ("surface mimicry, not real humanization").

Run after R0/R1/R2 are generated:
    python src/evaluation/round_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.classify import Style7F, StyloF  # noqa: E402  (reuse featurizers)
from src.data_io import load_config, load_master, select_human_set  # noqa: E402
from src.features.stylo import assert_disjoint  # noqa: E402

cfg = load_config()
SEED = cfg.get("seed", 42)
QUESTIONS = cfg["questions"]
ROUNDS = ["R0", "R1", "R2"]
JUDGES = {"STYLE7": Style7F, "STYLO": StyloF}


def _round_map(round_name):
    path = ROOT / cfg["paths"]["llm_answers_dir"] / f"{round_name}.json"
    data = json.load(open(path, encoding="utf-8"))
    return {d["id"]: d["text"] for d in data}


def fit_frozen(judge_cls, train_texts, y):
    feat = judge_cls().fit(train_texts)
    scaler = StandardScaler().fit(feat.transform(train_texts))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(feat.transform(train_texts)), y)
    return feat, scaler, clf


def _auc(feat, scaler, clf, texts, y):
    p = clf.predict_proba(scaler.transform(feat.transform(texts)))[:, 1]
    return roc_auc_score(y, p)


def run(human_set=None, n_repeats=20, holdout=None, judges=None, rounds_list=None):
    assert_disjoint()
    judges = judges or JUDGES
    rounds_eval = rounds_list or ROUNDS
    human_set = human_set or cfg.get("human_set", "non_low")
    holdout = holdout or cfg["evaluation"]["holdout_human_frac"]
    hum = select_human_set(load_master(), human_set)
    rounds = {r: _round_map(r) for r in dict.fromkeys(["R0", *rounds_eval])}
    rng = np.random.default_rng(SEED)

    recs = []  # (question, judge, round, auc)
    for q in QUESTIONS:
        hq = hum[hum["question"] == q]
        ids = hq["id"].tolist()
        text_of = dict(zip(hq["id"], hq["text"]))
        if len(ids) < 20:
            continue
        n_hold = max(4, int(round(len(ids) * holdout)))
        for _ in range(n_repeats):
            perm = rng.permutation(ids)
            hold, train = list(perm[:n_hold]), list(perm[n_hold:])
            # frozen classifier: human-train (0) + R0-train (1)
            tr_texts = [text_of[i] for i in train] + [rounds["R0"][i] for i in train]
            y_tr = np.array([0] * len(train) + [1] * len(train))
            for jname, jcls in judges.items():
                feat, scaler, clf = fit_frozen(jcls, tr_texts, y_tr)
                hold_h = [text_of[i] for i in hold]
                for rnd in rounds_eval:
                    te_texts = hold_h + [rounds[rnd][i] for i in hold]
                    y_te = np.array([0] * len(hold) + [1] * len(hold))
                    recs.append((q, jname, rnd, _auc(feat, scaler, clf, te_texts, y_te)))

    df = pd.DataFrame(recs, columns=["question", "judge", "round", "auc"])
    # per-question mean over repeats, then macro-mean over questions
    pq = df.groupby(["judge", "round", "question"], as_index=False)["auc"].mean()
    summary = pq.groupby(["judge", "round"], as_index=False)["auc"].agg(["mean", "std"])
    summary.columns = ["judge", "round", "auc_mean", "auc_std_across_q"]
    return df, pq, summary


def _plot(summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: {e})")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for judge, color in [("STYLE7", "#c0392b"), ("STYLO", "#2c6fbb")]:
        s = summary[summary["judge"] == judge].set_index("round").reindex(ROUNDS)
        ax.errorbar(ROUNDS, s["auc_mean"], yerr=s["auc_std_across_q"], marker="o",
                    capsize=4, label=judge, color=color)
    ax.axhline(0.5, ls="--", c="gray", lw=1, label="chance")
    ax.set_ylabel("held-out AUC (frozen R0 classifier)")
    ax.set_title("Round trajectory: STYLE7 (coached) vs STYLO (independent)")
    ax.set_ylim(0.4, 1.02)
    ax.legend()
    fig.tight_layout()
    out = ROOT / cfg["paths"]["figures_dir"] / "round_trajectory.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    df, pq, summary = run()
    print("\n── AUC trajectory (frozen R0 classifier, held-out 30% humans, per-question macro-mean) ──")
    piv = summary.pivot(index="round", columns="judge", values="auc_mean").reindex(ROUNDS)
    print(piv.round(3).to_string())
    print("\nΔAUC vs R0 (negative = coaching reduced detectability):")
    for j in ["STYLE7", "STYLO"]:
        base = piv.loc["R0", j]
        print(f"  {j}: " + "  ".join(f"{r}={piv.loc[r, j]-base:+.3f}" for r in ROUNDS))

    tdir = ROOT / cfg["paths"]["tables_dir"]
    tdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tdir / "round_eval_summary.csv", index=False)
    pq.to_csv(tdir / "round_eval_per_question.csv", index=False)
    print(f"saved -> {(tdir / 'round_eval_summary.csv').relative_to(ROOT)}")
    _plot(summary)
