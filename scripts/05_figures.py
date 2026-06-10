"""Generate all presentation/report figures + a results summary, into results/.

Reproducible: reads the analysis tables (round_eval / topic_confound / negative_control
CSVs) and recomputes the lighter pieces (feature means, LOF %, PCA, data stats) from the
master + R0/R1/R2 JSONs. All figure text is English (font-safe).

    python scripts/05_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.analysis.outlier_lof import score_humans  # noqa: E402
from src.data_io import load_config, load_master, select_human_set  # noqa: E402
from src.features.style7 import STYLE7_NAMES, style7_matrix  # noqa: E402
from src.features.stylo import Stylometry  # noqa: E402

cfg = load_config()
FIG = ROOT / cfg["paths"]["figures_dir"]
TAB = ROOT / cfg["paths"]["tables_dir"]
FIG.mkdir(parents=True, exist_ok=True)
ROUNDS = ["R0", "R1", "R2"]

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_HUMAN, C_S7, C_STYLO = "#2c6fbb", "#c0392b", "#1b7837"
C_ROUND = {"R0": "#9ecae1", "R1": "#fb9a4f", "R2": "#b2182b"}


def save(fig, name):
    fig.savefig(FIG / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved ->", (FIG / name).relative_to(ROOT))


# ── load shared data (primary analysis set = config.human_set) ─────
HSET = cfg.get("human_set", "en_with_text")
hum = select_human_set(load_master(), HSET)
keep = set(hum["id"])
rtext = {r: {d["id"]: d["text"] for d in json.load(open(ROOT / "data/llm_answers" / f"{r}.json", encoding="utf-8"))}
         for r in ROUNDS}


# ── Fig 1: HEADLINE — AUC trajectory STYLE7 vs STYLO ───────────────
summ = pd.read_csv(TAB / "round_eval_summary.csv")
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for judge, c in [("STYLE7", C_S7), ("STYLO", C_STYLO)]:
    s = summ[summ["judge"] == judge].set_index("round").reindex(ROUNDS)
    ax.errorbar(ROUNDS, s["auc_mean"], yerr=s["auc_std_across_q"], marker="o", ms=8,
                lw=2.5, capsize=4, color=c, label=judge)
ax.axhline(0.5, ls="--", c="gray", lw=1)
ax.text(2.02, 0.52, "chance", color="gray", fontsize=9)
s7 = summ[summ.judge == "STYLE7"].set_index("round").reindex(ROUNDS)["auc_mean"]
st = summ[summ.judge == "STYLO"].set_index("round").reindex(ROUNDS)["auc_mean"]
ax.annotate("surface coached away\n(even inverted)", (2, s7["R2"]), (1.15, 0.18),
            color=C_S7, fontsize=9, ha="center", arrowprops=dict(arrowstyle="->", color=C_S7))
ax.annotate("signature persists", (2, st["R2"]), (1.0, 0.83),
            color=C_STYLO, fontsize=9, ha="center", arrowprops=dict(arrowstyle="->", color=C_STYLO))
ax.set(ylabel="held-out AUC (frozen R0 classifier)", ylim=(0, 1.03),
       title="Human vs LLM detectability across style-coaching rounds")
ax.set_xticks(range(3))
ax.set_xticklabels(["R0\n(naive)", "R1\n(light)", "R2\n(aggressive)"])
ax.legend(title="judge", loc="center left")
save(fig, "fig1_auc_trajectory.png")


# ── Fig 2: STYLE7 feature trajectory (overshoot) ───────────────────
mh = style7_matrix(hum["text"].tolist()).mean(0)
M = {r: style7_matrix([rtext[r][i] for i in hum["id"]]).mean(0) for r in ROUNDS}
fig, axes = plt.subplots(2, 4, figsize=(13, 6))
for k, name in enumerate(STYLE7_NAMES):
    ax = axes.flat[k]
    vals = [mh[k], M["R0"][k], M["R1"][k], M["R2"][k]]
    bars = ax.bar(["Human", "R0", "R1", "R2"],
                  vals, color=[C_HUMAN, C_ROUND["R0"], C_ROUND["R1"], C_ROUND["R2"]])
    ax.axhline(mh[k], ls="--", c=C_HUMAN, lw=1, alpha=0.7)
    ax.set_title(name, fontsize=10)
    ax.tick_params(labelsize=8)
axes.flat[-1].axis("off")
axes.flat[-1].text(0.5, 0.5, "dashed line = Human\nR2 crosses it on\nevery feature\n(overshoot)",
                   ha="center", va="center", fontsize=10)
fig.suptitle("STYLE7 surface features: R0 (LLM-like) → R2 overshoots Human", y=1.0)
fig.tight_layout()
save(fig, "fig2_style7_feature_trajectory.png")


# ── Fig 3: two-technique convergence (classifier vs LOF) ───────────
lof_frac = {r: float(score_humans(r)["below_llm_p5"].mean()) for r in ROUNDS}
fig, ax1 = plt.subplots(figsize=(6.2, 4.2))
ax1.plot(ROUNDS, st.values, "o-", lw=2.5, ms=8, color=C_STYLO, label="Classifier: STYLO AUC")
ax1.set_ylabel("STYLO AUC (Technique ①)", color=C_STYLO)
ax1.tick_params(axis="y", labelcolor=C_STYLO)
ax1.set_ylim(0.4, 1.02)
ax1.axhline(0.5, ls="--", c="gray", lw=1)
ax2 = ax1.twinx()
ax2.plot(ROUNDS, [lof_frac[r] * 100 for r in ROUNDS], "s--", lw=2.5, ms=8,
         color="#6a51a3", label="LOF: humans outside LLM cloud")
ax2.set_ylabel("% humans outside LLM cloud (Technique ②)", color="#6a51a3")
ax2.tick_params(axis="y", labelcolor="#6a51a3")
ax2.set_ylim(40, 102)
ax2.grid(False)
ax1.set_title("Two techniques agree: surface mimicry ≠ signature erased")
ax1.set_xticks(range(3))
ax1.set_xticklabels(["R0", "R1", "R2"])
save(fig, "fig3_two_technique_convergence.png")


# ── Fig 4: STYLO-space PCA for one question (R0 vs R2 vs Human) ────
Q = "Q5"
hq = hum[hum["question"] == Q]
ids = hq["id"].tolist()
th = hq["text"].tolist()
t0 = [rtext["R0"][i] for i in ids]
t2 = [rtext["R2"][i] for i in ids]
S = Stylometry().fit(th + t0 + t2)
X = StandardScaler().fit_transform(np.vstack([S.transform(th), S.transform(t0), S.transform(t2)]))
P = PCA(2, random_state=cfg["seed"]).fit_transform(X)
n = len(ids)
fig, ax = plt.subplots(figsize=(6.2, 5))
ax.scatter(P[:n, 0], P[:n, 1], c=C_HUMAN, s=40, label="Human", edgecolor="white", lw=0.5)
ax.scatter(P[n:2 * n, 0], P[n:2 * n, 1], c=C_ROUND["R0"], s=40, marker="^", label="LLM R0 (naive)")
ax.scatter(P[2 * n:, 0], P[2 * n:, 1], c=C_ROUND["R2"], s=40, marker="s", label="LLM R2 (mimic)")
ax.set(title=f"STYLO space ({Q}): even mimic R2 stays separable from humans",
       xlabel="PC1", ylabel="PC2")
ax.legend()
save(fig, "fig4_stylo_pca_Q5.png")


# ── Fig 5: topic-confound justification (why per-question) ─────────
tc = pd.read_csv(TAB / "topic_confound.csv")
fig, ax = plt.subplots(figsize=(5.6, 4))
ax.bar(["length", "STYLE7", "STYLO"], tc.set_index("feature_set").loc[
    ["length_only(word_count)", "STYLE7", "STYLO"], "macro_auc_ovr"],
    color=["#bdbdbd", C_S7, C_STYLO])
ax.axhline(0.5, ls="--", c="gray", lw=1)
ax.text(2.1, 0.52, "chance", color="gray", fontsize=9)
ax.set(ylim=(0, 1.05), ylabel="macro AUC (predict the question)",
       title="Style features encode TOPIC → evaluate per question")
save(fig, "fig5_topic_confound.png")


# ── Fig 6: data overview (counts / language / quality) ─────────────
allm = load_master(usable_only=False)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
cnt = allm.groupby("question").size().reindex(cfg["questions"])
axes[0].bar(cnt.index, cnt.values, color="#4a78a8")
axes[0].set_title("Responses per question")
axes[0].tick_params(axis="x", rotation=45)
lang = allm.groupby(["question", "language"]).size().unstack(fill_value=0).reindex(cfg["questions"])
lang.plot(kind="bar", stacked=True, ax=axes[1], colormap="Set2", legend=True)
axes[1].set_title("Language composition")
axes[1].tick_params(axis="x", rotation=45)
axes[1].set_xlabel("")
qual = allm.groupby(["question", "quality"]).size().unstack(fill_value=0).reindex(cfg["questions"])[["high", "medium", "low"]]
qual.plot(kind="bar", stacked=True, ax=axes[2], color=["#1a9850", "#fee08b", "#d73027"])
axes[2].set_title("Quality composition")
axes[2].tick_params(axis="x", rotation=45)
axes[2].set_xlabel("")
fig.suptitle("Dataset overview (5 topics · 9 questions · 542 question-level responses)", y=1.02)
fig.tight_layout()
save(fig, "fig6_data_overview.png")


# ── Fig 7: negative control (human vs human ≈ 0.5) ─────────────────
nc = pd.read_csv(TAB / "negative_control.csv")
fig, ax = plt.subplots(figsize=(6.4, 4))
x = np.arange(len(nc))
ax.bar(x - 0.2, nc["hvh_style7"], 0.4, label="STYLE7", color=C_S7)
ax.bar(x + 0.2, nc["hvh_stylo"], 0.4, label="STYLO", color=C_STYLO)
ax.axhline(0.5, ls="--", c="gray", lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels(nc["question"])
ax.set(ylim=(0, 1.0), ylabel="AUC", title="Negative control: human vs human ≈ 0.5 (no false signal)")
ax.legend()
save(fig, "fig7_negative_control.png")


# ── Fig 8: robustness across human-set definitions ─────────────────
rb_path = TAB / "robustness_human_set.csv"
if rb_path.exists():
    rb = pd.read_csv(rb_path)
    fig, ax = plt.subplots(figsize=(6.8, 4))
    x = np.arange(len(rb))
    w = 0.36
    ax.bar(x - w / 2, rb["STYLE7_R2"], w, label="STYLE7 @ R2 (broken)", color=C_S7)
    ax.bar(x + w / 2, rb["STYLO_R2"], w, label="STYLO @ R2 (holds)", color=C_STYLO)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(rb["human_set"], rb["n"])], fontsize=8)
    ax.set(ylim=(0, 1.0), ylabel="held-out AUC at R2",
           title="Robustness: conclusion holds across data definitions")
    ax.legend()
    save(fig, "fig8_robustness.png")


# ── Results summary (markdown) for the report ──────────────────────
piv = summ.pivot(index="round", columns="judge", values="auc_mean").reindex(ROUNDS).round(3)
feat_tbl = pd.DataFrame({"Human": mh, **{r: M[r] for r in ROUNDS}}, index=STYLE7_NAMES).round(3)
lines = ["# Results summary\n",
         "## AUC trajectory (frozen R0 classifier, held-out 30% humans)\n",
         piv.to_markdown(), "",
         "## LOF — % humans outside the LLM STYLO cloud\n",
         pd.Series({r: f"{lof_frac[r]*100:.0f}%" for r in ROUNDS}, name="outside").to_markdown(), "",
         "## STYLE7 feature means (Human vs rounds)\n",
         feat_tbl.to_markdown(), ""]
(TAB / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
print("saved ->", (TAB / "RESULTS_SUMMARY.md").relative_to(ROOT))
print("\nAll figures generated.")
