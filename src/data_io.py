"""Data loading & label-table construction.

Loads the preprocessed master dataset (data/preprocessed/Q*_corrected.json) and the
generated LLM rounds (data/llm_answers/R{0,1,2}.json), then builds the per-question
labeled tables (human=0 vs LLM=1) consumed by the analysis modules.

Text field comes from config.text_field (english_text by default; cleaned_text = multilingual).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"

# Sentinels written by the preprocessing prompt for empty / uninterpretable responses.
NON_TEXT = {"missing", "unclear", ""}


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Read config/config.yaml (single source of truth)."""
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_master(text_field: str | None = None, usable_only: bool = False) -> pd.DataFrame:
    """Load all Q*_corrected.json into one DataFrame.

    Adds a unified ``text`` column = the configured analysis text field.
    ``usable_only=True`` drops quality=='low' and missing/unclear text rows.
    """
    cfg = load_config()
    text_field = text_field or cfg.get("text_field", "english_text")

    frames = []
    for p in sorted(glob.glob(str(ROOT / cfg["paths"]["master_glob"]))):
        with open(p, encoding="utf-8") as f:
            frames.append(pd.DataFrame(json.load(f)))
    df = pd.concat(frames, ignore_index=True)

    df["text"] = df[text_field].astype(str)
    df["text_field"] = text_field
    df["source"] = "human"
    df["label"] = 0  # human=0 (LLM rounds will be label=1)

    if usable_only:
        df = df[df["quality"] != "low"]
        df = df[~df["text"].str.strip().str.lower().isin(NON_TEXT)]
        df = df.reset_index(drop=True)
    return df


def select_human_set(df: pd.DataFrame, human_set: str) -> pd.DataFrame:
    """Filter the human master to a named analysis set (and reset index).

      non_low      : quality != low (drops fragmented ESL too)          ~403
      with_text    : any usable english_text (keeps fragmented ESL)     ~510
      en_non_low   : English-original AND quality != low                ~362
      en_with_text : English-original AND usable text (keeps rough ESL) ~420   <- cleanest

    English-only sets remove the Claude-translation confound from STYLO (Korean/mixed
    responses' english_text is a machine translation, not the student's own English).
    """
    en = df["language"] == "en"
    nonlow = df["quality"] != "low"
    hastext = ~df["text"].str.strip().str.lower().isin(NON_TEXT)
    masks = {
        "non_low": nonlow & hastext,
        "with_text": hastext,
        "en_non_low": en & nonlow & hastext,
        "en_with_text": en & hastext,
    }
    if human_set not in masks:
        raise ValueError(f"unknown human_set: {human_set} (choose {list(masks)})")
    return df[masks[human_set]].reset_index(drop=True)


def load_round(round_name: str, text_field: str | None = None) -> pd.DataFrame:
    """Load a generated LLM round (R0/R1/R2) from data/llm_answers/<round>.json.

    Expected to share the human id/question schema; tagged source='llm', label=1.
    """
    cfg = load_config()
    text_field = text_field or cfg.get("text_field", "english_text")
    path = ROOT / cfg["paths"]["llm_answers_dir"] / f"{round_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Round file not generated yet: {path}")
    with open(path, encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))
    df["text"] = df.get(text_field, df.get("text")).astype(str)
    df["source"] = "llm"
    df["label"] = 1
    df["round"] = round_name
    return df


def build_labeled_table(question: str, round_name: str,
                        usable_only: bool = True) -> pd.DataFrame:
    """Human(0) vs LLM(1) rows for a single question - the classifier input unit.

    Per-question by design: NEVER pool across questions (topic confound, see plan 3-B/c).
    """
    human = load_master(usable_only=usable_only)
    human = human[human["question"] == question]
    llm = load_round(round_name)
    llm = llm[llm["question"] == question]
    cols = ["id", "question", "text", "source", "label"]
    return pd.concat([human[cols], llm[cols]], ignore_index=True)


if __name__ == "__main__":  # quick smoke test
    df = load_master(usable_only=True)
    print(f"master usable rows: {len(df)}  | questions: {sorted(df['question'].unique())}")
    print(df.groupby("question").size().to_string())
