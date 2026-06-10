"""Generate LLM answers for a round (R0/R1/R2).

Content is FIXED by the paired human's keywords; only the STYLE instruction changes
across rounds. Output schema: {id, q, round, text, keywords}, with id = paired human id.
Generator = ELICE gpt-5-mini (differs from Claude preprocessing -> generator != judge).

Preview (no save):
    python src/generation/generate.py --round R0 --question Q5 --limit 5
Full round (save -> data/llm_answers/<round>.json):
    python src/generation/generate.py --round R0 --save
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_io import load_config, load_master  # noqa: E402
from src.generation.elice_client import elice_chat  # noqa: E402

cfg = load_config()
PERSONA = cfg["generation"]["persona"]

# Only this string changes across rounds (content stays fixed by keywords).
STYLE = {
    "R0": "",  # naive — zero style coaching (GPT's natural voice = baseline)
    "R1": "Write like an undergraduate quickly noting a thought: keep it fairly short "
          "(a few sentences), use first person, plain everyday words, and don't over-explain "
          "or define terms.",
    # R2 targets the MEASURED STYLE7 tells in the HUMAN direction (short, short sentences,
    # plain/short words, few commas, first person, more hedging). STYLO (function words,
    # char n-grams, punctuation rhythm) is deliberately left untargeted.
    "R2": "Write like a student jotting a very quick discussion note: very short, just 1-3 "
          "sentences (around 40 words). Use first person (I, my). Use short plain words and "
          "short sentences, with few commas. It is fine to sound tentative (e.g. 'I think', "
          "'maybe') and to leave a fragment. Do NOT define terms or try to be comprehensive — "
          "give just one point or one concrete personal example.",
    # ── Ⓒ coaching-condition ablation: each isolates ONE component of R2 ──
    "C_short": "Keep it very brief: 1 to 3 sentences, about 40 words.",
    "C_firstperson": "Write in the first person (use I, me, my).",
    "C_hedge": "Sound tentative and unsure: use phrases like 'I think', 'maybe', 'probably'.",
    "C_personal": "Do not define any terms; instead give one concrete personal example, and "
                  "sentence fragments are fine.",
}

NON_TEXT = {"missing", "unclear", ""}


def build_prompt(keywords, round_name: str) -> str:
    kw = ", ".join(keywords)
    lines = [
        f"You are {PERSONA}.",
        f"Write ONE answer to an in-class data-science discussion. Cover these points: {kw}.",
    ]
    if STYLE[round_name]:
        lines.append(STYLE[round_name])
    lines.append("Return ONLY the answer text — no preamble, no quotation marks, no lists.")
    return "\n".join(lines)


def _clean(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1].strip()
    return text


def generate_one(row: dict, round_name: str) -> dict:
    prompt = build_prompt(list(row["keywords"]), round_name)
    text = _clean(elice_chat(prompt))
    return {"id": row["id"], "q": row["question"], "round": round_name,
            "text": text, "keywords": list(row["keywords"])}


def select_humans(question=None, limit=None, human_set=None) -> list[dict]:
    """Humans to generate for. Default = 510-superset; human_set restricts (e.g. en_with_text)."""
    if human_set:
        from src.data_io import select_human_set
        df = select_human_set(load_master(), human_set).sort_values("id")
    else:
        df = load_master(usable_only=False)
        df = df[~df["text"].str.strip().str.lower().isin(NON_TEXT)].sort_values("id")
    if question:
        df = df[df["question"] == question]
    if limit:
        df = df.head(limit)
    return df.to_dict("records")


def generate_round(round_name: str, question=None, limit=None, workers: int = 4, human_set=None):
    rows = select_humans(question, limit, human_set)
    out: list = [None] * len(rows)

    def work(i):
        try:
            return i, generate_one(rows[i], round_name)
        except Exception as e:  # keep going; record the failure
            return i, {"id": rows[i]["id"], "error": str(e)}

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(work, i) for i in range(len(rows))]):
            i, rec = fut.result()
            out[i] = rec
            done += 1
            print(f"  [{done}/{len(rows)}] {rec['id']}", file=sys.stderr)
    return rows, out


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True, choices=sorted(STYLE))
    ap.add_argument("--question")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--human-set", default=None)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    rows, out = generate_round(a.round, a.question, a.limit, a.workers, a.human_set)

    def wc(t):
        return len(str(t).split())

    ok = [r for r in out if r and "error" not in r]
    errs = [r for r in out if r and "error" in r]
    preview = list(zip(rows, out))
    preview = preview[:3] if len(rows) > 20 else preview

    print("\n" + "=" * 72)
    for ri, ro in preview:
        if ro and "error" not in ro:
            print(f"[{ro['id']}]  kw={ro['keywords']}")
            print(f"  HUMAN ({wc(ri['text'])}w): {ri['text'][:240]}")
            print(f"  {a.round}    ({wc(ro['text'])}w): {ro['text'][:240]}")
        else:
            print(f"[{ro['id']}]  ERROR: {ro.get('error')}")
        print("-" * 72)

    if len(rows) > 20:
        import statistics
        hw = statistics.mean(wc(r["text"]) for r in rows)
        lw = statistics.mean(wc(r["text"]) for r in ok) if ok else 0
        print(f"summary: {len(ok)} ok, {len(errs)} errors | "
              f"mean words: human={hw:.0f}  {a.round}={lw:.0f}")

    if a.save:
        recs = [r for r in out if r and "error" not in r]
        errs = [r for r in out if r and "error" in r]
        path = ROOT / cfg["paths"]["llm_answers_dir"] / f"{a.round}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        print(f"\nsaved {len(recs)} answers -> {path.relative_to(ROOT)}  ({len(errs)} errors)")
