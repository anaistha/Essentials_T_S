"""Score the model responses and write accuracy summaries.

The model's full response (reasoning included) is in the ``predicted`` column.
``done_reason`` only says why generation stopped ("stop"/"length") and never
contains an answer, so it must not be used for extraction.

Run from the repository root::

    python -m dialect_bench.eval            # update results/ in place
    python -m dialect_bench.eval --dry-run  # score without changing results/

Every TSV below ``results`` gets three extra columns:

``extracted_answer``
    The final number found in ``predicted`` (empty if there is none).
``correct``
    1 if it matches ``gt_answer``, 0 if not (a missing number counts as wrong),
    empty if ``gt_answer`` itself is missing.
``degenerate``
    1 if ``predicted`` is stuck in a repetition loop, otherwise 0. These are
    still scored, so a looping model shows up as wrong rather than disappearing.

A summary per model and thinking mode is printed and written to
``answer_summary.tsv``, and the model x dialect grid to ``accuracy_by_dialect.tsv``
(both next to where the script is run).
"""

from __future__ import annotations

import argparse
import math
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd

# A number as models write it, including Swiss/German grouping such as 1'430 or 7'425,00
NUMBER = re.compile(r"-?\d(?:[\d.,'’]*\d)?")
# Numbers that the model highlighted as the answer
BOLD = re.compile(r"\*\*([^*]+)\*\*|\\mathbf\{([^{}]*)\}")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
# Markdown/LaTeX lines that may legitimately repeat without meaning the model is stuck
PUNCTUATION_LINE = re.compile(r"^[\s\\${}\[\]|*_#>\-–—=+.,;:!?'\"()/]+$")


def parse_number(token: str) -> float | None:
    """Turn a number written by a model into a float.

    Handles Swiss/German formatting, where ' or . groups thousands and , is the
    decimal separator (e.g. 1'430, 7'425,00, 80.000, 2,5).
    """
    token = str(token).strip().replace("$", "").replace("%", "").rstrip(".,")
    if not token:
        return None
    negative = token[0] in "-−"
    token = token.lstrip("-−").replace("'", "").replace("’", "")

    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", token):
        token = token.replace(".", "")
    elif re.fullmatch(r"\d{1,3}(?:,\d{3})+", token):
        token = token.replace(",", "")
    elif "," in token and "." not in token:
        token = token.replace(",", ".")
    elif "." in token and "," in token:
        token = token.replace(",", "")

    try:
        value = float(token)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return -value if negative else value


def numbers_in(text: str) -> list[float]:
    values = (parse_number(token) for token in NUMBER.findall(str(text)))
    return [value for value in values if value is not None]


def extract_answer(text: str) -> float | None:
    """Return the answer the model settled on, or None if it wrote no number.

    The last highlighted number is used first (``**20 Dollar**``, ``\\boxed{20}``),
    because that is where models normally put their answer. Otherwise the last
    whole number in the response is used, ignoring trailing decimals or stray
    numbers such as a percentage in a closing remark.
    """
    text = str(text)
    if not text.strip():
        return None

    for a, b in reversed(BOLD.findall(text)):
        values = numbers_in(a or b)
        if values:
            return values[-1]
    for segment in reversed(BOXED.findall(text)):
        values = numbers_in(segment)
        if values:
            return values[-1]

    values = numbers_in(text)
    whole = [value for value in values if value.is_integer() and abs(value) < 1e15]
    if whole:
        return whole[-1]
    return values[-1] if values else None


def is_degenerate(text: str) -> bool:
    """True when a response is stuck repeating itself instead of finishing.

    Detects the loops small/local models fall into: the same token repeated
    consecutively, one token taking over most of a long response, or the same
    non-formatting line repeated many times.
    """
    text = str(text)
    tokens = text.split()
    if not tokens:
        return False

    run = best = 1
    for previous, token in zip(tokens, tokens[1:]):
        run = run + 1 if token == previous else 1
        best = max(best, run)
    if best >= 30:
        return True

    if len(tokens) >= 200 and max(Counter(tokens).values()) / len(tokens) >= 0.40:
        return True

    lines = [
        line.strip() for line in text.splitlines()
        if len(line.strip()) >= 20 and not PUNCTUATION_LINE.match(line.strip())
    ]
    return bool(lines) and max(Counter(lines).values()) >= 10


def process_file(path: Path, results_dir: Path, write: bool = True) -> dict | None:
    """Score one result file, optionally adding the columns to it in place."""
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        print(f"skipping empty file {path.relative_to(results_dir)}")
        return None
    except Exception as error:  # a run that is still writing can leave a broken file
        print(f"skipping unreadable file {path.relative_to(results_dir)}: {error}")
        return None

    answered = scored = correct = degenerate = 0
    if "predicted" in df.columns:
        answers = df["predicted"].map(extract_answer)
        df["extracted_answer"] = [
            int(value) if pd.notna(value) and float(value).is_integer() else value
            for value in answers
        ]
        df["degenerate"] = df["predicted"].map(lambda text: int(is_degenerate(text)))
        answered = int(answers.notna().sum())
        degenerate = int(df["degenerate"].sum())

        if "gt_answer" in df.columns:
            truth = df["gt_answer"].map(parse_number)
            df["correct"] = [
                # a response without a number counts as wrong, not as unscored
                "" if pd.isna(expected)
                else int(pd.notna(answer) and math.isclose(answer, expected, rel_tol=0, abs_tol=1e-9))
                for answer, expected in zip(answers, truth)
            ]
            scored = int(truth.notna().sum())
            correct = int(pd.to_numeric(df["correct"], errors="coerce").fillna(0).sum())
    else:
        print(f"warning: {path.relative_to(results_dir)} has no 'predicted' column, leaving it unchanged")

    if write:
        df.to_csv(path, sep="\t", index=False)

    parts = path.relative_to(results_dir).parts
    return {
        "model": parts[0] if len(parts) > 1 else "?",
        "thinking": parts[1] if len(parts) > 2 else "?",
        "dialect": path.stem,
        "file": path.name,
        "rows": len(df),
        "answered": answered,
        "scored": scored,
        "correct": correct,
        "degenerate": degenerate,
    }


def summarize(stats: list[dict], summary_path: Path) -> None:
    frame = pd.DataFrame(stats)
    if frame.empty:
        print("no result files found")
        return

    summary = (
        frame.groupby(["model", "thinking"], sort=True)
        .agg(files=("file", "count"), rows=("rows", "sum"), answered=("answered", "sum"),
             scored=("scored", "sum"), correct=("correct", "sum"), degenerate=("degenerate", "sum"))
        .reset_index()
    )
    summary["accuracy"] = (summary["correct"] / summary["scored"].where(summary["scored"] > 0)).round(4)
    summary["degenerate_pct"] = (summary["degenerate"] / summary["rows"] * 100).round(1)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, sep="\t", index=False)

    # Model x dialect accuracy grid
    by_dialect = (
        frame.groupby(["model", "thinking", "dialect"], sort=True)
        .agg(correct=("correct", "sum"), scored=("scored", "sum"))
        .reset_index()
    )
    by_dialect["accuracy"] = by_dialect["correct"] / by_dialect["scored"].where(by_dialect["scored"] > 0)
    wide = by_dialect.pivot_table(index=["model", "thinking"], columns="dialect", values="accuracy")
    columns = [name for name in wide.columns if name != "de"]
    if "de" in wide.columns:
        columns.append("de")  # Standard German last
    wide = wide[columns].round(4)
    wide["overall"] = summary.set_index(["model", "thinking"])["accuracy"]
    wide = wide.reset_index()
    wide.columns.name = None
    dialect_path = summary_path.with_name("accuracy_by_dialect.tsv")
    wide.to_csv(dialect_path, sep="\t", index=False)
    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()
    print(f"overall: {summary['correct'].sum()} / {summary['scored'].sum()} correct "
          f"({summary['correct'].sum() / max(summary['scored'].sum(), 1) * 100:.1f}%)")
    print(f"looping responses: {summary['degenerate'].sum()} / {summary['rows'].sum()} "
          f"({summary['degenerate'].sum() / max(summary['rows'].sum(), 1) * 100:.1f}%)")
    print(f"summary written to {summary_path}")
    print(f"accuracy by dialect written to {dialect_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the model responses in the result TSVs.")
    parser.add_argument("results", nargs="?", default="results", help="directory with the results (default: results)")
    parser.add_argument("--summary", default="answer_summary.tsv",
                        help="where to write the summary tables (default: answer_summary.tsv; "
                             "accuracy_by_dialect.tsv goes in the same folder)")
    parser.add_argument("--dry-run", action="store_true",
                        help="score and summarize without adding the columns to the result files")
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.is_dir():
        parser.error(f"{results_dir} is not a directory")

    paths = sorted(results_dir.rglob("*.tsv"))
    if not paths:
        parser.error(f"no .tsv files found below {results_dir}")

    # The columns are derived data, so they are simply rewritten. A run that is still in progress
    # rewrites whole files though, which would throw the added columns away again.
    changing = [path for path in paths if time.time() - path.stat().st_mtime < 120]
    if changing:
        print(f"warning: {len(changing)} result file(s) changed in the last 2 minutes. "
              f"A benchmark may still be running; it rewrites whole files and would drop the added columns.")

    if args.dry_run:
        print(f"dry run: scoring {len(paths)} file(s) in {results_dir} without changing them")
    else:
        print(f"updating {len(paths)} file(s) in {results_dir} in place")
    stats = [s for path in paths if (s := process_file(path, results_dir, write=not args.dry_run)) is not None]
    summarize(stats, Path(args.summary))


if __name__ == "__main__":
    main()
