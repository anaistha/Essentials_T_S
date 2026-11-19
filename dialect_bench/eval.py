"""Score the model responses and write accuracy summaries.

The model's full response (reasoning included) is in the ``predicted`` column.
``done_reason`` only says why generation stopped ("stop"/"length") and never
contains an answer, so it must not be used for extraction.

Run from the repository root::

    python -m dialect_bench.eval                  # regex scoring, results/ in place
    python -m dialect_bench.eval --dry-run        # score without changing results/
    python -m dialect_bench.eval --llm            # also let an LLM re-read wrong answers

Every TSV below ``results`` gets three extra columns:

``extracted_answer``
    The final number found by regex in ``predicted`` (empty if there is none).
``correct``
    1 if it matches ``gt_answer``, 0 if not (a missing number counts as wrong),
    empty if ``gt_answer`` itself is missing.
``degenerate``
    1 if ``predicted`` is stuck in a repetition loop, otherwise 0. These are
    still scored, so a looping model shows up as wrong rather than disappearing.

With ``--llm``, rows where regex says wrong also get:

``llm_answer``
    The answer the judge model found in ``predicted`` (empty if it found none).
``llm_correct``
    1 if ``llm_answer`` matches ``gt_answer``, 0 if not, empty if not judged.

A summary per model and thinking mode is printed and written to
``answer_summary.tsv``, and the model x dialect grids to
``accuracy_by_dialect.tsv`` (regex) and ``accuracy_by_dialect_llm.tsv`` (LLM judge).

Perturbed German questions are a memorisation check, not a dialect. They are kept out of
the dialect grids and the overall accuracy, and compared with Standard German in
``accuracy_perturbed_german.tsv`` (and the ``_llm`` variant when the judge has run).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# A number as models write it, including Swiss/German grouping such as 1'430 or 7'425,00
NUMBER = re.compile(r"-?\d(?:[\d.,'’]*\d)?")
# Numbers that the model highlighted as the answer
BOLD = re.compile(r"\*\*([^*]+)\*\*|\\mathbf\{([^{}]*)\}")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
# Markdown/LaTeX lines that may legitimately repeat without meaning the model is stuck
PUNCTUATION_LINE = re.compile(r"^[\s\\${}\[\]|*_#>\-–—=+.,;:!?'\"()/]+$")

DEFAULT_JUDGE_MODEL = "gemma4:31b-cloud"
JUDGE_PROMPT = """You extract the final numerical answer from a model's response to a math word problem.

The response may contain reasoning, false starts, corrections, or long repeated text. Ignore all of that and report only the number that the response finally settles on as its answer.

If the response contains no clear final numerical answer, reply exactly: NONE

<response>
{response}
</response>

Reply with only the number, or NONE. Do not explain."""


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


def format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.10g}"


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


class Judge:
    """Asks a local or cloud Ollama model for the answer in a response.

    Answers are cached on disk, so an interrupted run or a later ``--dry-run``
    does not pay for the same response twice. ``None`` means the judge found no
    answer; the key is still cached so it is not asked again.
    """

    def __init__(self, model: str, host: str, cache_path: Path, parallel: int = 4,
                 refresh: bool = False, limit: int | None = None):
        self.model = model
        self.host = host if host.startswith("http") else "http://" + host
        self.cache_path = cache_path
        self.refresh = refresh
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=max(1, parallel))
        self.remaining = limit
        self.asks = 0
        self.errors = 0
        self.cache = {}
        self._dirty = 0
        if cache_path and cache_path.exists() and not refresh:
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                print(f"warning: could not read {cache_path}, starting a new judge cache")

    def key(self, text: str) -> str:
        return self.model + "\x00" + hashlib.sha1(text.encode("utf-8")).hexdigest()

    def ask(self, text: str) -> tuple[float | None, str | None]:
        """Return ``(answer, error)``; answer is None when the judge says NONE."""
        text = str(text)
        key = self.key(text)
        with self.lock:
            if not self.refresh and key in self.cache:
                return self.cache[key], None
            if self.remaining is not None:
                if self.remaining <= 0:
                    return None, "limit"
                self.remaining -= 1

        answer, error = self._request(text)
        if error is not None:
            with self.lock:
                self.errors += 1
            return None, error

        with self.lock:
            self.cache[key] = answer
            self.asks += 1
            self._dirty += 1
            if self._dirty >= 20:
                self._save_locked()
        return answer, None

    def judge_many(self, texts: list[str], desc: str | None = None) -> list[tuple[float | None, str | None]]:
        results = self.pool.map(self.ask, texts)
        return list(tqdm(results, total=len(texts), desc=desc, leave=False, unit="row", position=1))

    def _request(self, text: str) -> tuple[float | None, str | None]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": JUDGE_PROMPT.format(response=text)}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": 0, "num_predict": 256},
        }
        error = "unknown error"
        for attempt in range(3):
            try:
                response = requests.post(f"{self.host}/api/chat", json=payload, timeout=600)
                if response.ok:
                    content = response.json().get("message", {}).get("content", "")
                    return self._parse(content), None
                error = f"HTTP {response.status_code}: {response.text[:200]}"
            except requests.RequestException as exc:
                error = str(exc)
            time.sleep(2 * (attempt + 1))
        return None, error

    @staticmethod
    def _parse(content: str) -> float | None:
        text = str(content).strip()
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip().strip("\"'`* ")
        if text.upper() in {"", "NONE", "NO ANSWER", "N/A", "NA"}:
            return None
        value = parse_number(text)
        if value is not None:
            return value
        values = numbers_in(text)
        return values[0] if values else None

    def save(self, force: bool = False) -> None:
        with self.lock:
            self._save_locked(force)

    def _save_locked(self, force: bool = False) -> None:
        if not self.cache_path or (self._dirty == 0 and not force):
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.cache), encoding="utf-8")
        temporary.replace(self.cache_path)
        self._dirty = 0

    def close(self) -> None:
        self.pool.shutdown()
        self.save(force=True)


def process_file(path: Path, results_dir: Path, judge: Judge | None = None,
                 write: bool = True) -> dict | None:
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
    judge_errors = 0
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

            if judge is not None:
                for column in ("llm_answer", "llm_correct"):
                    if column not in df.columns:
                        df[column] = ""
                # Re-check every wrong row; the cache makes unchanged responses free.
                # This also refreshes rows whose predicted text changed since the last run.
                todo = pd.to_numeric(df["correct"], errors="coerce") == 0
                if todo.any():
                    positions = df.index[todo]
                    results = judge.judge_many(
                        df.loc[todo, "predicted"].tolist(),
                        desc=f"judge {path.relative_to(results_dir)}",
                    )
                    for position, (answer, error) in zip(positions, results):
                        if error == "limit":
                            continue  # leave this row unjudged; a later run with a larger limit can do it
                        if error is not None:
                            judge_errors += 1
                            print(f"judge failed for {path.relative_to(results_dir)} "
                                  f"row {df.at[position, 'id']}: {error}")
                            continue
                        expected = truth.at[position]
                        # keep these columns as strings: pandas' str dtype rejects ints
                        if answer is None:
                            df.at[position, "llm_answer"] = ""
                            df.at[position, "llm_correct"] = "0" if pd.notna(expected) else ""
                        else:
                            df.at[position, "llm_answer"] = format_number(answer)
                            df.at[position, "llm_correct"] = (
                                "" if pd.isna(expected)
                                else "1" if math.isclose(answer, expected, rel_tol=0, abs_tol=1e-9) else "0"
                            )
    else:
        print(f"warning: {path.relative_to(results_dir)} has no 'predicted' column, leaving it unchanged")

    if write:
        df.to_csv(path, sep="\t", index=False)

    judged = int((df["llm_correct"].astype(str).str.strip() != "").sum()) if "llm_correct" in df.columns else 0
    llm_correct = int(pd.to_numeric(df["llm_correct"], errors="coerce").fillna(0).sum()) if judged else 0
    if judged:
        regex = pd.to_numeric(df["correct"], errors="coerce")
        llm = pd.to_numeric(df["llm_correct"], errors="coerce")
        llm_changed = int((llm - regex).where(llm.notna(), 0).fillna(0).sum())
    else:
        llm_changed = 0

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
        "judged": judged,
        "llm_correct": llm_correct,
        "llm_changed": llm_changed,
        "judge_errors": judge_errors,
    }


def is_perturbation(dialect: str) -> bool:
    """True for the perturbed German memorisation check, which is not a dialect."""
    return "perturb" in str(dialect).lower()


def accuracy_by_group(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate regex and LLM accuracy per model and thinking mode."""
    grouped = (
        frame.groupby(["model", "thinking"], sort=True)
        .agg(correct=("correct", "sum"), scored=("scored", "sum"),
             llm_changed=("llm_changed", "sum"), judged=("judged", "sum"), rows=("rows", "sum"))
    )
    grouped["regex_accuracy"] = grouped["correct"] / grouped["scored"].where(grouped["scored"] > 0)
    grouped["llm_accuracy"] = (
        (grouped["correct"] + grouped["llm_changed"])
        / grouped["scored"].where(grouped["scored"] > 0)
    )
    return grouped


def summarize_perturbed(dialects: pd.DataFrame, perturbed: pd.DataFrame, summary_path: Path) -> None:
    """Compare each model's perturbed German accuracy with its Standard German accuracy."""
    perturbed_accuracy = accuracy_by_group(perturbed)
    german = dialects[dialects["dialect"] == "de"]
    if german.empty:
        print("warning: no Standard German (de) results, cannot compare the perturbed German accuracy")
        return
    german_accuracy = accuracy_by_group(german)

    table = pd.DataFrame(index=perturbed_accuracy.index)
    table["de_regex"] = german_accuracy["regex_accuracy"]
    table["perturbed_regex"] = perturbed_accuracy["regex_accuracy"]
    table["delta_regex"] = (table["perturbed_regex"] - table["de_regex"]).round(4)
    if perturbed_accuracy["judged"].sum() > 0:
        table["de_llm"] = german_accuracy["llm_accuracy"]
        table["perturbed_llm"] = perturbed_accuracy["llm_accuracy"]
        table["delta_llm"] = (table["perturbed_llm"] - table["de_llm"]).round(4)
    table = table.round(4).reset_index()

    path = summary_path.with_name("accuracy_perturbed_german.tsv")
    table.to_csv(path, sep="\t", index=False)

    print()
    print("perturbed German vs Standard German:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"perturbed German comparison written to {path}")


def summarize(stats: list[dict], summary_path: Path) -> None:
    frame = pd.DataFrame(stats)
    if frame.empty:
        print("no result files found")
        return

    # The perturbed German file is a memorisation check, so it must not be averaged into the
    # dialect accuracy or the overall score. It gets its own table below.
    perturbed = frame[frame["dialect"].map(is_perturbation)]
    frame = frame[~frame["dialect"].map(is_perturbation)]
    if frame.empty:
        print("no dialect result files found, only perturbed German")
        summarize_perturbed(frame, perturbed, summary_path)
        return

    summary = (
        frame.groupby(["model", "thinking"], sort=True)
        .agg(files=("file", "count"), rows=("rows", "sum"), answered=("answered", "sum"),
             scored=("scored", "sum"), correct=("correct", "sum"), degenerate=("degenerate", "sum"),
             judged=("judged", "sum"), llm_correct=("llm_correct", "sum"),
             llm_changed=("llm_changed", "sum"), judge_errors=("judge_errors", "sum"))
        .reset_index()
    )
    summary["accuracy"] = (summary["correct"] / summary["scored"].where(summary["scored"] > 0)).round(4)
    summary["final_correct"] = summary["correct"] + summary["llm_changed"]
    summary["llm_accuracy"] = (summary["final_correct"] / summary["scored"].where(summary["scored"] > 0)).round(4)
    summary["degenerate_pct"] = (summary["degenerate"] / summary["rows"] * 100).round(1)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, sep="\t", index=False)

    # Model x dialect grids: once from regex, once after the LLM corrections
    by_dialect = (
        frame.groupby(["model", "thinking", "dialect"], sort=True)
        .agg(correct=("correct", "sum"), scored=("scored", "sum"), llm_changed=("llm_changed", "sum"),
             judged=("judged", "sum"))
        .reset_index()
    )
    dialects = [name for name in by_dialect["dialect"].unique() if name != "de"]
    if "de" in set(by_dialect["dialect"]):
        dialects.append("de")  # Standard German last
    overall = summary.set_index(["model", "thinking"])

    def grid(column: str) -> pd.DataFrame:
        wide = by_dialect.pivot_table(index=["model", "thinking"], columns="dialect", values=column)
        wide = wide[[name for name in dialects if name in wide.columns]].round(4)
        wide["overall"] = overall["accuracy" if column == "regex_accuracy" else column]
        wide = wide.reset_index()
        wide.columns.name = None
        return wide

    by_dialect["regex_accuracy"] = by_dialect["correct"] / by_dialect["scored"].where(by_dialect["scored"] > 0)
    by_dialect["llm_accuracy"] = (
        (by_dialect["correct"] + by_dialect["llm_changed"])
        / by_dialect["scored"].where(by_dialect["scored"] > 0)
    )
    grid("regex_accuracy").to_csv(summary_path.with_name("accuracy_by_dialect.tsv"), sep="\t", index=False)

    if summary["judged"].sum() > 0:
        grid("llm_accuracy").to_csv(summary_path.with_name("accuracy_by_dialect_llm.tsv"), sep="\t", index=False)

    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()
    print(f"overall regex: {summary['correct'].sum()} / {summary['scored'].sum()} correct "
          f"({summary['correct'].sum() / max(summary['scored'].sum(), 1) * 100:.1f}%)")
    if summary["judged"].sum() > 0:
        print(f"overall after LLM: {summary['final_correct'].sum()} / {summary['scored'].sum()} correct "
              f"({summary['final_correct'].sum() / max(summary['scored'].sum(), 1) * 100:.1f}%), "
              f"{summary['judged'].sum()} wrong answers re-read, net {summary['llm_changed'].sum():+d}")
        print(f"judge errors: {summary['judge_errors'].sum()}")
    print(f"looping responses: {summary['degenerate'].sum()} / {summary['rows'].sum()} "
          f"({summary['degenerate'].sum() / max(summary['rows'].sum(), 1) * 100:.1f}%)")
    print(f"summary written to {summary_path}")
    print(f"regex accuracy by dialect written to {summary_path.with_name('accuracy_by_dialect.tsv')}")
    if summary["judged"].sum() > 0:
        print(f"LLM accuracy by dialect written to {summary_path.with_name('accuracy_by_dialect_llm.tsv')}")

    if not perturbed.empty:
        summarize_perturbed(frame, perturbed, summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the model responses in the result TSVs.")
    parser.add_argument("results", nargs="?", default="results", help="directory with the results (default: results)")
    parser.add_argument("--summary", default="answer_summary.tsv",
                        help="where to write the summary tables (default: answer_summary.tsv)")
    parser.add_argument("--dry-run", action="store_true",
                        help="score and summarize without adding the columns to the result files")
    parser.add_argument("--llm", action="store_true",
                        help="let an Ollama model re-read the answers regex marked wrong")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help=f"Ollama model used as judge (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--judge-host", default=os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"),
                        help="Ollama host for the judge (default: $OLLAMA_HOST or 127.0.0.1:11434)")
    parser.add_argument("--judge-parallel", type=int, default=4,
                        help="judge requests to run at the same time (default: 4)")
    parser.add_argument("--judge-limit", type=int, default=None,
                        help="only ask the judge about this many new rows in this run")
    parser.add_argument("--judge-cache", default=None,
                        help="judge answer cache (default: llm_judge_cache.json next to the summary)")
    parser.add_argument("--refresh-judge", action="store_true",
                        help="ignore the judge cache and ask again")
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

    summary_path = Path(args.summary)
    judge = None
    if args.llm:
        cache_path = Path(args.judge_cache) if args.judge_cache else summary_path.with_name("llm_judge_cache.json")
        judge = Judge(args.judge_model, args.judge_host, cache_path, args.judge_parallel,
                      args.refresh_judge, args.judge_limit)
        print(f"LLM judge: {args.judge_model} via {judge.host} "
              f"(cache: {cache_path}, parallel: {args.judge_parallel})")

    if args.dry_run:
        print(f"dry run: scoring {len(paths)} file(s) in {results_dir} without changing them")
    else:
        print(f"updating {len(paths)} file(s) in {results_dir} in place")
    stats = []
    for path in tqdm(paths, desc="files", unit="file", position=0):
        result = process_file(path, results_dir, judge, write=not args.dry_run)
        if result is not None:
            stats.append(result)

    if judge is not None:
        judge.close()
        print(f"judge asked {judge.asks} new question(s), {judge.errors} error(s)")
    summarize(stats, summary_path)


if __name__ == "__main__":
    main()
