"""Create a perturbed version of the Standard German MGSM questions.

A model rephrases each German question with the same people, places, objects, nouns and
mathematical relationships, but with a different sentence structure and information order. Verbs
and adjectives may be replaced by meaning-preserving alternatives. The answer therefore stays the
same, and nothing in the story is reinvented, but the surface text no longer matches the benchmark
questions. This is useful for checking whether a model has memorised the original dataset: if
accuracy drops on the paraphrases, the original wording may have been familiar to it.

Run from the repository root::

    python -m dialect_bench.perturb                       # all German questions
    python -m dialect_bench.perturb --limit 5             # try five questions first
    python -m dialect_bench.perturb --overwrite           # regenerate from scratch

The output is a normal dialect file, so it can be benchmarked with the existing runner:

    python main.py run --data data/dialects/mgsm_german_perturbed.tsv --models ...

The file keeps the original German question in ``question`` and puts the paraphrase in
``dialect_question``. That way the run contains both the original ``de`` reference and the
perturbed variant, and the evaluation can compare them directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from .data import find_files, load_questions, question_id
from .models import OllamaModel

COLUMNS = ["question", "dialect_question", "answer"]
DEFAULT_DATA = ["data/dialects/*.tsv"]
DEFAULT_OUT = "data/dialects/mgsm_german_perturbed.tsv"
DEFAULT_MODEL = "kimi-k3.1-pro:cloud"

# The model is only a rewriter, not a solver. Keeping the arithmetic fixed means the answer
# column can stay exactly as it is in the source data and does not need to be recomputed.
SYSTEM_PROMPT = """Formuliere die gegebene Aufgabe neu, ohne ihren Inhalt zu verändern. Behalte alle Personen, \
Orte, Gegenstände, Substantive, Zahlen und mathematischen Beziehungen exakt bei. Erfinde \
nichts Neues; Satzstruktur, Verben und Adjektive darfst du neu formulieren.

Regeln:
- Behalte alle Personen, Orte, Gegenstände und Substantive unverändert. Führe keine neuen \
Personen, Orte, Gegenstände oder Substantive ein.
- Behalte jede Zahl unverändert bei, egal ob sie als Ziffer oder als Wort geschrieben ist \
(zum Beispiel "16", "drei", "dreimal", "die Hälfte"). Es dürfen keine Zahlen oder Zahlwörter \
geändert, hinzugefügt oder entfernt werden.
- Behalte alle mathematischen Beziehungen und die gesuchte Größe exakt bei.
- Verben und Adjektive darfst du durch bedeutungsgleiche Wörter ersetzen. Sie dürfen die \
Bedeutung aber nicht verändern (nicht "verkauft" zu "kauft", nicht "groß" zu "riesig").
- Übernimm keine Satzstruktur und keine längeren Formulierungen aus der Vorlage. Jeder Satz \
muss neu formuliert sein.
- Beginne mit einem anderen Satz als die Vorlage, ohne neue Personen, Orte oder Gegenstände \
einzuführen.
- Ändere die Reihenfolge, in der die Informationen eingeführt werden, und variiere die \
Satzlänge: Teile lange Sätze auf oder verbinde kurze Sätze.
- Du darfst Satzteile umstellen, Aktiv und Passiv wechseln sowie Nebensätze bilden oder \
auflösen. Erfinde dabei keine neuen Inhalte.
- Formuliere auch die Frage neu. Sie muss nicht mit "Wie viele ..." beginnen und nicht am \
Ende stehen.
- Füge keine Informationen hinzu, die für die Lösung nötig sind oder sie verändern.
- Antworte ausschließlich mit der neu formulierten Aufgabe. Keine Einleitung, keine Erklärung, \
keine Anführungszeichen und kein Markdown."""

USER_PROMPT = """Formuliere die folgende Textaufgabe neu. Ändere die Satzstruktur und die \
Reihenfolge der Informationen. Verben und Adjektive darfst du durch bedeutungsgleiche Wörter \
ersetzen. Behalte alle Personen, Orte, Gegenstände, Substantive, Zahlen und mathematischen \
Beziehungen exakt bei. Erfinde nichts Neues und verändere die Bedeutung nicht.

<aufgabe>
{question}
</aufgabe>"""


def read_output(path: Path) -> dict[str, dict]:
    """Existing rows, keyed by the id that data.py derives from the German question."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = set(COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = {}
        for row in reader:
            if row["question"].strip():
                rows[question_id(row["question"])] = {column: row.get(column, "") for column in COLUMNS}
    return rows


def write_output(path: Path, questions, rows: dict[str, dict]) -> None:
    """Write the completed rows in source order, so a run can be resumed safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, COLUMNS, delimiter="\t")
        writer.writeheader()
        for question in questions:
            if question.id in rows:
                writer.writerow(rows[question.id])
    temporary.replace(path)


def rewrite(model: OllamaModel, question, options: dict, seed: int) -> str:
    """Ask the model for one paraphrase of a German question."""
    prompt = USER_PROMPT.format(question=question.text)
    answer, _, _ = model.ask(
        prompt, thinking=False, system=SYSTEM_PROMPT,
        options={**options, "seed": seed},
    )
    return answer.strip()


def save_meta(path: Path, info: dict) -> None:
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paraphrase the Standard German MGSM questions.")
    parser.add_argument("--data", nargs="+", default=DEFAULT_DATA,
                        help=f"source files or glob patterns (default: {DEFAULT_DATA[0]})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"output TSV (default: {DEFAULT_OUT})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model used as rewriter (default: {DEFAULT_MODEL})")
    parser.add_argument("--host", default=None,
                        help="Ollama host (default: $OLLAMA_HOST or 127.0.0.1:11434)")
    parser.add_argument("--parallel", type=int, default=4,
                        help="questions sent to the model at the same time (default: 4)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only work on the first N questions, for testing")
    parser.add_argument("--seed", type=int, default=0,
                        help="sampling seed (default: 0)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="sampling temperature for the rewrite (default: 0.9)")
    parser.add_argument("--overwrite", action="store_true",
                        help="ignore existing output and paraphrase every question again")
    args = parser.parse_args()
    if args.parallel < 1:
        parser.error("--parallel must be at least 1")

    output = Path(args.out)
    # The default glob also matches the output file. It only contains Standard German questions,
    # so it would not add anything, but leaving it out keeps the run from reading its own result.
    files = [path for path in find_files(args.data) if path.resolve() != output.resolve()]
    if not files:
        parser.error("no source files left after excluding the output file")
    # Every dialect file contains the same German questions. load_questions de-duplicates them by
    # the whitespace-normalised German text and returns the Standard German ones as dialect "de".
    all_questions = [q for q in load_questions(files, reference="de") if q.dialect == "de"]
    if not all_questions:
        parser.error("no Standard German questions found in the source files")

    rows = {} if args.overwrite else read_output(output)
    selected = all_questions[:args.limit] if args.limit else all_questions
    todo = [q for q in selected if not rows.get(q.id, {}).get("dialect_question", "").strip()]

    if args.host:
        os.environ["OLLAMA_HOST"] = args.host
    options = {"temperature": args.temperature, "top_p": 0.95, "num_ctx": 8192, "num_predict": 4096}
    try:
        model = OllamaModel(args.model, options)
    except Exception as error:
        parser.error(f"could not set up Ollama model {args.model!r}: {error}")

    print(f"{args.model} via {model.url}")
    print(f"{len(todo)} question(s) to rewrite, {len(rows)} already in {output}")

    failures = []
    if todo:
        with ThreadPoolExecutor(args.parallel) as pool:
            futures = {
                pool.submit(rewrite, model, q, options, args.seed): q
                for q in todo
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="rewriting", mininterval=10):
                question = futures[future]
                try:
                    rewritten = future.result()
                except Exception as error:
                    failures.append(question.id)
                    print(f"{question.id} failed, rerun to retry: {error}")
                    continue
                rows[question.id] = {
                    "question": question.text,
                    "dialect_question": rewritten,
                    "answer": question.answer,
                }
                write_output(output, all_questions, rows)

    # Make sure the header exists even when nothing had to be done.
    write_output(output, all_questions, rows)

    try:
        model_info = model.describe()
    except Exception as error:  # the generated data is more important than the metadata
        model_info = {"tag": args.model, "error": str(error)}

    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": [path.name for path in files],
        "output": output.name,
        "questions": len(all_questions),
        "written": sum(bool(rows.get(q.id, {}).get("dialect_question", "").strip()) for q in all_questions),
        "model": model_info,
        "options": options,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "seed": args.seed,
        "failures": failures,
    }
    meta_path = output.with_suffix(".meta.json")
    save_meta(meta_path, meta)

    print(f"{meta['written']} / {len(all_questions)} question(s) in {output}")
    print(f"generation settings in {meta_path}")
    if failures:
        print(f"{len(failures)} question(s) still need a rewrite: {', '.join(failures)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
