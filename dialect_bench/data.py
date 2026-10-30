import csv
import glob
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Question:
    id: str
    dialect: str
    text: str
    answer: str


def question_id(german_text):
    # The id is made from the German text, so a question gets the same id in every dialect file.
    text = re.sub(r"\s+", " ", german_text).strip()
    return "q" + hashlib.sha1(text.encode()).hexdigest()[:8]


def find_files(patterns, base=Path(".")):
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(base / pattern)))
        if not matches:
            raise FileNotFoundError(f"no files match {pattern}")
        files += [Path(m) for m in matches]
    return files


def read_file(path):
    if path.suffix == ".tsv":
        sep = "\t"
    else:
        sample = path.read_text(encoding="utf-8-sig")[:10000]
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = {"question", "dialect_question", "answer"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}, found {list(df.columns)}")
    return df


def load_questions(files, reference="de"):
    """Returns all questions, the Standard German ones first."""
    german = {}
    questions = []
    for path in files:
        df = read_file(path)
        for _, row in df.iterrows():
            qid = question_id(row["question"])
            german.setdefault(qid, Question(qid, reference, row["question"].strip(), row["answer"].strip()))
            if row["dialect_question"].strip():
                questions.append(Question(qid, path.stem, row["dialect_question"].strip(), row["answer"].strip()))
        print(f"{path.name}: {len(df)} questions")
    return list(german.values()) + questions
