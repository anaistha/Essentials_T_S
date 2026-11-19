"""Compare question lengths: German GT of the 125 dialect questions vs. full MGSM-Rev2 German (250).

Usage: python compare_lengths.py [--unit question|sentence] [--out lengths_hist.png]
"""
import argparse
import csv
import os
import io
import re
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd

MGSM_DE_URL = ("https://raw.githubusercontent.com/google-research-datasets/"
               "MGSM-Rev2/main/MGSM-Rev2/mgsm_de.tsv")
DIALECT_TSV = "data/dialects/mgsm_zuerich.tsv"  # any dialect file; "question" column = German GT


def n_words(text):
    return len(text.split())


def lengths(texts, unit):
    if unit == "question":
        return [n_words(t) for t in texts]
    # split into sentences on . ! ? followed by whitespace (avoids "80.000")
    return [n_words(s) for t in texts for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", choices=["question", "sentence"], default="question")
    ap.add_argument("--out", default="images/lengths_hist.svg")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    raw = urllib.request.urlopen(MGSM_DE_URL).read().decode("utf-8")
    full = pd.read_csv(io.StringIO(raw), sep="\t", header=None, names=["question", "answer"],
                       quoting=csv.QUOTE_NONE, dtype=str)
    subset = pd.read_csv(DIALECT_TSV, sep="\t", dtype=str, keep_default_na=False,
                         encoding="utf-8-sig")

    full_len = lengths(full["question"], args.unit)
    sub_len = lengths(subset["question"], args.unit)

    for name, xs in [(f"Original MGSM-Rev2 de (n={len(full)})", full_len),
                     (f"Selected Subset de (n={len(subset)})", sub_len)]:
        s = pd.Series(xs)
        print(f"{name:28s} {args.unit}s={len(s):4d}  mean={s.mean():5.1f}  "
              f"median={s.median():5.1f}  min={s.min()}  max={s.max()}")

    bins = range(0, max(full_len + sub_len) + 3, 3 if args.unit == "question" else 2)
    plt.figure(figsize=(8, 4.5))
    plt.hist(full_len, bins=bins, density=True, alpha=0.5, label=f"MGSM-Rev2 de (250)")
    plt.hist(sub_len, bins=bins, density=True, alpha=0.5, label=f"Dialect GT de ({len(subset)})")
    plt.xlabel(f"Words per {args.unit}")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
