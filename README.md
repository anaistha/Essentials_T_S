# Math word problems in German dialects

Do LLMs solve math word problems equally well in Standard German and in German dialects?
We ask open-weight models (Gemma 4, DeepSeek-R1, Apertus) the same questions in each dialect and compare
their accuracy to Standard German.

## Data

The questions come from [MGSM-Rev2](https://github.com/google-research-datasets/MGSM-Rev2), a revised
version of the MGSM benchmark: 250 grade-school math problems in English and 10 other languages, including
German, with fixed translation errors. Each answer is a single number. We use 125 of the German
questions and translate them into the dialects.

There is one file per dialect in `data/dialects/`, named after the dialect (e.g. `mgsm_biel.tsv`), as TSV
or CSV with a header row and these columns:

| question | dialect_question | answer |
|---|---|---|
| question in Standard German | the same question in the dialect | the answer |

The Standard German questions are evaluated once, as the reference `de`.

### Perturbed German questions

`data/dialects/mgsm_german_perturbed.tsv` starts as a copy of the German questions with an empty
`dialect_question` column. `dialect_bench/perturb.py` fills it with paraphrases made by an Ollama
cloud model (default `kimi-k3.1-pro:cloud`). The German system prompt asks the model to keep the
same people, places, objects and nouns, but to change the sentence order and sentence structure.
Verbs and adjectives may be replaced by meaning-preserving alternatives. All numbers and
mathematical relationships must stay the same, so the original answers remain valid. The script
writes the model's response directly; correctness is the prompt's responsibility, not a separate
check.

```bash
python -m dialect_bench.perturb --limit 5     # try five questions first
python -m dialect_bench.perturb               # rewrite the rest; rerunning resumes
```

The resulting file has the original German in `question` and the paraphrase in `dialect_question`.
This means a normal run evaluates both the `de` reference and the perturbed variant, so a drop in
accuracy on `mgsm_german_perturbed` can indicate that a model memorised the original wording.
Generation settings and the full German prompt are saved in `mgsm_german_perturbed.meta.json`.

Perturbed German is a memorisation check, not a dialect, so the evaluation keeps it out of the
dialect grids and the overall accuracy. `accuracy_perturbed_german.tsv` compares each model's
perturbed German accuracy with its Standard German accuracy instead (`_llm` columns are included
after running the judge).

## Running

```bash
conda env create -f environment.yml     # once; the Slurm job activates it itself
conda activate dialect_math
```

On the cluster, submit `slurm/run_ollama.submit`. Its options are passed on to `main.py run`:

```bash
sbatch slurm/run_ollama.submit --thinking on off                  # all models, one after the other
sbatch slurm/run_ollama.submit --models gemma4-12b apertus-8b --thinking off
sbatch slurm/run_ollama.submit --models gemma4-e2b --limit 3      # quick test
```

The job downloads [Ollama](https://ollama.com) and the models to `/scratch/thmv/ollama` on its node (only
the first time on each node). Each model gets up to 6 questions at the same time (`parallel` in the
config), fewer if it doesn't fit in GPU memory otherwise. If the job stops, submit it again: questions
that already have an answer are skipped. Logs are in `~/log`.

Models and settings are in `config.yaml`. The DeepSeek-R1 1.5b, 7b, 14b and 32b models always think, so
they are skipped for `--thinking off` (marked `always_thinks` in the config). Apertus can't think at all,
so it is skipped for `--thinking on`.

Ollama cloud models are supported as well. Sign in with `ollama signin`, pull the model, and add its
cloud tag (for example `gpt-oss:120b-cloud`) to the `models` section in `config.yaml`. Cloud models
are sent through the same Ollama API, but are not checked against local GPU memory.

## Running with Codex CLI

Codex runs use `config_codex.yaml` and require the Codex CLI to be installed and signed in with a ChatGPT account. From the repository root, activate the project’s Python environment and run:

```powershell
conda activate dialect_math
python main.py run --config config_codex.yaml --data "data/dialects/*.tsv" --models codex-luna --thinking off --parallel 1
```

This processes all TSV files in `data/dialects/` and their matching Standard German questions. To run only selected dialects, replace the pattern after `--data` with their file paths.

Each question is sent in a separate ephemeral Codex CLI invocation. Results are saved under `results/codex-luna/thinking-off/`, with the complete final response in `predicted`. Run settings are recorded in `run_info.jsonl`. If a run is interrupted, run the same command again; questions with saved answers are skipped.

## Running with Claude Code

Claude models (Opus 5.5, Sonnet 5.5, Haiku 5.5) run through the Claude Code CLI with a Claude subscription,
using `config_claude.yaml`. Sign in once by running `claude` and `/login`. No GPU is needed, so run it on a
login node or your own computer, ideally inside `tmux`:

```bash
conda activate dialect_math
python main.py run --config config_claude.yaml --models claude-haiku-5.5
python main.py run --config config_claude.yaml --models "claude-*" --data data/dialects/mgsm_biel.tsv
```

Each question is a separate `claude -p` call in an empty folder, with no tools, no settings files and an
empty system prompt. Results go to `results/claude-<model>/thinking-on/`. When the subscription's usage
limit is reached, the run waits and continues; if it stops, run the same command again.

Differences from the open-weight models:
- Only thinking on: Claude Code can't reliably turn thinking off.
- No temperature or seed can be set, so answers can differ between runs.
- Claude doesn't return its reasoning, so the `reasoning` column stays empty.
- Claude Code adds a few hundred tokens of its own context to every call, even with an empty system prompt.

## Scoring the results

`dialect_bench/eval.py` extracts the final number from every response in `results/` and scores it against
`gt_answer`. Run it from the repository root after a benchmark finishes:

```bash
python -m dialect_bench.eval                 # update results/ in place
python -m dialect_bench.eval --dry-run       # score only, leave results/ untouched
python -m dialect_bench.eval --summary out.tsv
```

Each result TSV gets three columns:

| column | meaning |
|---|---|
| `extracted_answer` | the final number in `predicted` (empty if there is none) |
| `correct` | 1 if it matches `gt_answer`, 0 if not (a missing answer counts as wrong) |
| `degenerate` | 1 if the model got stuck repeating itself, e.g. `zäma zäma zäma …` |

The answer comes from `predicted`, not `done_reason`: `done_reason` only records why generation stopped
(`stop`/`length`) and never contains the answer. Number formats such as `1'430`, `7'425,00` and `$20` are
handled.

### LLM re-check

Regex sometimes picks a number that is not the model's real final answer. With `--llm`, an Ollama model
re-reads every row where `correct` is 0 and writes two more columns:

| column | meaning |
|---|---|
| `llm_answer` | the final answer the judge found (empty if it found none) |
| `llm_correct` | 1 if `llm_answer` matches `gt_answer`, 0 if not, empty if the row was not judged |

```bash
python -m dialect_bench.eval --llm                    # judge with gemma4:31b-cloud
python -m dialect_bench.eval --llm --judge-limit 50   # try it on 50 rows first
python -m dialect_bench.eval --llm --judge-model gpt-oss:120b-cloud
```

The judge sees the full `predicted` response and is asked for only the final number (or `NONE`). Answers
are cached in `llm_judge_cache.json`, so an interrupted run continues without asking the same question
again; `--refresh-judge` ignores the cache. Related options: `--judge-parallel` (default 4),
`--judge-host` (default `$OLLAMA_HOST` or `127.0.0.1:11434`) and `--judge-limit`.

The summary then shows regex accuracy and accuracy after the judge's corrections. The dialect grids are
written to `accuracy_by_dialect.tsv` (regex) and `accuracy_by_dialect_llm.tsv` (after corrections).

The script prints a summary per model and thinking mode and writes these files next to the `--summary` path
(`answer_summary.tsv` by default):

- `answer_summary.tsv`: rows, answered, scored, correct, regex accuracy, LLM-corrected accuracy, looping count and percentage.
- `accuracy_by_dialect.tsv`: regex accuracy, one row per model and thinking mode, one column per dialect plus `de` and `overall`.
- `accuracy_by_dialect_llm.tsv`: the same grid after the LLM corrections (only written with `--llm`).

Looping responses are marked `degenerate=1` but are still scored as wrong, so a model that fails to
produce an answer is visible in the accuracy instead of being dropped.
