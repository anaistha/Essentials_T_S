# Math word problems in German dialects

Do LLMs solve math word problems equally well in Standard German and in German dialects?
We ask open-weight models (Gemma 4, DeepSeek-R1, Apertus) the same questions in each dialect and compare
their accuracy to Standard German.

## Data

The questions come from [MGSM-Rev2](https://github.com/google-research-datasets/MGSM-Rev2), a revised
version of the MGSM benchmark: 250 grade-school math problems in English and 10 other languages, including
German, with fixed translation errors. Each answer is a single number. We use 125 of the German
questions and translate them into the dialects.

There is one file per dialect in `data/dialects/`, named after the dialect (e.g. `gsw_zh.tsv`), as TSV or
CSV with three columns:

| Standard German question | question in dialect | answer |
|---|---|---|

The Standard German questions are evaluated once, as the reference `de`. The `example_*` files in
`data/dialects/` are 3-question examples for testing; delete them once the real files are there.

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

## Results

`results/<model>/thinking-<mode>/` has one file per input file (same name and format), plus `de` for
Standard German, with the columns `id`, `prompt`, `predicted` (the model's full answer), `gt_answer` and
`reasoning` (the model's thinking, if any). The answers are not evaluated yet.
