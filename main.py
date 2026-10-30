import argparse
import sys
from fnmatch import fnmatch
from pathlib import Path

import yaml

from dialect_bench.data import find_files, load_questions
from dialect_bench import run_info
from dialect_bench.models import THINKING, OllamaModel
from dialect_bench.run import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["list", "fits", "run"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", nargs="+", help="input files or glob patterns (default: from config)")
    parser.add_argument("--models", nargs="+", default=["*"],
                        help="model names from the config, wildcards allowed (default: all)")
    parser.add_argument("--thinking", nargs="+", choices=THINKING, help="default: from config")
    parser.add_argument("--parallel", type=int, default=1, help="questions sent to the model at the same time")
    parser.add_argument("--limit", type=int, help="only the first N questions, for testing")
    parser.add_argument("--out", help="results directory (default: from config)")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    base = config_path.parent
    files = find_files(args.data) if args.data else find_files(config["data"], base)
    reference = config["reference"]
    results_dir = Path(args.out) if args.out else base / config["results_dir"]

    questions = load_questions(files, reference)
    if args.limit:
        keep = list(dict.fromkeys(q.id for q in questions))[:args.limit]
        questions = [q for q in questions if q.id in keep]

    # One output file per input file, with the same name. Standard German always goes to de.tsv, so when
    # another dialect is run with the same model, its answers are found and not asked a second time.
    file_names = {reference: reference + ".tsv"} | {f.stem: f.name for f in files}

    models = {name: m for name, m in config["models"].items()
              if any(fnmatch(name, pattern) for pattern in args.models)}
    if not models:
        parser.error(f"--models must match one of: {', '.join(config['models'])}")
    # YAML turns on/off into True/False, so we turn them back into text
    thinking = args.thinking or [{True: "on", False: "off"}.get(t, t) for t in config["thinking"]]

    def thinking_modes(m):
        # these models think anyway, so asking them with thinking off makes no sense
        if m.get("always_thinks"):
            return [t for t in thinking if t != "off"]
        return thinking

    if args.command == "list":
        # used by the Slurm script: the models that have something to do
        print(*[name for name, m in models.items() if thinking_modes(m)])
        return

    if args.command == "fits":
        # used by the Slurm script: exit code 3 if the model doesn't fit on the GPU with the current
        # OLLAMA_NUM_PARALLEL (any other error exits with 1)
        for m in models.values():
            if not OllamaModel(m["tag"], {**config["ollama_options"], **m.get("options", {})}).fits_on_gpu():
                sys.exit(3)
        return

    for name, m in models.items():
        modes = thinking_modes(m)
        if modes != thinking:
            print(f"{name} always thinks, skipping thinking=off")
        if not modes:
            continue
        model = OllamaModel(m["tag"], {**config["ollama_options"], **m.get("options", {})})
        info = {"dataset": [f.name for f in files], "limit": args.limit, "gpu": run_info.gpu()}
        run(name, model, questions, modes, config["prompt"], results_dir, file_names, args.parallel, info)


if __name__ == "__main__":
    main()
