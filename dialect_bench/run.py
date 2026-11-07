import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from . import run_info
from .models import THINKING

COLUMNS = ["id", "prompt", "predicted", "gt_answer", "reasoning", "done_reason"]

# reasoning fields can be longer than the csv default of 131072 characters
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def read_output(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t" if path.suffix == ".tsv" else ","))


def write_output(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, COLUMNS, delimiter="\t" if path.suffix == ".tsv" else ",")
        writer.writeheader()
        writer.writerows(rows)


def run(name, model, questions, thinking_modes, prompt_template, out_dir, file_names, parallel=1, info=None):
    """Asks the questions, `parallel` of them at the same time.

    file_names maps each dialect to the name of its output file, e.g. gsw_zh -> gsw_zh.tsv
    info is saved to run_info.jsonl together with the model's details.
    """
    for mode in thinking_modes:
        if THINKING[mode] and not model.can_think:
            print(f"{name} can't think, skipping thinking={mode}")
            continue

        run_info.save(out_dir / name / f"thinking-{mode}",
                      {"model": name, **model.describe(), "thinking": mode, "options": model.options,
                       "prompt": prompt_template, **(info or {})})

        for dialect, file_name in file_names.items():
            path = out_dir / name / f"thinking-{mode}" / file_name
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = {row["id"]: row for row in read_output(path)}
            order = [q.id for q in questions if q.dialect == dialect]
            # a question is asked again if it has no answer yet, also when only its reasoning was saved
            todo = [q for q in questions if q.dialect == dialect and not rows.get(q.id, {}).get("predicted", "").strip()]

            with ThreadPoolExecutor(parallel) as pool:
                asked = {}
                for q in todo:
                    prompt = prompt_template.format(question=q.text)
                    asked[pool.submit(model.ask, prompt, THINKING[mode])] = (q, prompt)

                # answers come back in any order; each one is saved as soon as it arrives
                for future in tqdm(as_completed(asked), total=len(asked),
                                   desc=f"{name} thinking={mode} {dialect}", mininterval=30):
                    q, prompt = asked[future]
                    try:
                        answer, reasoning, done_reason = future.result()
                    except Exception as e:
                        print(f"{q.id} failed, will be asked again next time: {e}")
                        continue
                    rows[q.id] = {"id": q.id, "prompt": prompt, "predicted": answer,
                                  "gt_answer": q.answer, "reasoning": reasoning or "",
                                  "done_reason": done_reason or ""}
                    # rewrite the whole file, so it keeps the order of the input file
                    write_output(path, [rows[i] for i in order if i in rows])
