"""Merge aligned binary benchmark outputs into analysis-friendly files."""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--prefix", default="all-models")
    args = parser.parse_args()

    model_rows: dict[str, dict[str, dict]] = {}
    order: list[str] | None = None
    reference: dict[str, tuple[str, int]] = {}
    for model in args.models:
        path = args.results_dir / f"{model}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids = [row["case_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate case_id in {path}")
        if order is None:
            order = ids
            reference = {row["case_id"]: (str(row["user_id"]), int(row["label"])) for row in rows}
        elif set(ids) != set(order):
            missing = set(order) - set(ids)
            extra = set(ids) - set(order)
            raise ValueError(f"Case mismatch for {model}: missing={len(missing)}, extra={len(extra)}")
        for row in rows:
            expected = reference[row["case_id"]]
            actual = (str(row["user_id"]), int(row["label"]))
            if actual != expected:
                raise ValueError(f"Label/user mismatch for {model}, case {row['case_id']}")
        model_rows[model] = {row["case_id"]: row for row in rows}

    assert order is not None
    nested_path = args.results_dir / f"{args.prefix}-cases.jsonl"
    wide_path = args.results_dir / f"{args.prefix}-cases.csv"
    with nested_path.open("w") as nested, wide_path.open("w", newline="") as wide:
        columns = ["case_id", "user_id", "label"]
        for model in args.models:
            columns.extend([f"{model}.probability", f"{model}.latency_s", f"{model}.error"])
        writer = csv.DictWriter(wide, fieldnames=columns)
        writer.writeheader()
        for case_id in order:
            user_id, label = reference[case_id]
            models = {model: model_rows[model][case_id] for model in args.models}
            nested.write(json.dumps({"case_id": case_id, "user_id": user_id,
                                     "label": label, "models": models}) + "\n")
            flat = {"case_id": case_id, "user_id": user_id, "label": label}
            for model, row in models.items():
                flat[f"{model}.probability"] = row.get("probability")
                flat[f"{model}.latency_s"] = row.get("latency_s")
                flat[f"{model}.error"] = row.get("error")
            writer.writerow(flat)

    print(json.dumps({"cases": len(order), "models": len(args.models),
                      "jsonl": str(nested_path), "csv": str(wide_path)}, indent=2))


if __name__ == "__main__":
    main()
