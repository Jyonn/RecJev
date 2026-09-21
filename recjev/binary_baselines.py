"""Leakage-safe classical baselines for the paired MovieLens rating task."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .baselines import CollaborativeData
from .binary import load_cases, summarize


def predictions(data_path: str | Path, cases_file: str | Path) -> dict[str, list[dict]]:
    cases = load_cases(cases_file)
    data = CollaborativeData(data_path, train_fraction=0.8)
    popularity = Counter(movie for liked in data.user_items.values() for movie in liked)
    output = {name: [] for name in ("popularity", "itemknn", "userknn")}
    for case in cases:
        candidate = case.candidate.id
        liked = {item.id for item in case.history if item.rating is not None and item.rating >= 4}
        candidate_users = data.item_users.get(candidate, set())
        item_sims = []
        overlap = Counter()
        for movie in liked:
            other_users = data.item_users.get(movie, set())
            if candidate_users and other_users:
                item_sims.append(len(candidate_users & other_users) /
                                 math.sqrt(len(candidate_users) * len(other_users)))
            overlap.update(other_users)
        # The target user's own prefix is allowed as context, never as a neighbor.
        overlap.pop(case.user_id, None)
        neighbors = sorted((shared / math.sqrt(len(liked) * len(data.user_items[user])), user)
                           for user, shared in overlap.items() if liked and data.user_items[user])[-50:]
        scores = {
            "popularity": float(popularity[candidate]),
            "itemknn": sum(sorted(item_sims, reverse=True)[:50]),
            "userknn": sum(sim for sim, user in neighbors
                           if candidate in data.user_items[user]),
        }
        for name, score in scores.items():
            output[name].append({"case_id": case.id, "user_id": case.user_id,
                                 "label": case.label, "probability": score,
                                 "error": None, "latency_s": 0.0})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cases-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, rows in predictions(args.data, args.cases_file).items():
        path = output_dir / f"{name}.jsonl"
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        metrics = summarize(rows)
        # These are uncalibrated ranking scores; probability and timing fields
        # in the shared evaluator have no meaning for this offline computation.
        report[name] = {key: metrics[key] for key in
                        ("cases", "users", "failures", "pairwise_accuracy", "comparable_pairs")}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
