"""Export a fixed, inspectable MovieLens evaluation set without API calls."""

import argparse
import hashlib
import json
from pathlib import Path

from .movielens import MovieLens1M


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--history-size", type=int, default=20)
    parser.add_argument("--negatives", choices=["unseen", "rated-low"], default="unseen")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--output", default="data/ml-1m/eval-1000-seed42.jsonl")
    args = parser.parse_args()
    cases = MovieLens1M(args.data).cases(sample_size=args.sample_size, seed=args.seed,
                                          candidates=args.candidates, history_size=args.history_size,
                                          negatives=args.negatives, train_fraction=args.train_fraction)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        for case in cases:
            file.write(json.dumps({"case_id": case.id,
                                   "history": [vars(item) for item in case.history],
                                   "candidates": [vars(item) for item in case.candidates],
                                   "positive_id": case.positive_id}, ensure_ascii=False) + "\n")
    print(json.dumps({"cases": len(cases), "seed": args.seed, "candidates": args.candidates,
                      "history_size": args.history_size, "negatives": args.negatives,
                      "train_fraction": args.train_fraction, "output": str(output),
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
