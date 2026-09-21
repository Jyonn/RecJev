"""Create a fixed shuffled-history control for paired binary cases."""

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

from .binary import BinaryCase, load_cases


def shuffled_history(cases: list[BinaryCase], seed: int) -> list[BinaryCase]:
    if len(cases) < 4:
        raise ValueError("Need at least two user pairs")
    pairs = list(zip(cases[::2], cases[1::2]))
    histories = [first.history for first, _ in pairs]
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    # Circularly shift the permutation: each user receives a different user's history.
    assignments = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
    output = []
    for i, (first, second) in enumerate(pairs):
        history = histories[assignments[i]]
        for case in (first, second):
            output.append(BinaryCase(case.id, case.user_id, history,
                                     case.candidate, case.label))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-file", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = shuffled_history(load_cases(args.cases_file, args.limit), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        for case in cases:
            file.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    print(json.dumps({"cases": len(cases), "seed": args.seed,
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
