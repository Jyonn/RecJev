"""Read an exported, fixed evaluation set."""

import json
from pathlib import Path

from .protocol import Case, Item


def load_cases(path: str | Path, limit: int | None = None) -> list[Case]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    cases = []
    with Path(path).open() as file:
        for line in file:
            if limit is not None and len(cases) >= limit:
                break
            row = json.loads(line)
            cases.append(Case(str(row["case_id"]),
                              tuple(Item(**item) for item in row["history"]),
                              tuple(Item(**item) for item in row["candidates"]),
                              str(row["positive_id"]), row.get("constraints", "")))
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("cases file must contain unique, nonempty cases")
    return cases
