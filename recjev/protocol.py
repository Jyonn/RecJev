from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Item:
    id: str
    title: str


@dataclass(frozen=True)
class Case:
    id: str
    history: tuple[Item, ...]
    candidates: tuple[Item, ...]
    positive_id: str
    # Future datasets may populate explicit preferences or hard constraints.
    constraints: str = ""


class DatasetAdapter(Protocol):
    def cases(self, *, sample_size: int | None, seed: int,
              candidates: int, history_size: int) -> list[Case]: ...


class Recommender(Protocol):
    def rank(self, case: Case, top_k: int) -> list[str]: ...


def validate_ranking(case: Case, ranking: list[str], top_k: int) -> list[str]:
    allowed = {item.id for item in case.candidates}
    if len(ranking) != top_k or len(set(ranking)) != top_k or not set(ranking) <= allowed:
        raise ValueError(f"Invalid ranking for case {case.id}: {ranking}")
    return ranking
