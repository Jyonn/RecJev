"""Simple non-API baselines for candidate ranking."""

from collections import Counter
from pathlib import Path

from .protocol import Case


class Popularity:
    model = "popularity"

    def __init__(self, data_path: str | Path, min_rating: int = 4):
        users: dict[str, list[tuple[int, str, int]]] = {}
        with (Path(data_path) / "ratings.dat").open() as file:
            for line in file:
                user, movie, rating, timestamp = line.rstrip("\n").split("::")
                users.setdefault(user, []).append((int(timestamp), movie, int(rating)))
        self.counts: Counter[str] = Counter()
        for events in users.values():
            ordered = sorted(events, key=lambda event: (event[0], int(event[1])))
            positives = [i for i, event in enumerate(ordered)
                         if i > 0 and event[2] >= min_rating]
            if positives:
                ordered = ordered[:positives[-1]]  # Exclude this user's held-out target.
            self.counts.update(movie for _, movie, rating in ordered if rating >= min_rating)

    def rank(self, case: Case, top_k: int) -> list[str]:
        return [item.id for item in sorted(case.candidates,
                    key=lambda item: (-self.counts[item.id], int(item.id)))[:top_k]]
