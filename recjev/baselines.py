"""Simple non-API baselines for candidate ranking."""

from collections import Counter
import math
from pathlib import Path

from .protocol import Case


class Popularity:
    model = "popularity"

    def __init__(self, data_path: str | Path, min_rating: int = 4,
                 train_fraction: float | None = None):
        users: dict[str, list[tuple[int, str, int]]] = {}
        with (Path(data_path) / "ratings.dat").open() as file:
            for line in file:
                user, movie, rating, timestamp = line.rstrip("\n").split("::")
                users.setdefault(user, []).append((int(timestamp), movie, int(rating)))
        self.counts: Counter[str] = Counter()
        for events in users.values():
            ordered = sorted(events, key=lambda event: (event[0], int(event[1])))
            if train_fraction is not None:
                ordered = ordered[:int(len(ordered) * train_fraction)]
            else:
                positives = [i for i, event in enumerate(ordered)
                             if i > 0 and event[2] >= min_rating]
                if positives:
                    ordered = ordered[:positives[-1]]
            self.counts.update(movie for _, movie, rating in ordered if rating >= min_rating)

    def rank(self, case: Case, top_k: int) -> list[str]:
        return [item.id for item in sorted(case.candidates,
                    key=lambda item: (-self.counts[item.id], int(item.id)))[:top_k]]


class CollaborativeData:
    """Positive interactions from each user's training prefix only."""

    def __init__(self, data_path: str | Path, train_fraction: float = 0.8,
                 min_rating: int = 4):
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        events: dict[str, list[tuple[int, str, int]]] = {}
        with (Path(data_path) / "ratings.dat").open() as file:
            for line in file:
                user, movie, rating, timestamp = line.rstrip("\n").split("::")
                events.setdefault(user, []).append((int(timestamp), movie, int(rating)))
        self.user_items: dict[str, set[str]] = {}
        self.item_users: dict[str, set[str]] = {}
        for user, ratings in events.items():
            ordered = sorted(ratings, key=lambda row: (row[0], int(row[1])))
            liked = {movie for _, movie, rating in ordered[:int(len(ordered) * train_fraction)]
                     if rating >= min_rating}
            self.user_items[user] = liked
            for movie in liked:
                self.item_users.setdefault(movie, set()).add(user)


class ItemKNN:
    """Cosine similarity of positive item interactions, with top-N neighbors."""

    model = "itemknn"

    def __init__(self, data_path: str | Path, train_fraction: float = 0.8,
                 neighbors: int = 50):
        self.data = CollaborativeData(data_path, train_fraction)
        self.neighbors = neighbors

    def rank(self, case: Case, top_k: int) -> list[str]:
        liked = {item.id for item in case.history if item.rating is None or item.rating >= 4}
        scores = {}
        for item in case.candidates:
            users = self.data.item_users.get(item.id, set())
            sims = []
            for past in liked:
                other = self.data.item_users.get(past, set())
                if users and other:
                    sims.append(len(users & other) / math.sqrt(len(users) * len(other)))
            scores[item.id] = sum(sorted(sims, reverse=True)[:self.neighbors])
        return [item.id for item in sorted(case.candidates,
                    key=lambda item: (-scores[item.id], int(item.id)))[:top_k]]


class UserKNN:
    """Cosine nearest users on positive training interactions."""

    model = "userknn"

    def __init__(self, data_path: str | Path, train_fraction: float = 0.8,
                 neighbors: int = 50):
        self.data = CollaborativeData(data_path, train_fraction)
        self.neighbors = neighbors

    def rank(self, case: Case, top_k: int) -> list[str]:
        liked = {item.id for item in case.history if item.rating is None or item.rating >= 4}
        overlap: Counter[str] = Counter()
        for movie in liked:
            overlap.update(self.data.item_users.get(movie, set()))
        overlap.pop(case.id, None)
        similar = sorted(((shared / math.sqrt(len(liked) * len(self.data.user_items[user])), user)
                          for user, shared in overlap.items()), reverse=True)[:self.neighbors] if liked else []
        scores = {item.id: sum(sim for sim, user in similar
                               if item.id in self.data.user_items[user]) for item in case.candidates}
        return [item.id for item in sorted(case.candidates,
                    key=lambda item: (-scores[item.id], int(item.id)))[:top_k]]
