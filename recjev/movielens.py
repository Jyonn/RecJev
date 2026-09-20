import random
from pathlib import Path

from .protocol import Case, Item


class MovieLens1M:
    """Leave-last-positive-out tasks from the MovieLens 1M .dat files."""

    def __init__(self, path: str | Path, min_rating: int = 4):
        self.path = Path(path)
        self.min_rating = min_rating

    def cases(self, *, sample_size: int | None, seed: int,
              candidates: int, history_size: int,
              negatives: str = "unseen", train_fraction: float = 0.8) -> list[Case]:
        if candidates < 2 or history_size < 1 or sample_size is not None and sample_size < 1:
            raise ValueError("candidates >= 2, history_size >= 1, sample_size >= 1 required")
        if negatives not in {"unseen", "rated-low"} or not 0 < train_fraction < 1:
            raise ValueError("negatives must be unseen or rated-low; train_fraction must be between 0 and 1")
        movies = {}
        with (self.path / "movies.dat").open(encoding="latin-1") as file:
            for line in file:
                movie_id, title, _genres = line.rstrip("\n").split("::", 2)
                movies[movie_id] = Item(movie_id, title)
        users: dict[str, list[tuple[int, str, int]]] = {}
        with (self.path / "ratings.dat").open() as file:
            for line in file:
                user, movie, rating, timestamp = line.rstrip("\n").split("::")
                if movie in movies:
                    users.setdefault(user, []).append((int(timestamp), movie, int(rating)))
        rng = random.Random(seed)
        tasks = []
        all_ids = set(movies)
        for user in sorted(users, key=int):
            events = sorted(users[user], key=lambda event: (event[0], int(event[1])))
            split = int(len(events) * train_fraction) if negatives == "rated-low" else 0
            positives = [i for i, event in enumerate(events)
                         if event[2] >= self.min_rating and i >= max(split, 1)]
            if not positives:
                continue
            target_index = positives[-1]
            history_end = split if negatives == "rated-low" else target_index
            history_events = events[:history_end][-history_size:]
            positive = events[target_index][1]
            if negatives == "rated-low":
                pool = sorted({movie for _, movie, rating in events[split:]
                               if rating < 3 and movie != positive}, key=int)
            else:
                pool = sorted(all_ids - {movie for _, movie, _ in events}, key=int)
            if len(pool) < candidates - 1:
                continue
            candidate_ids = rng.sample(pool, candidates - 1) + [positive]
            rng.shuffle(candidate_ids)
            history = tuple(Item(movie, movies[movie].title, rating if negatives == "rated-low" else None)
                            for _, movie, rating in history_events)
            tasks.append(Case(user, history,
                              tuple(movies[i] for i in candidate_ids), positive))
        if sample_size is not None and sample_size < len(tasks):
            tasks = rng.sample(tasks, sample_size)
        return tasks
