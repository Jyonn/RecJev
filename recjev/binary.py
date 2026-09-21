"""Paired single-candidate high-rating probability benchmark."""

import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .protocol import Item


def api_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504, 529],
                  allowed_methods=["POST"], respect_retry_after_header=True)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


@dataclass(frozen=True)
class BinaryCase:
    id: str
    user_id: str
    history: tuple[Item, ...]
    candidate: Item
    label: int


def movielens_pairs(path: str | Path, *, sample_size: int = 1000, seed: int = 42,
                    history_size: int = 20, train_fraction: float = 0.8) -> list[BinaryCase]:
    if sample_size < 2 or sample_size % 2 or history_size < 1 or not 0 < train_fraction < 1:
        raise ValueError("sample_size must be positive and even; valid history_size and train_fraction required")
    path = Path(path)
    movies = {}
    with (path / "movies.dat").open(encoding="latin-1") as file:
        for line in file:
            movie, title, _ = line.rstrip("\n").split("::", 2)
            movies[movie] = title
    users: dict[str, list[tuple[int, str, int]]] = {}
    with (path / "ratings.dat").open() as file:
        for line in file:
            user, movie, rating, timestamp = line.rstrip("\n").split("::")
            if movie in movies:
                users.setdefault(user, []).append((int(timestamp), movie, int(rating)))
    eligible = []
    for user in sorted(users, key=int):
        events = sorted(users[user], key=lambda row: (row[0], int(row[1])))
        split = int(len(events) * train_fraction)
        if split < history_size:
            continue
        heldout = events[split:]
        positives = [row for row in heldout if row[2] >= 4]
        negatives = [row for row in heldout if row[2] < 3]
        if positives and negatives:
            eligible.append((user, events[:split], positives, negatives))
    rng = random.Random(seed)
    if len(eligible) < sample_size // 2:
        raise ValueError(f"Only {len(eligible)} users have both held-out labels")
    selected = rng.sample(eligible, sample_size // 2)
    cases = []
    for user, train, positives, negatives in selected:
        history = tuple(Item(movie, movies[movie], rating)
                        for _, movie, rating in train[-history_size:])
        # Randomize the held-out movies and the within-pair order reproducibly.
        pos = rng.choice(positives)
        neg = rng.choice(negatives)
        pair = [(pos, 1), (neg, 0)]
        rng.shuffle(pair)
        for (timestamp, movie, _rating), label in pair:
            cases.append(BinaryCase(f"{user}:{movie}", user, history,
                                    Item(movie, movies[movie]), label))
    return cases


def save_cases(cases: list[BinaryCase], output: str | Path) -> str:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        for case in cases:
            file.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    return hashlib.sha256(output.read_bytes()).hexdigest()


def load_cases(path: str | Path, limit: int | None = None) -> list[BinaryCase]:
    if limit is not None and (limit < 2 or limit % 2):
        raise ValueError("limit must be positive and even")
    cases = []
    with Path(path).open() as file:
        for line in file:
            if limit is not None and len(cases) >= limit:
                break
            row = json.loads(line)
            cases.append(BinaryCase(str(row["id"]), str(row["user_id"]),
                                    tuple(Item(**item) for item in row["history"]),
                                    Item(**row["candidate"]), int(row["label"])))
    if not cases or len(cases) % 2 or len({case.id for case in cases}) != len(cases):
        raise ValueError("Need unique, nonempty paired cases")
    for first, second in zip(cases[::2], cases[1::2]):
        if first.user_id != second.user_id or {first.label, second.label} != {0, 1} or first.history != second.history:
            raise ValueError("Each adjacent pair must share a user and history and contain one positive and one negative")
    return cases


def context(case: BinaryCase) -> str:
    history = "\n".join(f"- {item.title} (rating {item.rating}/5)" for item in case.history)
    return f"Previously rated movies:\n{history}\n\nCandidate movie: {case.candidate.title}"


class JevProbability:
    def __init__(self, key: str, endpoint: str, model: str):
        self.key, self.endpoint, self.model = key, endpoint, model
        self.session = api_session()
        self.last_usage = None
        self.last_response_model = None

    def predict(self, case: BinaryCase) -> float:
        self.last_usage = self.last_response_model = None
        response = self.session.post(self.endpoint,
            headers={"Authorization": f"Bearer {self.key}"},
            json={"model": self.model, "state": context(case),
                  "questions": {"liked": {"type": "noul", "instructions":
                    "Would this user rate the candidate movie 4 or 5 out of 5, based on their prior ratings?"}}}, timeout=60)
        response.raise_for_status()
        body = response.json()
        self.last_usage = body.get("usage")
        self.last_response_model = body.get("model")
        return float(body["answers"]["liked"]["noul"])


class LLMProbability:
    def __init__(self, key: str, base_url: str, model: str,
                 omit_temperature: bool = False, disable_thinking: bool = False):
        self.key, self.base_url, self.model = key, base_url.rstrip("/"), model
        self.omit_temperature, self.disable_thinking = omit_temperature, disable_thinking
        self.session = api_session()
        self.last_usage = None
        self.last_response_model = None

    def predict(self, case: BinaryCase) -> float:
        self.last_usage = self.last_response_model = None
        payload = {"model": self.model, "messages": [
            {"role": "system", "content": "Estimate the probability from 0 to 1 that this user would rate the candidate movie 4 or 5 out of 5. Return only a JSON object of the form {\"probability\": 0.5}."},
            {"role": "user", "content": context(case)}]}
        if not self.omit_temperature:
            payload["temperature"] = 0
        if self.disable_thinking:
            payload["thinking"] = {"type": "disabled"}
        response = self.session.post(f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"}, json=payload, timeout=60)
        response.raise_for_status()
        body = response.json()
        self.last_usage = body.get("usage")
        self.last_response_model = body.get("model")
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Model did not return text")
        match = re.search(r"\{[\s\S]*?\}", content)
        if not match:
            raise ValueError(f"Expected JSON object, got {content[:200]}")
        return float(json.loads(match.group())["probability"])


def summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["error"] is None]
    pairs = list(zip(rows[::2], rows[1::2]))
    comparable = [(a, b) for a, b in pairs if a["error"] is None and b["error"] is None]
    wins = 0.0
    for first, second in comparable:
        positive, negative = (first, second) if first["label"] else (second, first)
        if positive["probability"] > negative["probability"]:
            wins += 1
        elif positive["probability"] == negative["probability"]:
            wins += 0.5
    latency = sorted(row["latency_s"] for row in valid)
    return {"cases": len(rows), "users": len(pairs), "failures": len(rows)-len(valid),
            "pairwise_accuracy": wins/len(comparable) if comparable else None,
            "comparable_pairs": len(comparable),
            "brier": statistics.mean((r["probability"]-r["label"])**2 for r in valid) if valid else None,
            "mean_positive_probability": statistics.mean(r["probability"] for r in valid if r["label"] == 1) if any(r["label"] == 1 for r in valid) else None,
            "mean_negative_probability": statistics.mean(r["probability"] for r in valid if r["label"] == 0) if any(r["label"] == 0 for r in valid) else None,
            "median_latency_s": statistics.median(latency) if latency else None,
            "p95_latency_s": latency[math.ceil(.95*len(latency))-1] if latency else None}


def evaluate(cases: list[BinaryCase], model, output: str | Path,
             *, rate: dict | None = None, resume: bool = False, config: dict | None = None) -> dict:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = output.with_suffix(output.suffix + ".meta.json")
    fingerprint = {"case_sha256": hashlib.sha256(json.dumps([asdict(c) for c in cases],sort_keys=True).encode()).hexdigest(),
                   "model": model.model, "rate": rate, "config": config or {}}
    rows = []
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        if not meta.exists() or json.loads(meta.read_text()) != fingerprint:
            raise ValueError("Run settings or cases differ from existing results")
        with output.open("rb+") as file:
            while True:
                start = file.tell()
                line = file.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    file.truncate(start)
                    break
                row = json.loads(line)
                index = len(rows)
                if index >= len(cases) or row["case_id"] != cases[index].id or row["label"] != cases[index].label:
                    raise ValueError("Existing rows do not match case prefix")
                rows.append(row)
    else:
        meta.write_text(json.dumps(fingerprint,indent=2)+"\n")
    with output.open("a") as file:
        for case in cases[len(rows):]:
            start = time.perf_counter()
            try:
                probability = float(model.predict(case))
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("Probability must be finite and between 0 and 1")
                error = None
            except requests.RequestException:
                raise
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                probability, error = None, str(exc)
            row = {"case_id": case.id, "user_id": case.user_id, "label": case.label,
                   "probability": probability, "error": error,
                   "latency_s": time.perf_counter()-start, "requested_model": model.model,
                   "response_model": getattr(model,"last_response_model",None),
                   "usage": getattr(model,"last_usage",None), "published_rate": rate}
            file.write(json.dumps(row)+"\n")
            file.flush();os.fsync(file.fileno())
            rows.append(row)
    return summarize(rows) | {"predictions":str(output)}
