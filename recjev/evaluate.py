import json
import math
import time
from pathlib import Path
import requests

from .protocol import Case, Recommender, validate_ranking


def evaluate(cases: list[Case], model: Recommender, top_k: int, output: str | Path,
             rate: dict | None = None) -> dict:
    if not cases or top_k < 1 or any(top_k > len(case.candidates) for case in cases):
        raise ValueError("Need nonempty cases and 1 <= top_k <= candidate count")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    hits = ndcg = successful_latency = failures = 0
    with output.open("w") as file:
        for case in cases:
            start = time.perf_counter()
            try:
                ranking = validate_ranking(case, model.rank(case, top_k), top_k)
                error = None
            except (ValueError, KeyError, TypeError, RuntimeError, requests.RequestException) as exc:
                ranking, error = [], str(exc)
                failures += 1
            elapsed = time.perf_counter() - start
            if error is None:
                successful_latency += elapsed
            if case.positive_id in ranking:
                hits += 1
                ndcg += 1 / math.log2(ranking.index(case.positive_id) + 2)
            file.write(json.dumps({"case_id": case.id, "positive_id": case.positive_id,
                                   "ranking": ranking, "latency_s": elapsed, "error": error,
                                   "requested_model": getattr(model, "model", None),
                                   "response_model": getattr(model, "last_response_model", None),
                                   "usage": getattr(model, "last_usage", None),
                                   "published_rate": rate}) + "\n")
    return {"cases": len(cases), "failures": failures, "hit_at_k": hits / len(cases),
            "ndcg_at_k": ndcg / len(cases),
            "mean_latency_s": successful_latency / (len(cases) - failures) if failures < len(cases) else None,
            "predictions": str(output)}
