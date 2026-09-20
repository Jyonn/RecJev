import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import requests

from .protocol import Case, Recommender, validate_ranking


def _fingerprint(cases: list[Case], model: Recommender, top_k: int, rate: dict | None,
                 run_config: dict | None) -> dict:
    content = json.dumps([asdict(case) for case in cases], sort_keys=True).encode()
    return {"case_sha256": hashlib.sha256(content).hexdigest(), "case_count": len(cases),
            "model": getattr(model, "model", None), "top_k": top_k,
            "published_rate": rate, "run_config": run_config or {}}


def _completed_rows(output: Path, cases: list[Case], model: Recommender) -> list[dict]:
    rows = []
    with output.open("rb+") as file:
        while True:
            start = file.tell()
            line = file.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                file.truncate(start)  # Interrupted write of the final record.
                break
            row = json.loads(line)
            index = len(rows)
            if index >= len(cases) or row.get("case_id") != cases[index].id or row.get("positive_id") != cases[index].positive_id or row.get("requested_model") != getattr(model, "model", None):
                raise ValueError(f"Existing results do not match the selected case prefix at row {index + 1}")
            rows.append(row)
    return rows


def evaluate(cases: list[Case], model: Recommender, top_k: int, output: str | Path,
             rate: dict | None = None, resume: bool = False,
             run_config: dict | None = None) -> dict:
    if not cases or top_k < 1 or any(top_k > len(case.candidates) for case in cases):
        raise ValueError("Need nonempty cases and 1 <= top_k <= candidate count")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.with_suffix(output.suffix + ".meta.json")
    fingerprint = _fingerprint(cases, model, top_k, rate, run_config)
    if resume and output.exists():
        if not manifest.exists() or json.loads(manifest.read_text()) != fingerprint:
            raise ValueError("Run settings or case set differ from the existing results")
        rows = _completed_rows(output, cases, model)
    else:
        if output.exists() and not resume:
            raise FileExistsError(f"Results already exist: {output}; use --resume or a new output path")
        manifest.write_text(json.dumps(fingerprint, indent=2) + "\n")
        rows = []
    with output.open("a") as file:
        for case in cases[len(rows):]:
            start = time.perf_counter()
            try:
                ranking = validate_ranking(case, model.rank(case, top_k), top_k)
                error = None
            except requests.RequestException:
                # A transport or provider failure did not produce a model answer.
                # Leave this case pending so --resume can retry it later.
                raise
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                ranking, error = [], str(exc)
            row = {"case_id": case.id, "positive_id": case.positive_id,
                   "ranking": ranking, "latency_s": time.perf_counter() - start, "error": error,
                   "requested_model": getattr(model, "model", None),
                   "response_model": getattr(model, "last_response_model", None),
                   "usage": getattr(model, "last_usage", None), "published_rate": rate}
            file.write(json.dumps(row) + "\n")
            file.flush()
            os.fsync(file.fileno())
            rows.append(row)
    successful = [row["latency_s"] for row in rows if row["error"] is None]
    ordered = sorted(successful)
    hits = [row for row in rows if row["positive_id"] in row["ranking"]]
    ndcg = sum(1 / math.log2(row["ranking"].index(row["positive_id"]) + 2) for row in hits)
    return {"cases": len(rows), "failures": len(rows) - len(successful),
            "hit_at_k": len(hits) / len(rows), "ndcg_at_k": ndcg / len(rows),
            "mean_latency_s": statistics.mean(successful) if successful else None,
            "median_latency_s": statistics.median(successful) if successful else None,
            "p95_latency_s": ordered[math.ceil(.95 * len(ordered)) - 1] if ordered else None,
            "predictions": str(output)}
