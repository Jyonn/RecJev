import json
import re

import requests

from .protocol import Case, validate_ranking


def _context(case: Case) -> str:
    history = "\n".join(f"- {item.title}" for item in case.history)
    options = "\n".join(f"{item.id}: {item.title}" for item in case.candidates)
    return f"Previously watched movies:\n{history}\n\nCandidate movies:\n{options}\n\nPreferences or constraints: {case.constraints or 'none'}"


class OhMyGPT:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 60,
                 temperature: float | None = 0):
        self.api_key, self.base_url, self.model, self.timeout = api_key, base_url.rstrip("/"), model, timeout
        self.temperature = temperature

    def rank(self, case: Case, top_k: int) -> list[str]:
        ids = [item.id for item in case.candidates]
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": f"Recommend movies. Return only a JSON array of exactly {top_k} distinct candidate IDs, best first. Allowed IDs: {', '.join(ids)}."},
                                {"role": "user", "content": _context(case)}]}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Model did not return text")
        match = re.search(r"\[[\s\S]*?\]", content)
        if not match:
            raise ValueError(f"Expected JSON array, got: {content[:200]}")
        return validate_ranking(case, [str(item) for item in json.loads(match.group())], top_k)


class Jev:
    def __init__(self, api_key: str, endpoint: str, model: str, timeout: float = 60):
        self.api_key, self.endpoint, self.model, self.timeout = api_key, endpoint, model, timeout

    def rank(self, case: Case, top_k: int) -> list[str]:
        criteria = {item.id: item.title for item in case.candidates}
        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "state": _context(case),
                  "questions": {"recommend": {"type": "choice", "instructions": "Which candidate movie is most likely to suit this user's past preferences?", "criteria": criteria}}},
            timeout=self.timeout,
        )
        response.raise_for_status()
        probabilities = response.json()["answers"]["recommend"]["probabilities"]
        if set(probabilities) != set(criteria):
            raise ValueError("Jev returned a different candidate set")
        ranking = sorted(criteria, key=lambda item_id: (-probabilities[item_id], list(criteria).index(item_id)))[:top_k]
        return validate_ranking(case, ranking, top_k)
