import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from recjev.evaluate import evaluate
from recjev.models import Jev, OhMyGPT
from recjev.movielens import MovieLens1M
from recjev.protocol import Case, Item


class FirstCandidate:
    def rank(self, case, top_k):
        return [item.id for item in case.candidates[:top_k]]


class FailingModel:
    def rank(self, case, top_k):
        raise RuntimeError("unavailable")


class SmokeTest(unittest.TestCase):
    def test_provider_request_and_ranking(self):
        case = Case("u", (Item("1", "History"),),
                    (Item("2", "Alpha"), Item("3", "Beta")), "3")
        with patch("recjev.models.requests.post") as post:
            post.return_value.json.return_value = {"model": "model-version", "usage": {"prompt_tokens": 10, "completion_tokens": 2}, "choices": [{"message": {"content": '["3", "2"]'}}]}
            llm = OhMyGPT("key", "https://example.test/v1", "model")
            self.assertEqual(llm.rank(case, 2), ["3", "2"])
            self.assertEqual(llm.last_usage["prompt_tokens"], 10)
            self.assertEqual(llm.last_response_model, "model-version")
            self.assertEqual(post.call_args.kwargs["json"]["model"], "model")
            self.assertEqual(OhMyGPT("key", "https://example.test/v1", "model", temperature=None).rank(case, 2), ["3", "2"])
            self.assertNotIn("temperature", post.call_args.kwargs["json"])
        with patch("recjev.models.requests.post") as post:
            post.return_value.json.return_value = {"model": "jev-version", "usage": {"input_tokens": 12}, "answers": {"recommend": {"probabilities": {"2": 0.2, "3": 0.8}}}}
            jev = Jev("key", "https://example.test/systemone", "jev")
            self.assertEqual(jev.rank(case, 2), ["3", "2"])
            self.assertEqual(jev.last_usage["input_tokens"], 12)
            self.assertEqual(post.call_args.kwargs["json"]["questions"]["recommend"]["type"], "choice")

    def test_reproducible_tasks_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "movies.dat").write_text("1::One (2000)::Drama\n2::Two (2000)::Drama\n3::Three (2000)::Drama\n4::Four (2000)::Drama\n5::Five (2000)::Drama\n")
            (root / "ratings.dat").write_text("1::1::5::1\n1::2::5::2\n2::3::5::1\n2::4::5::2\n")
            adapter = MovieLens1M(root)
            a = adapter.cases(sample_size=1, seed=7, candidates=2, history_size=1)
            b = adapter.cases(sample_size=1, seed=7, candidates=2, history_size=1)
            self.assertEqual(a, b)
            self.assertEqual(len(a), 1)
            report = evaluate(a, FirstCandidate(), 2, root / "predictions.jsonl")
            self.assertEqual(report["hit_at_k"], 1)
            self.assertTrue((root / "predictions.jsonl").exists())

    def test_prediction_records_usage_and_rate(self):
        case = Case("u", (Item("1", "History"),),
                    (Item("2", "Alpha"), Item("3", "Beta")), "3")
        with tempfile.TemporaryDirectory() as tmp, patch("recjev.models.requests.post") as post:
            post.return_value.json.return_value = {"model": "model-version", "usage": {"prompt_tokens": 10}, "choices": [{"message": {"content": '["3"]'}}]}
            model = OhMyGPT("key", "https://example.test/v1", "model")
            path = Path(tmp) / "predictions.jsonl"
            evaluate([case], model, 1, path, rate={"currency": "USD", "input": 1})
            row = json.loads(path.read_text())
            self.assertEqual(row["usage"], {"prompt_tokens": 10})
            self.assertEqual(row["published_rate"]["currency"], "USD")
            self.assertEqual(row["response_model"], "model-version")

            failed = evaluate([case], FailingModel(), 1, Path(tmp) / "failed.jsonl")
            self.assertEqual(failed["failures"], 1)
            self.assertIsNone(failed["mean_latency_s"])


if __name__ == "__main__":
    unittest.main()
