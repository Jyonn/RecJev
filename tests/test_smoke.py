import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recjev.evaluate import evaluate
from recjev.models import Jev, OhMyGPT
from recjev.movielens import MovieLens1M
from recjev.protocol import Case, Item


class FirstCandidate:
    def rank(self, case, top_k):
        return [item.id for item in case.candidates[:top_k]]


class SmokeTest(unittest.TestCase):
    def test_provider_request_and_ranking(self):
        case = Case("u", (Item("1", "History"),),
                    (Item("2", "Alpha"), Item("3", "Beta")), "3")
        with patch("recjev.models.requests.post") as post:
            post.return_value.json.return_value = {"choices": [{"message": {"content": '["3", "2"]'}}]}
            self.assertEqual(OhMyGPT("key", "https://example.test/v1", "model").rank(case, 2), ["3", "2"])
            self.assertEqual(post.call_args.kwargs["json"]["model"], "model")
            self.assertEqual(OhMyGPT("key", "https://example.test/v1", "model", temperature=None).rank(case, 2), ["3", "2"])
            self.assertNotIn("temperature", post.call_args.kwargs["json"])
        with patch("recjev.models.requests.post") as post:
            post.return_value.json.return_value = {"answers": {"recommend": {"probabilities": {"2": 0.2, "3": 0.8}}}}
            self.assertEqual(Jev("key", "https://example.test/systemone", "jev").rank(case, 2), ["3", "2"])
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


if __name__ == "__main__":
    unittest.main()
