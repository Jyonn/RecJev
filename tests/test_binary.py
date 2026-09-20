import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recjev.binary import (BinaryCase, JevProbability, LLMProbability,
                           evaluate, load_cases, movielens_pairs, save_cases)
from recjev.protocol import Item


class BinaryTest(unittest.TestCase):
    def test_pairs_are_reproducible_and_hide_candidate_ratings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "movies.dat").write_text("".join(
                f"{i}::Movie {i}::Drama\n" for i in range(1, 11)))
            (root / "ratings.dat").write_text("".join(
                f"{user}::{movie}::{5 if movie == 9 else 1}::{movie}\n"
                for user in range(1, 3) for movie in range(1, 11)))
            cases = movielens_pairs(root, sample_size=4, seed=4,
                                    history_size=2, train_fraction=.8)
            self.assertEqual(cases, movielens_pairs(root, sample_size=4, seed=4,
                                                      history_size=2, train_fraction=.8))
            self.assertTrue(all({a.label,b.label} == {0,1}
                                for a,b in zip(cases[::2],cases[1::2])))
            self.assertTrue(all(c.candidate.rating is None for c in cases))
            path = root / "cases.jsonl"
            save_cases(cases,path)
            self.assertEqual(load_cases(path,2),cases[:2])

    def test_probability_requests_and_pairwise_metric(self):
        cases = [BinaryCase("u:2","u",(Item("1","One",5),),Item("2","Two"),1),
                 BinaryCase("u:3","u",(Item("1","One",5),),Item("3","Three"),0)]
        with patch("recjev.binary.requests.Session.post") as post:
            post.return_value.json.return_value = {"model":"jev","usage":{},"answers":{"liked":{"noul":.8}}}
            model = JevProbability("key","https://example.test","jev")
            self.assertEqual(model.predict(cases[0]),.8)
            self.assertEqual(post.call_args.kwargs["json"]["questions"]["liked"]["type"],"noul")
            post.return_value.json.return_value = {"model":"llm","usage":{},"choices":[{"message":{"content":"{\"probability\": 0.7}"}}]}
            llm = LLMProbability("key","https://example.test/v1","llm")
            self.assertEqual(llm.predict(cases[0]),.7)
            self.assertNotIn("rating",post.call_args.kwargs["json"]["messages"][1]["content"].split("Candidate movie:")[1])
        class Fixed:
            model="fixed"
            def predict(self,case):
                return .8 if case.label else .2
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"out.jsonl"
            result = evaluate(cases,Fixed(),path)
            self.assertEqual(result["pairwise_accuracy"],1)
            self.assertAlmostEqual(result["brier"],.04)
            self.assertEqual(evaluate(cases,Fixed(),path,resume=True)["cases"],2)
            self.assertEqual(len(path.read_text().splitlines()),2)


if __name__ == "__main__":
    unittest.main()
