import json
import tempfile
import unittest
from pathlib import Path

from recjev.binary import BinaryCase, movielens_pairs, save_cases
from recjev.finetune_data import build
from recjev.history_control import shuffled_history
from recjev.protocol import Item


class FinetuneDataTest(unittest.TestCase):
    def test_user_disjoint_balanced_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "movies.dat").write_text("".join(
                f"{movie}::Movie {movie}::Drama\n" for movie in range(1, 31)))
            (root / "ratings.dat").write_text("".join(
                f"{user}::{movie}::{5 if movie % 2 else 1}::{movie}\n"
                for user in range(1, 11) for movie in range(1, 31)))
            cases = root / "cases.jsonl"
            save_cases(movielens_pairs(root, sample_size=2, history_size=2, seed=42), cases)
            first = build(root, cases, root / "first", history_size=2,
                          max_pairs_per_user=2)
            second = build(root, cases, root / "second", history_size=2,
                           max_pairs_per_user=2)
            self.assertEqual(first["files_sha256"], second["files_sha256"])
            train = [json.loads(line) for line in (root / "first/train.jsonl").read_text().splitlines()]
            validation = [json.loads(line) for line in (root / "first/validation.jsonl").read_text().splitlines()]
            test_users = {case.user_id for case in movielens_pairs(
                root, sample_size=2, history_size=2, seed=42)}
            self.assertFalse({row["user_id"] for row in train} & test_users)
            self.assertFalse({row["user_id"] for row in validation} & test_users)
            self.assertFalse({row["user_id"] for row in train} &
                             {row["user_id"] for row in validation})
            self.assertEqual(sum(row["label"] for row in train), len(train)//2)
            self.assertEqual(sum(row["label"] for row in validation), len(validation)//2)
            self.assertTrue(all(row["answer"] == ("B" if row["label"] else "A")
                                for row in train + validation))

    def test_shuffled_history_keeps_targets_and_changes_each_user_history(self):
        cases = []
        for user in range(4):
            history = (Item(str(user), f"History {user}", 5),)
            for label in (1, 0):
                cases.append(BinaryCase(f"{user}:{label}", str(user), history,
                                        Item(f"candidate-{label}", f"Candidate {label}"), label))
        shuffled = shuffled_history(cases, 42)
        self.assertEqual(shuffled, shuffled_history(cases, 42))
        self.assertEqual([(x.id, x.label, x.candidate) for x in cases],
                         [(x.id, x.label, x.candidate) for x in shuffled])
        self.assertTrue(all(a.history != b.history for a, b in zip(cases, shuffled)))
        self.assertTrue(all(a.history == b.history
                            for a, b in zip(shuffled[::2], shuffled[1::2])))


if __name__ == "__main__":
    unittest.main()
