"""CPU unit tests for Venus solve-from-scratch corpus and reward helpers."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from grpo import venus_solve_corpus as corpus
from grpo import venus_solve_reward as reward
from grpo.solve_grpo_train import grpo_advantages


class VenusSolveTests(unittest.TestCase):
    def test_combine_solve_reward_prefers_pass(self):
        self.assertGreater(
            reward.combine_solve_reward(True, False),
            reward.combine_solve_reward(False, True),
        )
        self.assertAlmostEqual(reward.combine_solve_reward(True, True), 1.0)
        self.assertAlmostEqual(reward.combine_solve_reward(False, False), 0.0)
        partial = reward.combine_solve_reward(False, 0.6, test_fraction=0.5)
        nothing = reward.combine_solve_reward(False, 0.0, test_fraction=0.0)
        self.assertGreater(partial, nothing)
        self.assertLess(reward.combine_solve_reward(False, 1.0, test_fraction=1.0), 1.0)

    def test_make_solve_record_has_no_baseline(self):
        problem = {
            "problem_id": 7,
            "title": "Echo",
            "difficulty": "Easy",
            "question_content": "Read n and print n.",
            "test_case_runners": "==Code Submission==\nprint(1)",
            "test_case_evaluator": "def evaluate(expected, actual):\n    return True",
            "test_cases": json.dumps([{"input": "1", "output": "1"}]),
            "solutions": [{"code": "print(1)", "passed": True, "time": 1, "memory": 1, "integral": 1}],
        }
        row = corpus.make_solve_record(problem, "train")
        user = row["prompt"][1]["content"]
        self.assertIn("Problem Description", user)
        self.assertIn("Only code", user)
        self.assertIn("no <thinking>", user.lower())
        self.assertNotIn("Original Solution", user)
        self.assertNotIn("Original Performance", user)
        self.assertTrue(corpus.SOLVE_RESPONSE_PREFIX.startswith("<solution>"))
        info = row["extra_info"]
        self.assertEqual(info["problem_id"], 7)
        self.assertIn("instance", info)
        instance = json.loads(info["instance"])
        self.assertIn("test_cases", instance)

    def test_assign_buckets_maps_three_groups(self):
        rows = [
            corpus.make_solve_record(
                {
                    "problem_id": i,
                    "title": f"p{i}",
                    "difficulty": "Easy",
                    "question_content": "x",
                    "test_case_runners": "==Code Submission==",
                    "test_case_evaluator": "def evaluate(expected, actual):\n    return True",
                    "test_cases": "[]",
                    "solutions": [{"code": "pass", "passed": True, "time": 1, "memory": 1, "integral": 1}],
                },
                "train",
            )
            for i in range(3)
        ]
        corpus.assign_buckets(rows, [0, 1, 2], [0.1, 1.2, 2.3])
        self.assertEqual([r["extra_info"]["bucket"] for r in rows], ["easy", "medium", "hard"])

    def test_grpo_advantages_zero_mean_per_group(self):
        rewards = torch.tensor([1.0, 0.0, 0.5, 0.5])
        advantages = grpo_advantages(rewards, group_size=2)
        groups = advantages.view(2, 2)
        self.assertTrue(torch.allclose(groups.mean(dim=-1), torch.zeros(2), atol=1e-6))

    def test_extract_solve_code_accepts_markdown_without_tags(self):
        code = "class Solution:\n    def echo(self, n):\n        return n"
        text = "Sure.\n```python\n" + code + "\n```\n"
        self.assertEqual(reward.extract_solve_code(text).strip(), code)
        self.assertGreater(reward.format_score(text), 0.0)
        unclosed = "<solution>\n```python\n" + code
        self.assertEqual(reward.extract_solve_code(unclosed).strip(), code)
        self.assertTrue(reward.code_is_valid(reward.extract_solve_code(unclosed)))
        loop = "<solution>\n```python\ndef f():\n    return [\n" + ", ".join(["1"] * 40)
        self.assertFalse(reward.code_is_valid(reward.extract_solve_code(loop)))
        self.assertGreater(reward.format_score(unclosed), reward.format_score(loop))
        trailed = unclosed + "\n" + ", ".join(["1"] * 24)
        recovered = reward.extract_solve_code(trailed)
        self.assertIn("return n", recovered)
        self.assertTrue(reward.code_is_valid(recovered))
        self.assertLess(recovered.count(", 1"), 4)
        second = "print('junk')"
        both = "<solution>\n```python\n" + code + "\n```\n```python\n" + second + "\n```\n</solution>"
        self.assertEqual(reward.extract_solve_code(both).strip(), code)
        kept = reward.gradient_continuation(code + "\n" + ", ".join(["1"] * 24))
        self.assertTrue(kept.startswith("class Solution"))
        self.assertNotIn(", 1, 1, 1, 1", kept)

    def test_local_judge_accepts_and_rejects(self):
        import os

        os.environ["VENUS_JUDGE"] = "local"
        reward._JUDGE_MODE = "local"
        instance = {
            "test_case_runners": "==Code Submission==\nprint(Solution().echo(int(input())))",
            "test_case_evaluator": (
                "def evaluate(expected, actual):\n    return int(expected) == int(actual)"
            ),
            "test_cases": json.dumps([{"input": "7", "output": "7"}]),
        }
        info = {"instance": instance, "case_multiply": 1}
        good = "class Solution:\n    def echo(self, n):\n        return n"
        bad = "class Solution:\n    def echo(self, n):\n        return 0"
        self.assertTrue(reward.evaluate_solution_correctness(good, info))
        self.assertFalse(reward.evaluate_solution_correctness(bad, info))

    def test_score_parser_and_syntax(self):
        self.assertTrue(reward.code_is_valid("def f():\n    return 1\n"))
        self.assertFalse(reward.code_is_valid("def f(:\n"))
        detail = reward._parse_harness_detail("VENUS_SCORE:3/10\nFailed\n")
        self.assertEqual(detail["verdict"], "failed")
        self.assertEqual(detail["tests_passed"], 3)
        self.assertEqual(detail["tests_total"], 10)
        src = Path(__file__).resolve().parents[1].joinpath("solve_grpo_train.py").read_text()
        self.assertNotIn("dist.broadcast(payload", src)
        self.assertIn("Each rank must score", src)


if __name__ == "__main__":
    unittest.main()
