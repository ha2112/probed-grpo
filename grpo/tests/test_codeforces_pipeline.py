import copy
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


GRPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GRPO_DIR))

import codeforces_corpus as corpus  # noqa: E402
import codeforces_reward as reward  # noqa: E402


def problem(name, contest, index, rating):
    return {
        "name": name,
        "description": f"Solve {name}.",
        "source": 2,
        "cf_rating": rating,
        "cf_contest_id": contest,
        "cf_index": index,
        "solutions": {"language": [1], "solution": [f"print('{name}')"]},
        "public_tests": {"input": ["\n"], "output": [f"{name}\n"]},
        "private_tests": {"input": [], "output": []},
        "generated_tests": {"input": [], "output": []},
        "time_limit": {"seconds": 2},
    }


class CorpusRouteTests(unittest.TestCase):
    def setUp(self):
        self.problems = [
            problem("middle", 3, "B", 1500),
            problem("hard", 1, "C", 2200),
            problem("easy", 2, "A", 800),
            problem("upper", 4, "D", 1800),
        ]
        self.scores = {
            corpus.problem_key("train", row): score
            for row, score in zip(self.problems, (1100, 700, 1900, 1400))
        }

    def records(self, route):
        return corpus.build_records(self.problems, "train", route, self.scores, seed=42)

    def test_routes_change_only_physical_order(self):
        routes = {route: self.records(route) for route in corpus.ROUTES}
        canonical = {
            route: sorted(records, key=lambda row: row["extra_info"]["problem_id"])
            for route, records in routes.items()
        }
        self.assertEqual(canonical["random"], canonical["official"])
        self.assertEqual(canonical["official"], canonical["probed"])

        official = [row["extra_info"]["cf_rating"] for row in routes["official"]]
        probed = [row["extra_info"]["probe_difficulty"] for row in routes["probed"]]
        self.assertEqual(official, sorted(official))
        self.assertEqual(probed, sorted(probed))
        self.assertEqual(self.records("random"), routes["random"])

    def test_route_does_not_enter_prompt_or_reward_payload(self):
        for route in corpus.ROUTES:
            for record in self.records(route):
                serialized = repr(record)
                self.assertNotIn(f"'{route}'", serialized)
                self.assertIn("<thinking>", record["prompt"][0]["content"])
                self.assertTrue(json.loads(record["reward_model"]["ground_truth"])["baseline_passed"])

    def test_selection_is_identical_before_ordering(self):
        invalid = copy.deepcopy(self.problems[0])
        invalid["cf_rating"] = 0
        other_source = copy.deepcopy(self.problems[1])
        other_source["source"] = 5
        selected = corpus.select_codeforces(self.problems + [invalid, other_source])
        self.assertEqual(len(selected), len(self.problems))

    def test_missing_probe_score_fails_instead_of_silently_dropping(self):
        with self.assertRaisesRegex(KeyError, "Missing probe scores"):
            corpus.build_records(self.problems, "train", "official", {}, seed=42)

    def test_probe_model_is_reused_across_splits(self):
        scorer = Mock()
        scorer.score.side_effect = [1000.0, 1100.0]
        factory = Mock(return_value=scorer)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "scores.jsonl"
            corpus.initialize_score_cache(cache, "probe-hash")
            scores = corpus.load_score_cache(cache, "probe-hash")
            loaded = corpus.ensure_scores(
                [self.problems[0]], "train", scores, cache, factory
            )
            corpus.ensure_scores(
                [self.problems[1]], "validation", scores, cache, factory, loaded
            )
        factory.assert_called_once_with()


class RewardTests(unittest.TestCase):
    def test_extracts_afterburner_response_and_checks_format(self):
        text = "<thinking>reason</thinking><solution>```python\nprint(1)\n```</solution>"
        self.assertEqual(reward.extract_code(text), "print(1)")
        self.assertTrue(reward.valid_format(text))
        self.assertFalse(reward.valid_format("<solution>```python\nprint(1)\n```</solution>"))

    def test_reward_uses_afterburner_correctness_and_format_weights(self):
        self.assertAlmostEqual(reward.combine_reward(True, True, True), 0.45)
        self.assertAlmostEqual(reward.combine_reward(True, False, True), -0.3)
        self.assertAlmostEqual(reward.combine_reward(True, True, False), 0.25)

    def test_batch_reward_shares_one_function_for_all_routes(self):
        solution = "<thinking>x</thinking><solution>```python\nprint(1)\n```</solution>"
        ground_truth = {"baseline_passed": True, "tests": [{"input": "", "output": "1"}]}
        with patch.object(reward, "evaluate_solution", return_value=True) as evaluate:
            scores = reward.codeforces_reward_fn_batch(
                ["codecontests_afterburner"], [solution], [ground_truth], [{}]
            )
        self.assertEqual(scores, [0.45])
        evaluate.assert_called_once()


class LauncherTests(unittest.TestCase):
    def test_probe_and_model_use_repository_model_cache(self):
        args = corpus.parse_args([])
        self.assertEqual(args.probe, corpus.ROOT / "model-cache/probe/probe.pt")
        launcher = (GRPO_DIR / "train.sh").read_text(encoding="utf-8")
        self.assertIn('"${ROOT_DIR}/artifact_cache.py" "${MODEL_PATH}"', launcher)

    def test_one_launcher_contains_all_shared_afterburner_settings(self):
        launcher = subprocess.check_output(
            ["bash", str(GRPO_DIR / "train.sh"), "--dry-run", "random"], text=True,
        )
        arguments = shlex.split(launcher)
        for setting in (
            "data.shuffle=False",
            "data.train_batch_size=32",
            "data.max_prompt_length=2048",
            "data.max_response_length=8192",
            "actor_rollout_ref.actor.optim.lr=1e-6",
            "actor_rollout_ref.rollout.n=32",
            "reward_model.reward_manager=batch",
            "trainer.total_epochs=200",
        ):
            self.assertIn(setting, arguments)
        self.assertEqual(launcher.count("verl.trainer.main_ppo"), 1)


if __name__ == "__main__":
    unittest.main()
