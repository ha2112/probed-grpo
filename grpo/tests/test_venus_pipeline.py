"""Venus data/reward regressions; Codeforces remains the probe training input."""

import json
import math
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from artifact_cache import VENUS_DATASET_ID, VENUS_DATASET_REVISION, load_cached_dataset
from grpo import venus_corpus as corpus, venus_reward as reward, preflight


def problem(identifier=1, difficulty="Easy"):
    return {
        "problem_id": identifier, "title": f"problem-{identifier}",
        "question_content": "Return the integer supplied to echo.", "difficulty": difficulty,
        "code_prompt": "class Solution:\n    def echo(self, n: int) -> int:",
        "test_case_runners": "==Code Submission==\nprint(Solution().echo(int(input())))",
        "test_case_evaluator": "def evaluate(expected_output, program_output):\n    return int(expected_output) == int(program_output)",
        "test_cases": json.dumps([{"input": "1", "output": "1"}, {"input": "2", "output": "2"}]),
        "solutions": [{"code": "class Solution:\n    def echo(self, n: int) -> int:\n        return n",
                       "passed": True, "time": 2.0, "memory": 100.0, "integral": 150.0, "status": "success"}],
    }


class VenusCorpusTests(unittest.TestCase):
    def test_separate_dataset_and_pinned_download(self):
        import linear_probe
        self.assertEqual(linear_probe.parse_args([]).data, "deepmind/code_contests")
        args = corpus.parse_args([])
        self.assertEqual((args.train_split, args.validation_split), ("train", "test"))
        self.assertEqual(args.output_dir, ROOT / "grpo/data/venus")
        with tempfile.TemporaryDirectory() as directory, patch("datasets.load_dataset") as load:
            load_cached_dataset(VENUS_DATASET_ID, "test", Path(directory))
            load.assert_called_once_with(VENUS_DATASET_ID, split="test", cache_dir=str(Path(directory).resolve()),
                                         revision=VENUS_DATASET_REVISION)

    def test_three_objectives_and_native_venus_payload(self):
        rows = corpus.make_records([problem()], "train", 42)
        self.assertEqual({r["extra_info"]["efficiency_instruction"] for r in rows}, {"time", "memory", "integral"})
        for row in rows:
            self.assertEqual(row["data_source"], VENUS_DATASET_ID)
            self.assertEqual(json.loads(row["reward_model"]["ground_truth"]), problem()["solutions"][0])
            info = row["extra_info"]
            self.assertEqual(info["case_multiply"], 64)
            self.assertEqual(json.loads(info["instance"])["test_cases"], problem()["test_cases"])
            self.assertIn("Original Performance", row["prompt"][1]["content"])
            self.assertNotIn("cf_rating", repr(row))

    def test_selection_and_baselines_are_deterministic(self):
        p = problem()
        p["solutions"].append(dict(p["solutions"][0], code="alternative", passed=False))
        self.assertEqual(corpus.make_records([p, problem(2)], "train", 42),
                         corpus.make_records([problem(2), p], "train", 42))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            corpus.select_venus([p, p])
        broken = problem()
        broken["test_cases"] = "[]"
        with self.assertRaisesRegex(ValueError, "test"):
            corpus.select_venus([broken])

    def test_failed_baselines_preserve_timeout_measurements_and_empty_code(self):
        p = problem()
        p["solutions"] = [{"code": "", "passed": False, "status": "timeout",
                           "time": math.inf, "memory": math.inf, "integral": math.inf}]
        rows = corpus.make_records([p], "train", 42)
        baseline = json.loads(rows[0]["reward_model"]["ground_truth"])
        current = {"passed": True, "time": 1.0, "memory": 100.0, "integral": 100.0}
        self.assertTrue(math.isfinite(reward.combine_reward(baseline, current, "time", True)))

    def test_order_only_changes_after_shared_filter_and_trim(self):
        rows = corpus.make_records([problem(1, "Hard"), problem(2, "Easy"), problem(3, "Medium")], "train", 42)
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = [range(n) for n in (1, 1, 99, 1, 1, 1, 1, 1, 1)]
        rows, stats = corpus.prepare_records(rows, tokenizer, 10, 4)
        self.assertEqual(stats, {"selected": 9, "overlong": 1, "batch_tail": 0})
        for row in rows:
            row["extra_info"]["probe_difficulty"] = -row["extra_info"]["problem_id"]
        routes = {route: corpus.order_records(rows, "train", route, 42) for route in corpus.ROUTES}
        canonical = lambda rs: sorted(json.dumps(r, sort_keys=True) for r in rs)
        self.assertTrue(all(canonical(rs) == canonical(rows) for rs in routes.values()))
        official = [corpus.DIFFICULTIES[r["extra_info"]["difficulty"]] for r in routes["official"]]
        self.assertEqual(official, sorted(official))
        predicted = [r["extra_info"]["probe_difficulty"] for r in routes["probed"]]
        self.assertEqual(predicted, sorted(predicted))

    def test_probe_scores_statement_once_per_problem_and_resumes(self):
        rows = corpus.make_records([problem()], "train", 42)
        scorer = Mock()
        scorer.score.return_value = 1200.0
        factory = Mock(return_value=scorer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            scores = {}
            loaded = corpus.ensure_scores(rows, scores, path, factory)
            corpus.ensure_scores(rows, scores, path, factory, loaded)
        scorer.score.assert_called_once_with(problem()["question_content"])
        self.assertTrue(all(r["extra_info"]["probe_difficulty"] == 1200.0 for r in rows))

    def test_real_parquet_preflight_and_rejects_codeforces(self):
        train = corpus.make_records([problem(1), problem(2)], "train", 42)
        validation = corpus.make_records([problem(3)], "test", 42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            corpus.write_corpora(train, validation, path, 42,
                                 {"dataset": VENUS_DATASET_ID, "dataset_revision": VENUS_DATASET_REVISION,
                                  "train_batch_size": 3, "max_prompt_length": 2048})
            manifest = preflight.verify_corpora(path, 3, 2048)
            self.assertEqual((manifest["train_count"], manifest["validation_count"]), (6, 3))
            with self.assertRaisesRegex(ValueError, "Batch size differs"):
                preflight.verify_corpora(path, 2, 2048)
            manifest["schema_version"] = 2
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Venus"):
                preflight.verify_corpora(path)
            manifest["schema_version"] = 3
            (path / "manifest.json").write_text(json.dumps(manifest))
            with (path / "random/train.parquet").open("ab") as handle:
                handle.write(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                preflight.verify_corpora(path)


class VenusRewardTests(unittest.TestCase):
    def test_harness_uses_class_runner_custom_evaluator_and_all_cases(self):
        instance = problem()
        code = reward.runner_code(instance["solutions"][0]["code"], instance, 2)
        def execute(program):
            return subprocess.run([sys.executable, "-c", program], text=True,
                                  capture_output=True, check=True).stdout
        self.assertEqual(execute(code), "Success\n")
        wrong = code.replace("return n", "return 1")
        self.assertEqual(execute(wrong), "Failed\n")
        invalid = reward.runner_code("invalid python !", instance, 1)
        self.assertEqual(execute(invalid), "Failed\n")

    def test_profiled_response_and_missing_profiling_fails(self):
        info = corpus.make_records([problem()], "train", 42)[0]["extra_info"]
        response = Mock()
        response.json.return_value = {"status": "success", "output_dict": {
            "stdout": "Success\n", "duration": 1.0, "peak_memory": 50.0, "integral": 75.0}}
        with patch("requests.post", return_value=response) as post:
            result = reward.evaluate_solution(problem()["solutions"][0]["code"], info)
        self.assertEqual(result, {"passed": True, "time": 1.0, "memory": 50.0, "integral": 75.0})
        self.assertTrue(post.call_args.kwargs["json"]["run_profiling"])
        response.json.return_value = {"stdout": "Success\n"}
        with patch("requests.post", return_value=response), self.assertRaisesRegex(RuntimeError, "profil"):
            reward.evaluate_solution("pass", info)

    def test_candidate_failure_vs_infrastructure_failure(self):
        info = corpus.make_records([problem()], "train", 42)[0]["extra_info"]
        response = Mock()
        response.json.return_value = {"status": "success", "output_dict": {
            "stdout": "Failed\n", "duration": 1, "peak_memory": 50, "integral": 75}}
        with patch("requests.post", return_value=response):
            self.assertFalse(reward.evaluate_solution("pass", info)["passed"])
        response.json.return_value = {"status": "timeout", "output_dict": None}
        with patch("requests.post", return_value=response):
            self.assertFalse(reward.evaluate_solution("pass", info)["passed"])
        with patch("requests.post", side_effect=ConnectionError("offline")), self.assertRaisesRegex(RuntimeError, "sandbox"):
            reward.evaluate_solution("pass", info)

    def test_afterburner_efficiency_weights_and_numpy_collation(self):
        rows = corpus.make_records([problem()], "train", 42)
        text = "<thinking>x</thinking><solution>```python\npass\n```</solution>"
        result = {"passed": True, "time": 1.0, "memory": 50.0, "integral": 75.0}
        with patch.object(reward, "evaluate_solution", return_value=result):
            actual = reward.venus_reward_fn_batch(np.array([VENUS_DATASET_ID] * 3), [text] * 3,
                np.array([r["reward_model"]["ground_truth"] for r in rows]),
                np.array([r["extra_info"] for r in rows], dtype=object))
        for score in actual:
            self.assertAlmostEqual(score, 0.45 + 0.25 * math.tanh(0.5))
        self.assertAlmostEqual(reward.combine_reward(problem()["solutions"][0], dict(result, passed=False), "time", True), -0.3)
        self.assertEqual(reward.venus_reward_fn_batch([], [], [], []), [])

    def test_launch_defaults_to_venus(self):
        command = subprocess.check_output(["bash", str(ROOT / "grpo/train.sh"), "--dry-run", "random"], text=True)
        args = shlex.split(command)
        self.assertIn("custom_reward_function.name=venus_reward_fn_batch", args)
        self.assertIn(f"data.train_files=['{ROOT}/grpo/data/venus/random/train.parquet']", args)
        self.assertIn(f"trainer.default_local_dir='{ROOT}/grpo/checkpoints/venus/random-seed-42'", args)


if __name__ == "__main__":
    unittest.main()
