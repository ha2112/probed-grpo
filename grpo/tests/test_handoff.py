"""Regression checks for the fresh-machine pipeline and verl boundaries."""

import json
import os
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
from grpo import codeforces_corpus as corpus, codeforces_reward as reward, preflight, judge_server
from grpo import venus_corpus, venus_reward
from test_codeforces_pipeline import problem
from test_venus_pipeline import problem as venus_problem


class HandoffTests(unittest.TestCase):
    def test_local_judge_isolated_and_request_container_cleaned_up(self):
        with patch.object(judge_server.subprocess, "run", return_value=Mock(
            returncode=0, stdout="CODECONTESTS_RESULT:1/1", stderr="",
        )) as run:
            result = judge_server.execute({"language": "python", "code": "print(1)", "timeout": 90})
        self.assertEqual(result["stdout"], "CODECONTESTS_RESULT:1/1")
        command = run.call_args_list[0].args[0]
        self.assertIn("none", command)
        self.assertIn("--read-only", command)
        self.assertIn("65534:65534", command)
        self.assertNotIn("--mount", command)
        self.assertNotIn("--privileged", command)
        self.assertEqual(run.call_args_list[1].args[0][-1], command[command.index("--name") + 1])

    def test_local_judge_cleans_up_after_timeout(self):
        with patch.object(judge_server.subprocess, "run", side_effect=[
            subprocess.TimeoutExpired("docker", 93), Mock(returncode=0),
        ]) as run, self.assertRaises(subprocess.TimeoutExpired):
            judge_server.execute({"language": "python", "code": "while True: pass", "timeout": 90})
        self.assertEqual(run.call_count, 2)

    def test_numpy_batch_from_verl_is_accepted(self):
        truth = json.dumps({"baseline_passed": True, "tests": [{"input": "", "output": "1"}]})
        with patch.object(reward, "evaluate_solution", return_value=True):
            actual = reward.codeforces_reward_fn_batch(
                np.array(["code", "code"]), ["", ""], np.array([truth, truth]),
                np.array([{}, {}], dtype=object),
            )
        self.assertEqual(actual, [0.25, 0.25])

    def test_all_cases_are_judged_in_bounded_requests(self):
        cases = [{"input": "", "output": "1"}] * 19
        response = Mock()
        response.json.side_effect = [{"stdout": f"CODECONTESTS_RESULT:{n}/{n}"} for n in (8, 8, 3)]
        with patch("requests.post", return_value=response) as post:
            self.assertTrue(reward.evaluate_solution("print(1)", cases))
        self.assertEqual(post.call_count, 3)

    def test_inconsistent_judge_count_fails(self):
        response = Mock()
        response.json.return_value = {"stdout": "CODECONTESTS_RESULT:2/2"}
        with patch("requests.post", return_value=response), self.assertRaisesRegex(RuntimeError, "case count"):
            reward.evaluate_solution("print(1)", [{"input": "", "output": "1"}])

    def test_filter_and_tail_selection_happen_before_ordering(self):
        problems = [problem(str(i), i, "A", 800 + i * 100) for i in range(7)]
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = [list(range(n)) for n in (1, 1, 100, 1, 1, 1, 1)]
        selected, stats = corpus.prepare_problems(problems, tokenizer, max_prompt_length=10, batch_size=4)
        self.assertEqual(stats, {"selected": 7, "overlong": 1, "batch_tail": 2})
        self.assertEqual([row["name"] for row in selected], ["0", "1", "3", "4"])
        scores = {corpus.problem_key("train", row): 2000 - i for i, row in enumerate(selected)}
        sets = [sorted(row["name"] for row in corpus.order_problems(selected, "train", route, scores, 42))
                for route in corpus.ROUTES]
        self.assertTrue(all(value == sets[0] for value in sets))

    def test_legacy_codeforces_corpus_is_rejected_for_venus_training(self):
        train = [problem(str(i), i, "A", 800 + i * 100) for i in range(4)]
        validation = [problem("valid", 10, "B", 2000)]
        scores = {corpus.problem_key(split, row): row["cf_rating"]
                  for split, rows in (("train", train), ("validation", validation)) for row in rows}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            corpus.write_corpora(train, validation, scores, path, 42,
                                {"train_batch_size": 4, "max_prompt_length": 2048})
            with self.assertRaisesRegex(ValueError, "Rebuild Venus"):
                preflight.verify_corpora(path, 4, 2048)


@unittest.skipUnless(os.environ.get("VERL_SOURCE_DIR"), "Set VERL_SOURCE_DIR to the pinned verl checkout")
class UpstreamConfigTests(unittest.TestCase):
    def test_every_route_composes_against_the_real_upstream_schema(self):
        config_dir = Path(os.environ["VERL_SOURCE_DIR"]) / "verl/trainer/config"
        for route in corpus.ROUTES:
            command = subprocess.check_output(["bash", str(ROOT / "grpo/train.sh"), "--dry-run", route], text=True)
            config = preflight.compose_config(shlex.split(command)[3:], config_dir)
            preflight.validate_config(config)
            self.assertIn(route + "/train.parquet", config.data.train_files[0])
            self.assertEqual(Path(config.actor_rollout_ref.model.path).parent.name, "afterburner")
            config.actor_rollout_ref.rollout.tensor_model_parallel_size = config.trainer.n_gpus_per_node + 1
            with self.assertRaisesRegex(ValueError, "divisible"):
                preflight.validate_config(config)

    @unittest.skipUnless(os.environ.get("VERL_INTEGRATION") == "1", "Requires installed pinned verl dependencies")
    def test_real_verl_dataset_collation_and_reward_manager(self):
        from transformers import AutoTokenizer
        from verl import DataProto
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        from verl.workers.reward_manager.batch import BatchRewardManager
        import torch
        import artifact_cache

        tokenizer = AutoTokenizer.from_pretrained(artifact_cache.DEFAULT_MODEL_DIR, local_files_only=True)
        config_dir = Path(os.environ["VERL_SOURCE_DIR"]) / "verl/trainer/config"
        command = subprocess.check_output(["bash", str(ROOT / "grpo/train.sh"), "--dry-run", "random"], text=True)
        config = preflight.compose_config(shlex.split(command)[3:], config_dir)
        rows = venus_corpus.make_records([venus_problem(1), venus_problem(2)], "train", 42)
        validation = venus_corpus.make_records([venus_problem(3)], "test", 42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            venus_corpus.write_corpora(rows, validation, path, 42, {})
            data = RLHFDataset([str(path / "random/train.parquet")], tokenizer, config.data)
            batch = collate_fn([data[0], data[1]])
        response = "<thinking>x</thinking><solution>```python\nprint(1)\n```</solution>"
        response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False)["input_ids"].repeat(2, 1)
        batch["prompts"] = batch.pop("input_ids")
        batch["responses"] = response_ids
        batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(response_ids)], dim=1)
        protocol = DataProto.from_single_dict(batch)
        manager = BatchRewardManager(tokenizer, 0, venus_reward.venus_reward_fn_batch)
        with patch.object(venus_reward, "evaluate_solution", return_value={
            "passed": True, "time": 2.0, "memory": 100.0, "integral": 150.0,
        }):
            result = manager(protocol)
        self.assertEqual(tuple(result.shape), tuple(response_ids.shape))
        torch.testing.assert_close(result.sum(-1), torch.tensor([0.45, 0.45]))


if __name__ == "__main__":
    unittest.main()
