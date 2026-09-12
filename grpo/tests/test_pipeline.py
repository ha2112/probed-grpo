"""Exercise automatic sequencing and failure recovery without GPUs or network."""

import fcntl
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from grpo.pipeline import Pipeline, ROUTES


ENV = {
    "N_GPUS": "2", "TENSOR_PARALLEL_SIZE": "2", "TRAIN_BATCH_SIZE": "32",
    "PPO_MINI_BATCH_SIZE": "32", "PPO_MICRO_BATCH_SIZE": "1",
    "LOG_PROB_MICRO_BATCH_SIZE": "1", "ROLLOUT_N": "32",
    "MAX_PROMPT_LENGTH": "2048", "MAX_RESPONSE_LENGTH": "8192",
    "TOTAL_EPOCHS": "1", "JUDGE_WORKERS": "8", "MONOLITH_URL": "http://localhost/execute",
}


def write(path, content="fixture"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class SimulatedPipeline(Pipeline):
    """Simulate only expensive child commands; use the real orchestration/state."""

    def __init__(self, directory, env=ENV, fail=None, omit=None):
        super().__init__(directory, env)
        self.calls = []
        self.fail = fail
        self.omit = omit

    def execute(self, name, command, env=None):
        self.calls.append(name)
        if name == self.fail:
            raise subprocess.CalledProcessError(7, command)
        if name == self.omit:
            return
        if name == "grpo-tests":
            assert env["VERL_INTEGRATION"] == "1"
        if name == "probe":
            for item in ("probe.pt", "metrics.json", "embeddings.npz", "predictions.csv", "losses.csv"):
                write(self.probe / item)
        elif name == "corpora":
            for route in ROUTES:
                write(self.corpus / route / "train.parquet")
            write(self.corpus / "shared/validation.parquet")
            write(self.corpus / "probe_scores.jsonl")
            write(self.corpus / "manifest.json", json.dumps({"train_count": 64}))
        elif name == "smoke" or name.startswith("train-"):
            route = "random" if name == "smoke" else name.removeprefix("train-")
            base = Path((env or self.env)["AFTERBURNER_CHECKPOINT_DIR"])
            step = 1 if name == "smoke" else 2
            for rank in range(2):
                write(base / f"{route}-seed-42/global_step_{step}/actor/model_world_size_2_rank_{rank}.pt")
        elif name.startswith("export-"):
            route = name.removeprefix("export-")
            for item in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
                write(self.directory / "exports" / route / item)


class PipelineTests(unittest.TestCase):
    def test_full_sequence_and_completed_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            pipeline = SimulatedPipeline(path)
            pipeline.run()
            self.assertEqual(pipeline.calls, [
                "runtime", "judge", "download", "environment", "probe-inputs", "probe-tests",
                "grpo-tests", "probe", "corpora", "corpus-check", "config", "smoke",
                "train-random", "export-random", "train-official", "export-official",
                "train-probed", "export-probed",
            ])
            resumed = SimulatedPipeline(path)
            resumed.run()
            self.assertEqual(resumed.calls, ["runtime", "judge", "download", "environment",
                                             "corpus-check", "config"])
            state = json.loads((path / "progress.json").read_text())
            self.assertNotIn("active", state)
            self.assertEqual(len(state["completed"]), 12)

    def test_failure_stops_downstream_and_rerun_resumes_failed_route(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            pipeline = SimulatedPipeline(path, fail="train-official")
            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.run()
            self.assertNotIn("train-probed", pipeline.calls)
            state = json.loads((path / "progress.json").read_text())
            self.assertEqual(state["active"], "train-official")
            self.assertNotIn("train-official", state["completed"])
            resumed = SimulatedPipeline(path)
            resumed.run()
            self.assertNotIn("train-random", resumed.calls)
            self.assertNotIn("probe", resumed.calls)
            self.assertIn("train-official", resumed.calls)
            self.assertIn("export-probed", resumed.calls)

    def test_successful_exit_without_smoke_checkpoint_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = SimulatedPipeline(Path(directory) / "run", omit="smoke")
            with self.assertRaisesRegex(RuntimeError, "No final actor checkpoint"):
                pipeline.run()
            self.assertNotIn("train-random", pipeline.calls)
            self.assertNotIn("smoke", pipeline.state["completed"])

    def test_changed_settings_and_missing_completed_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            SimulatedPipeline(path).run()
            changed = SimulatedPipeline(path, dict(ENV, TOTAL_EPOCHS="2"))
            with self.assertRaisesRegex(RuntimeError, "Source or settings changed"):
                changed.run()
            self.assertEqual(changed.calls, [])
            (path / "probe/probe.pt").unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing or empty output"):
                SimulatedPipeline(path).run()

    def test_lock_and_unmanaged_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with (path / ".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "Another pipeline"):
                    SimulatedPipeline(path).run()
            write(path / "existing-output")
            with self.assertRaisesRegex(RuntimeError, "Nonempty run directory"):
                SimulatedPipeline(path).run()

    def test_missing_rank_checkpoint_and_incomplete_export_block_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            pipeline = SimulatedPipeline(path)
            pipeline.run()
            actor = pipeline.checkpoint("probed")
            (actor / "model_world_size_2_rank_1.pt").unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing or empty output"):
                pipeline.checkpoint("probed")
            (path / "exports/random/model.safetensors").unlink()
            with self.assertRaisesRegex(RuntimeError, "Incomplete exported"):
                pipeline.verify_export("random")

    def test_child_failure_is_logged_and_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "logs").mkdir()
            pipeline = Pipeline(path, ENV)
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                pipeline.execute("failed", [sys.executable, "-c",
                                            "print('failure evidence'); raise SystemExit(7)"])
            self.assertEqual(raised.exception.returncode, 7)
            self.assertIn("failure evidence", (path / "logs/failed.log").read_text())

    def test_interruption_cleans_up_child_and_preserves_failed_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "logs").mkdir()
            pipeline = Pipeline(path, ENV)
            process = Mock(pid=12345)
            process.stdout = Mock()
            process.stdout.__iter__ = Mock(side_effect=KeyboardInterrupt)
            with patch("grpo.pipeline.subprocess.Popen") as popen, patch("grpo.pipeline.os.killpg") as kill:
                popen.return_value.__enter__.return_value = process
                with self.assertRaises(KeyboardInterrupt):
                    pipeline.stage("probe", ["unused"], lambda: None)
                self.assertEqual(kill.call_args.args[0], 12345)
                process.wait.assert_called_once_with(timeout=15)
            state = json.loads(pipeline.state_path.read_text())
            self.assertEqual(state["active"], "probe")
            self.assertNotIn("probe", state["completed"])

    def test_export_failure_resumes_export_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            with self.assertRaises(subprocess.CalledProcessError):
                SimulatedPipeline(path, fail="export-random").run()
            resumed = SimulatedPipeline(path)
            resumed.run()
            self.assertNotIn("train-random", resumed.calls)
            self.assertIn("export-random", resumed.calls)
            self.assertIn("train-official", resumed.calls)


if __name__ == "__main__":
    unittest.main()
