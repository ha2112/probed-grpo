"""Run the complete experiment inside the pinned container, with persistent progress."""

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("random", "official", "probed")
SETTINGS = (
    "N_GPUS", "TENSOR_PARALLEL_SIZE", "TRAIN_BATCH_SIZE", "PPO_MINI_BATCH_SIZE",
    "PPO_MICRO_BATCH_SIZE", "LOG_PROB_MICRO_BATCH_SIZE", "ROLLOUT_N",
    "MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH", "TOTAL_EPOCHS",
    "JUDGE_WORKERS", "MONOLITH_URL", "EXPERIMENT_SEED",
)


def signature(env):
    # Source/settings changes require a separate experiment directory.
    paths = [ROOT / "linear_probe.py", ROOT / "artifact_cache.py",
             ROOT / "requirements-probe.txt", ROOT / "requirements-grpo.txt"]
    paths += sorted((ROOT / "grpo").glob("*.py"))
    paths += sorted((ROOT / "grpo").glob("*.sh"))
    paths += [ROOT / "grpo/Dockerfile", ROOT / "grpo/requirements-overlay.lock",
              ROOT / "grpo/constraints-runtime.txt"]
    return {
        "version": 1,
        "root": str(ROOT),
        "python": sys.executable,
        "settings": {key: env.get(key, "") for key in SETTINGS},
        "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in paths},
    }


def require_files(paths):
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")


class Pipeline:
    def __init__(self, directory, env):
        self.directory = directory
        self.env = dict(env)
        self.env["PYTHONUNBUFFERED"] = "1"
        self.env["PYTHON_BIN"] = sys.executable
        self.env.setdefault("EXPERIMENT_SEED", "42")
        self.probe = directory / "probe"
        self.corpus = directory / "corpora"
        self.checkpoints = directory / "checkpoints"
        self.env["VENUS_GRPO_DATA_DIR"] = str(self.corpus)
        self.env["AFTERBURNER_CHECKPOINT_DIR"] = str(self.checkpoints)
        self.env["GRPO_LAUNCH_DIR"] = str(directory / "configs")
        self.state_path = directory / "progress.json"
        self.state = {"signature": signature(self.env), "completed": []}
        self.python = sys.executable

    def save(self):
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2) + "\n")
        temporary.replace(self.state_path)

    def execute(self, name, command, env=None):
        print(f"\n[{name}] {shlex.join(command)}", flush=True)
        # Inherit the terminal and retain combined stdout/stderr for each stage.
        with (self.directory / "logs" / f"{name}.log").open("a") as log:
            with subprocess.Popen(command, cwd=ROOT, env=env or self.env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors="replace", start_new_session=True) as process:
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                    if code:
                        raise subprocess.CalledProcessError(code, command)
                except BaseException:
                    # Stop the launched process group before releasing the run lock.
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise

    def stage(self, name, command, verify, env=None):
        if name in self.state["completed"]:
            verify()
            print(f"[{name}] already completed; outputs verified", flush=True)
            return
        self.state["active"] = name
        self.save()
        self.execute(name, command, env)
        verify()
        self.state["completed"].append(name)
        self.state.pop("active", None)
        self.save()

    def checkpoint(self, route, smoke=False):
        if smoke:
            step = 1
            base = self.checkpoints / "smoke"
        else:
            manifest = json.loads((self.corpus / "manifest.json").read_text())
            step = manifest["train_count"] // int(self.env["TRAIN_BATCH_SIZE"])
            step *= int(self.env["TOTAL_EPOCHS"])
            if step < 1:
                raise RuntimeError("The full run must contain at least one optimizer step")
            base = self.checkpoints
        actor = base / f"{route}-seed-{self.env['EXPERIMENT_SEED']}" / f"global_step_{step}" / "actor"
        if not actor.is_dir():
            raise RuntimeError(f"No final actor checkpoint at {actor}")
        # Every rank must have saved its model shard.
        world_size = int(self.env["N_GPUS"])
        shards = [actor / f"model_world_size_{world_size}_rank_{rank}.pt"
                  for rank in range(world_size)]
        require_files(shards)
        return actor

    def verify_export(self, route):
        sys.path.insert(0, str(ROOT))
        from artifact_cache import model_snapshot_available

        path = self.directory / "exports" / route
        if not model_snapshot_available(path):
            raise RuntimeError(f"Incomplete exported Hugging Face model: {path}")

    def run(self):
        # A kernel lock is released even when the runner is killed.
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Another pipeline is using {self.directory}") from None
            if self.state_path.exists():
                previous = json.loads(self.state_path.read_text())
                if previous["signature"] != self.state["signature"]:
                    raise RuntimeError("Source or settings changed; use a new --run-dir")
                self.state = previous
            elif any(path.name != ".lock" for path in self.directory.iterdir()):
                raise RuntimeError("Nonempty run directory has no progress.json; use a new --run-dir")
            self.save()
            (self.directory / "logs").mkdir(exist_ok=True)
            self.run_stages()
            print(f"\nPipeline complete. Results and exported models: {self.directory}", flush=True)

    def run_stages(self):
        py = self.python
        # Repeat live health checks on every invocation, including resumption.
        self.execute("runtime", [py, "grpo/preflight.py", "--stage", "runtime"])
        self.execute("judge", [py, "grpo/venus_reward.py", "--check"])
        self.execute("download", [py, "artifact_cache.py"])
        self.execute("venus-inputs", [py, "grpo/venus_corpus.py", "--download-only"])
        self.execute("environment", [py, "-m", "pip", "freeze"])
        self.stage("probe-inputs", [py, "grpo/preflight.py", "--stage", "probe"], lambda: None)
        self.stage("probe-tests", [py, "-m", "unittest", "-v",
                                  "test_artifact_cache.py", "test_linear_probe.py"], lambda: None)
        integration_env = dict(self.env, VERL_SOURCE_DIR="/opt/verl", VERL_INTEGRATION="1")
        self.stage("grpo-tests", [py, "-m", "unittest", "discover", "-s", "grpo/tests", "-v"],
                   lambda: None, integration_env)
        probe_outputs = [self.probe / name for name in
                         ("probe.pt", "metrics.json", "embeddings.npz", "predictions.csv", "losses.csv")]
        self.stage("probe", [py, "linear_probe.py", "--device", "cuda:0", "--dtype", "float16",
                             "--output-dir", str(self.probe)],
                   lambda: require_files(probe_outputs))
        corpus_outputs = [self.corpus / f"{route}/train.parquet" for route in ROUTES]
        corpus_outputs += [self.corpus / "shared/validation.parquet",
                           self.corpus / "manifest.json", self.corpus / "probe_scores.jsonl"]
        self.stage("corpora", [
            py, "grpo/venus_corpus.py", "--device", "cuda:0", "--probe", str(self.probe / "probe.pt"),
            "--output-dir", str(self.corpus), "--train-batch-size", self.env["TRAIN_BATCH_SIZE"],
            "--max-prompt-length", self.env["MAX_PROMPT_LENGTH"], "--seed", self.env["EXPERIMENT_SEED"],
        ], lambda: require_files(corpus_outputs))
        # Check checksums and route equality even when corpus preparation was skipped.
        self.execute("corpus-check", [py, "grpo/preflight.py", "--stage", "corpus",
                                     "--data-dir", str(self.corpus)])
        self.execute("config", ["bash", "grpo/train.sh", "--config-only", "random"])
        smoke_env = dict(self.env, AFTERBURNER_CHECKPOINT_DIR=str(self.checkpoints / "smoke"),
                         GRPO_LAUNCH_DIR=str(self.directory / "configs/smoke"))
        self.stage("smoke", [
            "bash", "grpo/train.sh", "random", "trainer.total_training_steps=1",
            "trainer.save_freq=1", "trainer.test_freq=-1", "trainer.resume_mode=disable",
            "trainer.experiment_name=venus-smoke",
        ], lambda: self.checkpoint("random", smoke=True), smoke_env)
        for route in ROUTES:
            self.stage(f"train-{route}", ["bash", "grpo/train.sh", route],
                       lambda route=route: self.checkpoint(route))
            actor = self.checkpoint(route)
            self.stage(f"export-{route}", [
                py, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                "--local_dir", str(actor), "--target_dir", str(self.directory / "exports" / route),
            ], lambda route=route: self.verify_export(route))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/pipeline",
                        help="Persistent outputs and progress; use a new path for a changed experiment")
    args = parser.parse_args()
    # env.sh supplies the complete supported configuration.
    missing = [key for key in SETTINGS if key != "EXPERIMENT_SEED" and not os.environ.get(key)]
    if missing:
        parser.error("Use bash grpo/run_pipeline.sh so grpo/env.sh supplies the settings")
    for key in ("AFTERBURNER_MODEL_PATH", "AFTERBURNER_CHECKPOINT_DIR", "VENUS_GRPO_DATA_DIR",
                "CODECONTESTS_GRPO_DATA_DIR"):
        if os.environ.get(key):
            parser.error(f"{key} is not supported by this runner; outputs are controlled by --run-dir")
    # Treat a normal termination request like Ctrl-C so child cleanup still runs.
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        Pipeline(args.run_dir.resolve(), os.environ).run()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Pipeline stopped: {error}. Fix the cause and rerun the same command.") from error
    except KeyboardInterrupt:
        raise SystemExit("Pipeline interrupted. Rerun the same command to resume.") from None


if __name__ == "__main__":
    main()
