"""Checks that model artifacts are resolved into the repository cache."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import artifact_cache as cache


def make_snapshot(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    (path / "tokenizer_config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")


class ArtifactCacheTests(unittest.TestCase):
    def test_existing_snapshot_is_reused_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "existing"
            make_snapshot(snapshot)
            with patch("huggingface_hub.snapshot_download") as download:
                self.assertEqual(cache.ensure_model(snapshot), snapshot.resolve())
            download.assert_not_called()

    def test_hugging_face_id_downloads_into_model_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            model_cache = Path(directory) / "model-cache"

            def download(repo_id, local_dir):
                self.assertEqual(repo_id, "owner/model")
                make_snapshot(Path(local_dir))

            with patch("huggingface_hub.snapshot_download", side_effect=download) as mocked:
                resolved = cache.ensure_model("owner/model", model_cache=model_cache)

            self.assertEqual(resolved, (model_cache / "huggingface/owner/model").resolve())
            mocked.assert_called_once()

    def test_default_hub_id_uses_afterburner_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            model_cache = Path(directory) / "model-cache"

            def download(repo_id, local_dir, **options):
                self.assertEqual(repo_id, cache.DEFAULT_MODEL_ID)
                self.assertEqual(options["revision"], cache.DEFAULT_MODEL_REVISION)
                self.assertIn("model*.safetensors", options["allow_patterns"])
                make_snapshot(Path(local_dir))

            with patch("huggingface_hub.snapshot_download", side_effect=download) as mocked:
                resolved = cache.ensure_model(cache.DEFAULT_MODEL_ID, model_cache=model_cache)
                self.assertEqual(cache.ensure_model(cache.DEFAULT_MODEL_ID, model_cache=model_cache), resolved)
                mocked.assert_called_once()

            self.assertEqual(
                resolved,
                (model_cache / cache.DEFAULT_MODEL_RELATIVE).resolve(),
            )

    def test_incomplete_sharded_snapshot_is_not_treated_as_available(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "partial"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            (snapshot / "tokenizer.json").write_text("{}")
            (snapshot / "tokenizer_config.json").write_text("{}")
            (snapshot / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"layer": "model-00001-of-00002.safetensors"}
            }))
            self.assertFalse(cache.model_snapshot_available(snapshot))

    def test_dataset_download_is_pinned_and_uses_requested_cache(self):
        with tempfile.TemporaryDirectory() as directory, patch("datasets.load_dataset") as load:
            cache.load_cached_dataset(cache.DEFAULT_DATASET_ID, "train", Path(directory))
            load.assert_called_once_with(
                cache.DEFAULT_DATASET_ID, split="train", cache_dir=str(Path(directory).resolve()),
                revision=cache.DEFAULT_DATASET_REVISION,
            )


if __name__ == "__main__":
    unittest.main()
