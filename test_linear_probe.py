"""Checks for feature equivalence, normalization, and resumable extraction."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

import linear_probe as lp


class LinearProbeTests(unittest.TestCase):
    def test_backbone_matches_reference_feature(self):
        tokenizer = AutoTokenizer.from_pretrained(lp.DEFAULT_MODEL, local_files_only=True)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=512,
        )).eval()
        description = "Read two integers and print their sum."
        ids = tokenizer.apply_chat_template(lp.build_messages(description), tokenize=True,
                                            add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True)["input_ids"]
        with torch.inference_mode():
            reference = model(ids, output_hidden_states=True).hidden_states[-1][0, -1].numpy()
        actual = lp.embed_problem(description, tokenizer, model.model, "cpu", 512)
        np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)
        text = tokenizer.decode(ids[0])
        # This cold-start tokenizer uses System/Human/Assistant role prefixes.
        self.assertTrue(text.endswith("Assistant:"))
        self.assertIn("<solution>", text)
        self.assertIn("Python", text)
        self.assertNotIn("boxed", text)
        with self.assertRaisesRegex(ValueError, "exceeding limit"):
            lp.embed_problem(description, tokenizer, model.model, "cpu", 2)

    def test_scaler_fold_preserves_predictions(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(100, 4)).astype(np.float32)
        x[:, 0] = 7  # Also exercise a zero-variance feature.
        y = rng.normal(1900, 500, size=(100, 1)).astype(np.float32)
        sx, sy = StandardScaler().fit(x), StandardScaler().fit(y)
        probe = torch.nn.Linear(4, 1)
        with torch.no_grad():
            expected = sy.inverse_transform(probe(torch.from_numpy(sx.transform(x))).numpy())
            actual = lp.fold_scalers(probe, sx, sy)(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-3)

    def test_split_and_training_ignore_held_out_statistics(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(100, 4)).astype(np.float32)
        y = (1900 + 400 * x[:, :1]).astype(np.float32)
        train, val, test = lp.split_indices(100)
        self.assertEqual(tuple(map(len, (train, val, test))), (64, 16, 20))
        self.assertEqual(len(set(train) | set(val) | set(test)), 100)
        first, _, _, _ = lp.fit_probe(x, y, epochs=1)
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[np.r_[val, test]] += 1000
        changed_y[np.r_[val, test]] += 10000
        second, _, _, _ = lp.fit_probe(changed_x, changed_y, epochs=1)
        # With one epoch, changing held-out data cannot change trained weights.
        torch.testing.assert_close(first.weight, second.weight, rtol=0, atol=0)
        torch.testing.assert_close(first.bias, second.bias, rtol=0, atol=0)

    def test_partial_cache_resumes_and_mismatch_fails(self):
        records = [{"name": str(i), "description": str(i), "rating": 800} for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(model="test-model", max_length=512, output_dir=Path(directory))
            metadata = lp.cache_metadata(args, records, "float32")
            np.savez(args.output_dir / "embeddings.npz", embeddings=np.ones((1, 4), dtype=np.float32),
                     metadata=json.dumps(metadata))
            with patch.object(lp, "load_backbone", return_value=(None, None)), patch.object(
                lp, "embed_problem", return_value=np.zeros(4, dtype=np.float32)
            ) as embed:
                result, _ = lp.extract_embeddings(args, records, "cpu", "float32")
                self.assertEqual(embed.call_count, 2)
                self.assertEqual(result.shape, (3, 4))
            with patch.object(lp, "load_backbone") as load:
                lp.extract_embeddings(args, records, "cpu", "float32")
                load.assert_not_called()
            args.max_length = 256
            with self.assertRaisesRegex(ValueError, "does not match"):
                lp.extract_embeddings(args, records, "cpu", "float32")

    def test_saved_probe_reloads_without_pickle_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output_dir=Path(directory), epochs=2)
            rng = np.random.default_rng(42)
            features = rng.normal(size=(20, 4)).astype(np.float32)
            records = [{"name": str(i), "rating": 800 + i * 100} for i in range(20)]
            with patch.object(lp, "load_records", return_value=records), patch.object(
                lp, "extract_embeddings", return_value=(features, {})
            ):
                lp.train(args, "cpu", "float32")
            saved = torch.load(args.output_dir / "probe.pt", weights_only=True)
            probe = torch.nn.Linear(saved["hidden_size"], 1)
            probe.load_state_dict(saved["state_dict"])
            report = json.loads((args.output_dir / "metrics.json").read_text())
            self.assertEqual(report["test"]["n"], 4)
            self.assertEqual(report["normalization_fit"], "train_only")


if __name__ == "__main__":
    unittest.main()
