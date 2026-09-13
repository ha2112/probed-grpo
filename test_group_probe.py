"""Unit tests for five-group ordinal probe helpers."""

import unittest

import numpy as np
import torch

import group_probe as gp


class GroupProbeTests(unittest.TestCase):
    def test_quintile_edges_and_mapping_are_ordered(self):
        ratings = np.asarray([800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 3000], dtype=np.float64)
        train_idx = np.arange(len(ratings))
        edges = gp.fit_quintile_edges(ratings, train_idx, num_groups=5)
        self.assertEqual(len(edges), 4)
        self.assertTrue(np.all(np.diff(edges) >= 0))
        groups = gp.ratings_to_groups(ratings, edges)
        self.assertTrue(groups.min() >= 0)
        self.assertTrue(groups.max() <= 4)
        # Higher rating never maps to a lower group than a lower rating.
        order = np.argsort(ratings)
        self.assertTrue(np.all(np.diff(groups[order]) >= 0))

    def test_coral_targets_and_decode(self):
        groups = np.asarray([0, 1, 2, 3, 4])
        targets = gp.coral_targets(groups, num_groups=5)
        self.assertEqual(targets.shape, (5, 4))
        np.testing.assert_array_equal(targets[0], [0, 0, 0, 0])
        np.testing.assert_array_equal(targets[4], [1, 1, 1, 1])
        # Perfect logits: large positive for true thresholds.
        logits = torch.logit(torch.clamp(torch.from_numpy(targets), 1e-4, 1 - 1e-4))
        pred = gp.coral_predict_group(logits).numpy()
        np.testing.assert_array_equal(pred, groups)

    def test_classification_report_perfect(self):
        y = np.arange(5).repeat(3)
        report = gp.classification_report(y, y, y.astype(np.float64))
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["adjacent_accuracy"], 1.0)
        self.assertEqual(report["qwk"], 1.0)
        self.assertAlmostEqual(report["selection_score"], 1.0 + 0.15)

    def test_selection_score_prefers_adjacent(self):
        a = {"qwk": 0.60, "adjacent_accuracy": 0.90}
        b = {"qwk": 0.60, "adjacent_accuracy": 0.80}
        self.assertGreater(gp.selection_score(a), gp.selection_score(b))

    def test_coral_mlp_forward_shape(self):
        model = gp.CoralMLP(input_dim=16, hidden_dim=32, dropout=0.0, num_groups=5)
        x = torch.randn(7, 16)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (7, 4))


if __name__ == "__main__":
    unittest.main()
