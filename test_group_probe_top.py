"""CPU tests for the advanced five-group model's mathematical components."""

import unittest

import numpy as np
import torch

import group_probe as gp
import group_probe_top as top


class TopGroupProbeTests(unittest.TestCase):
    def test_earth_mover_prefers_correct_ordered_class(self):
        groups = torch.tensor([0, 2, 4])
        correct = torch.full((3, 5), -5.0)
        correct[torch.arange(3), groups] = 5.0
        reversed_logits = torch.flip(correct, dims=[1])
        self.assertLess(
            float(top.earth_mover_loss(correct, groups)),
            float(top.earth_mover_loss(reversed_logits, groups)),
        )

    def test_pairwise_rank_loss_prefers_correct_order(self):
        ratings = torch.tensor([-1.0, 0.0, 1.0])
        correct = torch.tensor([-1.0, 0.0, 1.0])
        reversed_scores = -correct
        self.assertLess(
            float(top.pairwise_rank_loss(correct, ratings, 0.1)),
            float(top.pairwise_rank_loss(reversed_scores, ratings, 0.1)),
        )

    def test_pairwise_rank_loss_handles_no_pairs(self):
        scores = torch.tensor([0.1, 0.2], requires_grad=True)
        ratings = torch.tensor([0.0, 0.01])
        loss = top.pairwise_rank_loss(scores, ratings, 1.0)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(scores.grad)

    def test_threshold_optimizer_finds_separable_groups(self):
        scores = np.arange(50, dtype=np.float64)
        labels = np.repeat(np.arange(5), 10)
        thresholds = top.fit_exact_ordered_thresholds(scores, labels)
        predicted = top.groups_from_thresholds(scores, thresholds)
        np.testing.assert_array_equal(predicted, labels)
        self.assertEqual(len(thresholds), 4)
        self.assertTrue(np.all(np.diff(thresholds) > 0))

    def test_threshold_optimizer_does_not_split_ties(self):
        scores = np.repeat(np.arange(5, dtype=np.float64), 4)
        labels = np.repeat(np.arange(5), 4)
        thresholds = top.fit_exact_ordered_thresholds(scores, labels)
        predicted = top.groups_from_thresholds(scores, thresholds)
        np.testing.assert_array_equal(predicted, labels)

    def test_calibration_round_trip(self):
        labels = np.repeat(np.arange(5), 20)
        base = labels.astype(np.float64)
        calibration = top.calibrate_decoder(base, base * 2, base * 10, labels)
        predicted, score = top.decode_with_calibration(
            base, base * 2, base * 10, calibration
        )
        np.testing.assert_array_equal(predicted, labels)
        self.assertTrue(np.all(np.diff(score) >= 0))
        self.assertAlmostEqual(
            calibration["validation_report"]["accuracy"], 1.0
        )

    def test_lora_target_layers_are_restricted(self):
        names, layers = top.lora_targets(36, 4)
        self.assertEqual(layers, [32, 33, 34, 35])
        self.assertEqual(len(names), 4 * 7)
        self.assertTrue(all(name.startswith("layers.") for name in names))

    def test_ordinal_probabilities_are_monotonic(self):
        head = top.MonotonicOrdinalHead(8, 5)
        probabilities = torch.sigmoid(head(torch.randn(12, 8)))
        self.assertTrue(torch.all(probabilities[:, :-1] >= probabilities[:, 1:]))

    def test_calibration_selection_split_is_disjoint(self):
        indices = np.arange(100)
        groups = np.repeat(np.arange(5), 20)
        calibration, selection = top.split_calibration_selection(indices, groups)
        self.assertEqual(len(calibration) + len(selection), len(indices))
        self.assertFalse(set(calibration) & set(selection))
        for group in range(5):
            self.assertGreater(np.sum(groups[calibration] == group), 0)
            self.assertGreater(np.sum(groups[selection] == group), 0)

    def test_three_group_edges(self):
        ratings = np.concatenate(
            [np.full(100, 800.0), np.full(100, 1600.0), np.full(100, 2800.0)]
        )
        idx = np.arange(len(ratings))
        edges = gp.fit_quintile_edges(ratings, idx, 3)
        self.assertEqual(len(edges), 2)
        groups = gp.ratings_to_groups(ratings, edges)
        self.assertEqual(sorted(np.unique(groups).tolist()), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
