"""Tests for LoRA group-probe helpers (CPU-only)."""

import unittest

import numpy as np
import torch

import group_probe_lora as gpl


class LoRAGroupHelpers(unittest.TestCase):
    def test_soft_ordinal_targets_normalized(self):
        groups = np.asarray([0, 1, 2, 3, 4])
        soft = gpl.soft_ordinal_targets(groups, neighbor=0.15)
        self.assertEqual(soft.shape, (5, 5))
        np.testing.assert_allclose(soft.sum(axis=1), 1.0, atol=1e-6)
        self.assertGreater(soft[2, 2], soft[2, 1])
        self.assertGreater(soft[0, 0], soft[0, 1])

    def test_batch_loss_finite(self):
        class_logits = torch.randn(4, 5)
        coral_logits = torch.randn(4, 4)
        soft = torch.tensor(gpl.soft_ordinal_targets(np.array([0, 1, 2, 4])))
        coral = torch.tensor(gpl.gp.coral_targets(np.array([0, 1, 2, 4])))
        loss, ce, bce = gpl.batch_loss(class_logits, coral_logits, soft, coral, 0.25)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(ce))
        self.assertTrue(torch.isfinite(bce))


if __name__ == "__main__":
    unittest.main()
