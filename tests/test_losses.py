import unittest

import torch

from eot.losses import EOTLoss, compute_positive_weight


class LossTests(unittest.TestCase):
    def test_positive_weight_and_user_balanced_prefix_loss_are_finite(self):
        self.assertEqual(compute_positive_weight([0, 0, 0, 1]), 3.0)
        loss_fn = EOTLoss(3.0, checkpoints=[1, 2, 4], final_weight=1.0, prefix_weight=0.5)
        final = torch.tensor([0.1, -0.2], requires_grad=True)
        prefixes = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [-0.2, -0.1, 0.0, 0.1]], requires_grad=True
        )
        loss, parts = loss_fn(
            final, prefixes, torch.tensor([1.0, 0.0]), torch.tensor([4, 2])
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(parts["final_loss"]))
        self.assertTrue(torch.isfinite(parts["prefix_loss"]))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
