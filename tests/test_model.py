import unittest

import torch

from eot.model import ContinuousTimeEncoder, EOTModel


def make_model() -> EOTModel:
    torch.manual_seed(11)
    return EOTModel(
        input_dim=8,
        projection_hidden_dim=12,
        model_dim=8,
        projection_dropout=0.0,
        ode_hidden_dim=12,
        ode_step_size=1.0,
        max_posts=8,
        n_heads=2,
        transformer_layers=1,
        feedforward_dim=16,
        transformer_dropout=0.0,
        classifier_hidden_dim=4,
        classifier_dropout=0.0,
    ).eval()


class ModelTests(unittest.TestCase):
    def test_zero_duration_does_not_propagate_state(self):
        encoder = ContinuousTimeEncoder(8, 8, 12, step_size=1.0)
        state = torch.randn(3, 8)
        self.assertTrue(torch.equal(encoder._propagate(state, torch.zeros(3, 1)), state))

    def test_future_posts_do_not_change_earlier_prefixes(self):
        model = make_model()
        x = torch.randn(2, 5, 8)
        changed = x.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 20
        delta = torch.rand(2, 5)
        delta[:, 0] = 0
        mask = torch.ones(2, 5, dtype=torch.bool)
        first = model(x, delta, mask)["prefix_logits"]
        second = model(changed, delta, mask)["prefix_logits"]
        self.assertTrue(torch.allclose(first[:, :3], second[:, :3], atol=1e-5))

    def test_padding_content_does_not_change_prediction(self):
        model = make_model()
        x = torch.randn(2, 5, 8)
        changed = x.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 20
        delta = torch.rand(2, 5)
        delta[:, 0] = 0
        delta[:, 3:] = 0
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
        first = model(x, delta, mask)["final_logits"]
        second = model(changed, delta, mask)["final_logits"]
        self.assertTrue(torch.allclose(first, second, atol=1e-5))

    def test_single_post_and_attention_shapes(self):
        model = make_model()
        output = model(
            torch.randn(2, 1, 8),
            torch.zeros(2, 1),
            torch.ones(2, 1, dtype=torch.bool),
            return_attention=True,
        )
        self.assertEqual(output["final_logits"].shape, (2,))
        self.assertEqual(output["prefix_logits"].shape, (2, 1))
        self.assertEqual(output["attention_weights"].shape, (2, 1, 1))


if __name__ == "__main__":
    unittest.main()
