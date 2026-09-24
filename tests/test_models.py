import unittest

import torch

from emotion_project.models import MultimodalEmotionModel, gradient_reverse


class ModelTests(unittest.TestCase):
    def test_tensor_shapes_and_gate_weights(self):
        model = MultimodalEmotionModel(10, 8, hidden_dim=16, embedding_dim=12, model_type="gated_quality")
        out = model(
            torch.randn(4, 10),
            torch.randn(4, 8),
            torch.tensor([[1, 1], [1, 0], [0, 1], [1, 1]], dtype=torch.float32),
            torch.ones(4, 2),
        )
        self.assertEqual(tuple(out["logits"].shape), (4, 2))
        self.assertTrue(torch.allclose(out["weights"].sum(dim=1), torch.ones(4), atol=1e-6))
        self.assertEqual(float(out["weights"][1, 1].detach()), 0.0)
        self.assertEqual(float(out["weights"][2, 0].detach()), 0.0)

    def test_both_modalities_missing_is_explicit_and_finite(self):
        model = MultimodalEmotionModel(10, 8, hidden_dim=16, embedding_dim=12, model_type="gated_quality")
        out = model(torch.randn(1, 10), torch.randn(1, 8), torch.zeros(1, 2), torch.zeros(1, 2))
        self.assertTrue(bool(out["insufficient"][0]))
        self.assertTrue(torch.isfinite(out["logits"]).all())
        self.assertEqual(out["weights"].sum().item(), 0.0)

    def test_gradient_reversal_sign(self):
        x = torch.tensor([2.0], requires_grad=True)
        y = gradient_reverse(x, 0.5) ** 2
        y.backward()
        self.assertAlmostEqual(float(x.grad), -2.0)

    def test_finite_forward_backward(self):
        model = MultimodalEmotionModel(10, 8, hidden_dim=16, embedding_dim=12, model_type="concat")
        out = model(torch.randn(3, 10), torch.randn(3, 8), torch.ones(3, 2), torch.ones(3, 2))
        loss = torch.nn.functional.cross_entropy(out["logits"], torch.tensor([0, 1, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
