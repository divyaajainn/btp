from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.autograd import Function


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def gradient_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReversal.apply(x, lambd)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalEmotionModel(nn.Module):
    def __init__(
        self,
        eeg_dim: int,
        face_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 128,
        model_type: Literal[
            "eeg_only",
            "face_only",
            "concat",
            "gated",
            "gated_quality",
            "concat_adversarial",
            "gated_quality_adversarial",
        ] = "concat",
        num_subjects: int = 0,
    ):
        super().__init__()
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.eeg_encoder = MLPEncoder(eeg_dim, embedding_dim, hidden_dim)
        self.face_encoder = MLPEncoder(face_dim, embedding_dim, hidden_dim)
        self.eeg_head = nn.Linear(embedding_dim, 2)
        self.face_head = nn.Linear(embedding_dim, 2)
        self.concat_head = nn.Sequential(nn.Linear(embedding_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        gate_input_dim = embedding_dim * 2 + 2 + 2
        self.gate = nn.Sequential(nn.Linear(gate_input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.fused_head = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        subject_dim = embedding_dim * 2 if model_type == "concat_adversarial" else embedding_dim
        self.subject_head = nn.Linear(subject_dim, num_subjects) if num_subjects > 0 else None

    def forward(
        self,
        eeg: torch.Tensor,
        face: torch.Tensor,
        availability: torch.Tensor,
        quality: torch.Tensor | None = None,
        adv_lambda: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if quality is None:
            quality = torch.zeros_like(availability)
        availability = availability.float()
        quality = quality.float()
        insufficient = availability.sum(dim=1) == 0
        eeg_emb = self.eeg_encoder(eeg) * availability[:, 0:1]
        face_emb = self.face_encoder(face) * availability[:, 1:2]

        if self.model_type == "eeg_only":
            logits = self.eeg_head(eeg_emb)
            embedding = eeg_emb
            weights = torch.stack([availability[:, 0], torch.zeros_like(availability[:, 0])], dim=1)
        elif self.model_type == "face_only":
            logits = self.face_head(face_emb)
            embedding = face_emb
            weights = torch.stack([torch.zeros_like(availability[:, 1]), availability[:, 1]], dim=1)
        elif self.model_type in {"concat", "concat_adversarial"}:
            embedding = torch.cat([eeg_emb, face_emb], dim=1)
            logits = self.concat_head(embedding)
            weights = _normalize_available(availability)
        elif self.model_type in {"gated", "gated_quality", "gated_quality_adversarial"}:
            gate_quality = quality if self.model_type in {"gated_quality", "gated_quality_adversarial"} else torch.zeros_like(quality)
            gate_input = torch.cat([eeg_emb, face_emb, gate_quality, availability], dim=1)
            raw_gate = self.gate(gate_input)
            masked_gate = raw_gate.masked_fill(availability <= 0, -1e9)
            weights = torch.softmax(masked_gate, dim=1)
            weights = torch.where(availability.sum(dim=1, keepdim=True) > 0, weights * availability, torch.zeros_like(weights))
            weights = _normalize_available(weights)
            embedding = weights[:, 0:1] * eeg_emb + weights[:, 1:2] * face_emb
            logits = self.fused_head(embedding)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        logits = torch.where(insufficient[:, None], torch.zeros_like(logits), logits)
        out = {"logits": logits, "weights": weights, "embedding": embedding, "insufficient": insufficient}
        if self.subject_head is not None:
            out["subject_logits"] = self.subject_head(gradient_reverse(embedding, adv_lambda))
        return out


def _normalize_available(weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum(dim=1, keepdim=True)
    return torch.where(denom > 0, weights / denom.clamp_min(1e-12), torch.zeros_like(weights))


def apply_modality_dropout(
    availability: torch.Tensor,
    dropout_p: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if dropout_p <= 0:
        return availability
    keep = torch.rand(availability.shape, device=availability.device, generator=generator) > dropout_p
    dropped = availability * keep.float()
    both_missing = dropped.sum(dim=1) == 0
    if both_missing.any():
        available_indices = availability[both_missing].argmax(dim=1)
        dropped[both_missing] = 0
        dropped[both_missing, available_indices] = availability[both_missing, available_indices]
    return dropped
