"""Emotion-ODE-Transformer model components."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchdiffeq import odeint

from .config import ExperimentConfig


class PostProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, post_hidden: torch.Tensor) -> torch.Tensor:
        return self.network(post_hidden)


class ODEFunc(nn.Module):
    """Deterministic autonomous vector field; intentionally contains no dropout."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        del time
        return self.network(state)


class _DurationScaledODEFunc(nn.Module):
    def __init__(self, base: ODEFunc, duration: torch.Tensor) -> None:
        super().__init__()
        self.base = base
        self.duration = duration

    def forward(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.duration * self.base(time, state)


class ContinuousTimeEncoder(nn.Module):
    """Propagate between posts with an ODE, then absorb the current post with a GRUCell."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        ode_hidden_dim: int,
        solver: str = "rk4",
        step_size: float = 0.25,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.solver = solver
        self.step_size = step_size
        self.ode_func = ODEFunc(state_dim, ode_hidden_dim)
        self.gru = nn.GRUCell(input_dim, state_dim)
        self.initial_state = nn.Parameter(torch.zeros(state_dim))

    def _propagate(self, state: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        if torch.count_nonzero(duration).item() == 0:
            return state
        integration_times = torch.tensor([0.0, 1.0], device=state.device, dtype=state.dtype)
        options = {"step_size": self.step_size} if self.solver in {"rk4", "euler", "midpoint"} else None
        return odeint(
            _DurationScaledODEFunc(self.ode_func, duration),
            state,
            integration_times,
            method=self.solver,
            options=options,
        )[-1]

    def forward(
        self,
        post_embedding: torch.Tensor,
        delta_t: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if post_embedding.ndim != 3 or delta_t.ndim != 2 or mask.ndim != 2:
            raise ValueError("Expected post embeddings [B,N,D], delta_t [B,N], mask [B,N].")
        batch, length, _ = post_embedding.shape
        if delta_t.shape != (batch, length) or mask.shape != (batch, length):
            raise ValueError("Sequence tensor shapes do not match.")
        if torch.any(delta_t < 0) or torch.any(delta_t > 1):
            raise ValueError("Normalized delta_t must be in [0, 1].")
        if torch.any(delta_t.masked_select(~mask.bool()) != 0):
            raise ValueError("Padding positions must have zero delta_t.")
        state = self.initial_state.unsqueeze(0).expand(batch, -1)
        states: list[torch.Tensor] = []
        valid_mask = mask.bool()
        for index in range(length):
            duration = delta_t[:, index].unsqueeze(-1)
            propagated = self._propagate(state, duration)
            updated = self.gru(post_embedding[:, index], propagated)
            state = torch.where(valid_mask[:, index].unsqueeze(-1), updated, state)
            states.append(state)
        return torch.stack(states, dim=1)


class GatedFusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.content = nn.Linear(dim, dim)
        self.time_state = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, content: torch.Tensor, time_state: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([content, time_state], dim=-1)))
        fused = gate * self.content(content) + (1.0 - gate) * self.time_state(time_state)
        return self.norm(fused)


class PrefixAttentionPooling(nn.Module):
    """Compute attention-pooled representations for every causal prefix in one pass."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(
        self, hidden: torch.Tensor, mask: torch.Tensor, return_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        valid = mask.bool()
        scores = self.score(torch.tanh(self.hidden(hidden))).squeeze(-1)
        unnormalized = torch.exp(scores.clamp(min=-20.0, max=20.0)) * valid.to(hidden.dtype)
        numerator = torch.cumsum(unnormalized.unsqueeze(-1) * hidden, dim=1)
        denominator = torch.cumsum(unnormalized, dim=1).clamp_min(1e-8).unsqueeze(-1)
        pooled = numerator / denominator
        weights = None
        if return_weights:
            length = hidden.shape[1]
            causal = torch.tril(torch.ones(length, length, device=hidden.device, dtype=hidden.dtype))
            prefix_numerators = causal.unsqueeze(0) * unnormalized.unsqueeze(1)
            weights = prefix_numerators / denominator
        return pooled, weights


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


class EOTModel(nn.Module):
    """Project post features, encode continuous time, model causal sequence, and classify."""

    def __init__(
        self,
        input_dim: int = 768,
        projection_hidden_dim: int = 256,
        model_dim: int = 128,
        projection_dropout: float = 0.2,
        ode_hidden_dim: int = 256,
        ode_solver: str = "rk4",
        ode_step_size: float = 0.25,
        max_posts: int = 256,
        n_heads: int = 4,
        transformer_layers: int = 2,
        feedforward_dim: int = 512,
        transformer_dropout: float = 0.2,
        classifier_hidden_dim: int = 64,
        classifier_dropout: float = 0.2,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.max_posts = max_posts
        self.projection = PostProjection(
            input_dim, projection_hidden_dim, model_dim, projection_dropout
        )
        self.continuous_time = ContinuousTimeEncoder(
            model_dim, model_dim, ode_hidden_dim, ode_solver, ode_step_size
        )
        self.fusion = GatedFusion(model_dim)
        self.position_embedding = nn.Embedding(max_posts, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=transformer_layers, enable_nested_tensor=False
        )
        self.pooling = PrefixAttentionPooling(model_dim)
        self.classifier = ClassificationHead(
            model_dim, classifier_hidden_dim, classifier_dropout
        )

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "EOTModel":
        projection = config.section("projection")
        ode = config.section("ode")
        transformer = config.section("transformer")
        classifier = config.section("classifier")
        return cls(
            input_dim=int(projection["input_dim"]),
            projection_hidden_dim=int(projection["hidden_dim"]),
            model_dim=int(projection["output_dim"]),
            projection_dropout=float(projection["dropout"]),
            ode_hidden_dim=int(ode["hidden_dim"]),
            ode_solver=str(ode["solver"]),
            ode_step_size=float(ode["step_size"]),
            max_posts=int(config.section("data")["max_posts"]),
            n_heads=int(transformer["n_heads"]),
            transformer_layers=int(transformer["num_layers"]),
            feedforward_dim=int(transformer["dim_feedforward"]),
            transformer_dropout=float(transformer["dropout"]),
            classifier_hidden_dim=int(classifier["hidden_dim"]),
            classifier_dropout=float(classifier["dropout"]),
            norm_first=bool(transformer["norm_first"]),
        )

    def forward(
        self,
        post_hidden: torch.Tensor,
        delta_t: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        batch, length, _ = post_hidden.shape
        if length > self.max_posts:
            raise ValueError(f"Sequence length {length} exceeds max_posts={self.max_posts}")
        valid = mask.bool()
        if torch.any(valid.sum(dim=1) < 1):
            raise ValueError("Every user must have at least one valid post.")
        content = self.projection(post_hidden)
        time_state = self.continuous_time(content, delta_t, valid)
        fused = self.fusion(content, time_state)
        positions = torch.arange(length, device=post_hidden.device)
        fused = fused + self.position_embedding(positions).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(length, length, device=post_hidden.device, dtype=torch.bool), diagonal=1
        )
        sequence_hidden = self.transformer(
            fused,
            mask=causal_mask,
            src_key_padding_mask=~valid,
        )
        prefix_representations, attention_weights = self.pooling(
            sequence_hidden, valid, return_attention
        )
        prefix_logits = self.classifier(prefix_representations)
        prefix_logits = prefix_logits.masked_fill(~valid, 0.0)
        lengths = valid.sum(dim=1).long()
        final_logits = prefix_logits.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        return {
            "final_logits": final_logits,
            "prefix_logits": prefix_logits,
            "attention_weights": attention_weights,
            "sequence_hidden": sequence_hidden,
            "time_states": time_state,
        }


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}
