"""User-balanced final and prefix classification losses."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import nn


def compute_positive_weight(labels: Sequence[int | float]) -> float:
    positives = sum(float(value) == 1.0 for value in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Training labels must contain both classes.")
    return negatives / positives


class EOTLoss(nn.Module):
    def __init__(
        self,
        positive_weight: float,
        checkpoints: Sequence[int],
        final_weight: float = 1.0,
        prefix_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.register_buffer("positive_weight", torch.tensor(float(positive_weight)))
        self.checkpoints = tuple(sorted({int(value) for value in checkpoints if int(value) > 0}))
        self.final_weight = float(final_weight)
        self.prefix_weight = float(prefix_weight)

    def _binary_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.positive_weight.to(logits.device),
            reduction="none",
        )

    def forward(
        self,
        final_logits: torch.Tensor,
        prefix_logits: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        final_per_user = self._binary_loss(final_logits, labels)
        prefix_per_user: list[torch.Tensor] = []
        for user_index in range(len(labels)):
            valid_indices = [
                checkpoint - 1
                for checkpoint in self.checkpoints
                if checkpoint <= int(lengths[user_index].item())
            ]
            if not valid_indices:
                valid_indices = [int(lengths[user_index].item()) - 1]
            user_logits = prefix_logits[user_index, valid_indices]
            user_labels = labels[user_index].expand_as(user_logits)
            prefix_per_user.append(self._binary_loss(user_logits, user_labels).mean())
        prefix_loss = torch.stack(prefix_per_user).mean()
        final_loss = final_per_user.mean()
        total = self.final_weight * final_loss + self.prefix_weight * prefix_loss
        return total, {"final_loss": final_loss.detach(), "prefix_loss": prefix_loss.detach()}
