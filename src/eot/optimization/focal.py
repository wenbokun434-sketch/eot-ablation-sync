"""O12: focal final branch with the unchanged user-balanced BCE prefix branch."""
import math
import torch
import torch.nn.functional as F
from eot.losses import EOTLoss
from .losses import OptimizationBCELoss
from .splits import checked_ids, digest, label_map


class FinalFocalLoss(OptimizationBCELoss):
    def __init__(self, training_labels, checkpoints=(1, 2, 4, 8, 16, 32, 64, 128, 256),
                 *, alpha=1.0, prefix_weight=.5, gamma=1.0):
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        super().__init__(training_labels, checkpoints, alpha=alpha, prefix_weight=prefix_weight)
        self.gamma = float(gamma)
        # Delegate all checkpoint, padding and per-user averaging to EOTLoss.
        self.prefix_only = EOTLoss(float(self.positive_weight), checkpoints,
                                   final_weight=0.0, prefix_weight=prefix_weight)

    def forward(self, final_logits, prefix_logits, labels, lengths):
        if self.gamma == 0:
            return super().forward(final_logits, prefix_logits, labels, lengths)
        positive = self.positive_weight * labels * torch.exp(-self.gamma * F.softplus(final_logits)) * F.softplus(-final_logits)
        negative = (1 - labels) * torch.exp(-self.gamma * F.softplus(-final_logits)) * F.softplus(final_logits)
        final_loss = (positive + negative).mean()
        prefix_total, diagnostics = self.prefix_only(final_logits, prefix_logits, labels, lengths)
        return final_loss + prefix_total, {"final_loss": final_loss.detach(),
                                           "prefix_loss": diagnostics["prefix_loss"]}


def build_loss(config, users, train_user_ids):
    settings = config.loss
    if settings.prefix_loss != "weighted_bce" or settings.final_weight != 1:
        raise ValueError("Only the final loss branch may change")
    ids = checked_ids(train_user_ids, "conditional loss fit")
    labels = label_map(users, ids)
    kwargs = dict(alpha=settings.alpha, prefix_weight=settings.prefix_weight)
    if settings.final_loss == "weighted_focal":
        loss = FinalFocalLoss([labels[u] for u in ids], settings.prefix_checkpoints,
                              gamma=settings.gamma, **kwargs)
    elif settings.final_loss == "weighted_bce" and settings.gamma == 0:
        loss = OptimizationBCELoss([labels[u] for u in ids], settings.prefix_checkpoints, **kwargs)
    else:
        raise ValueError("Unexpected conditional loss")
    loss.fit_user_ids = tuple(ids)
    loss.fit_labels_sha256 = digest([[u, labels[u]] for u in ids])
    loss.config_hash = config.config_hash
    return loss
