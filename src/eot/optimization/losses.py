"""Configurable BCE, reusing the frozen EOT loss and prefix semantics.

Use ``build_loss(config, users, train_user_ids)`` with the actual gradient
training IDs (G for inner training, S for final retraining). Never fit weights
from a minibatch, early-stop, holdout, calibration or test labels.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from eot.losses import EOTLoss, compute_positive_weight
from .config import OptimizationConfig
from .splits import checked_ids, digest, label_map


class OptimizationBCELoss(EOTLoss):
    """Same forward/diagnostic contract as EOTLoss; alpha scales both branches."""

    def __init__(self, training_labels: Sequence[int | float],
                 checkpoints: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256),
                 *, alpha: float = 1.0, prefix_weight: float = 0.5) -> None:
        for name, value in (("alpha", alpha), ("prefix_weight", prefix_weight)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (name == "alpha" and value == 0)):
                raise ValueError(f"Invalid {name}")
        labels = list(training_labels)
        if not labels or any(value not in (0, 1) for value in labels):
            raise ValueError("Training labels must be binary and nonempty")
        points = tuple(checkpoints)
        if not points or any(type(k) is not int or k <= 0 for k in points):
            raise ValueError("Checkpoints must be positive integers")
        base_weight = compute_positive_weight(labels)
        weight = alpha * base_weight
        # The reference buffer is float32; reject overflow/underflow in that dtype.
        if not math.isfinite(weight) or not 1.1754943508222875e-38 <= weight <= 3.4028234663852886e38:
            raise ValueError("Positive weight is outside the normal float32 range")
        super().__init__(weight, points, final_weight=1.0, prefix_weight=prefix_weight)
        self.alpha = float(alpha)
        self.base_positive_weight = base_weight


def build_loss(config: OptimizationConfig, users, train_user_ids) -> OptimizationBCELoss:
    """Select labels by explicit training membership, with fit provenance.

    The caller supplies IDs from the verified split; other users' labels are
    neither validated nor counted. Membership/role isolation is enforced by O04.
    """
    settings = config.loss
    if (settings.final_loss != "weighted_bce" or settings.prefix_loss != "weighted_bce"
            or settings.gamma != 0 or settings.final_weight != 1):
        raise ValueError("O05 supports weighted BCE only")
    ids = checked_ids(train_user_ids, "loss fit")
    labels = label_map(users, ids)
    loss = OptimizationBCELoss([labels[u] for u in ids], settings.prefix_checkpoints,
                               alpha=settings.alpha, prefix_weight=settings.prefix_weight)
    loss.fit_user_ids = tuple(ids)
    loss.fit_labels_sha256 = digest([[u, labels[u]] for u in ids])
    loss.config_hash = config.config_hash
    return loss
