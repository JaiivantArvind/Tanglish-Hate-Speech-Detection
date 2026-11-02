"""Losses for the imbalanced 6-class task and the hierarchical label heads."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .models import LABEL_TREE, NUM_CLASSES, tree_targets


def class_weights(labels, n_classes: int, mode: str) -> torch.Tensor:
    """Per-class weights normalised to mean 1 over classes that occur.

    balanced: inverse frequency (the guide's choice, ~12x for the rarest class here).
    sqrt:     square root of inverse frequency, a milder and more stable reweighting.
    """
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    if mode == "none":
        w = np.ones(n_classes)
    elif mode in ("balanced", "sqrt"):
        w = np.zeros(n_classes)
        seen = counts > 0
        w[seen] = counts[seen].sum() / counts[seen]
        if mode == "sqrt":
            w = np.sqrt(w)
    else:
        raise ValueError(f"unknown class_weight mode {mode!r}")
    w[counts > 0] /= w[counts > 0].mean()
    return torch.tensor(w, dtype=torch.float32)


def level_weights(labels, mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Class weights for each level of the label tree, from the 6-way train labels."""
    paths = np.array([LABEL_TREE[int(y)] for y in labels])
    a, b, c = paths[:, 0], paths[:, 1], paths[:, 2]
    return class_weights(a, 3, mode), class_weights(b[b >= 0], 2, mode), class_weights(c[c >= 0], 3, mode)


def classification_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None,
                        kind: str = "ce", gamma: float = 2.0, smoothing: float = 0.0) -> torch.Tensor:
    """Weighted CE or focal loss; targets of -1 are ignored. Log-probabilities are valid logits."""
    valid = target >= 0
    if not valid.any():
        return logits.sum() * 0.0
    logits, target = logits[valid].float(), target[valid]
    if kind == "ce":
        return F.cross_entropy(logits, target, weight=weight, label_smoothing=smoothing)
    if kind == "focal":
        ce = F.cross_entropy(logits, target, reduction="none", label_smoothing=smoothing)
        pt = torch.exp(-F.cross_entropy(logits, target, reduction="none"))
        w = weight[target] if weight is not None else torch.ones_like(ce)
        return ((1 - pt) ** gamma * ce * w).sum() / w.sum()
    raise ValueError(f"unknown loss {kind!r}")


class LossComputer:
    def __init__(self, train_labels, label_tree: bool, kind: str, weight_mode: str,
                 gamma: float = 2.0, smoothing: float = 0.0):
        self.label_tree = label_tree
        self.kind, self.gamma, self.smoothing = kind, gamma, smoothing
        if label_tree:
            self.weights = list(level_weights(train_labels, weight_mode))
        else:
            self.weights = [class_weights(train_labels, NUM_CLASSES, weight_mode)]

    def to(self, device):
        self.weights = [w.to(device) for w in self.weights]
        return self

    def __call__(self, out: dict, labels: torch.Tensor) -> torch.Tensor:
        loss = lambda logits, target, w: classification_loss(logits, target, w, self.kind, self.gamma, self.smoothing)
        if not self.label_tree:
            return loss(out["scores"], labels, self.weights[0])
        ta, tb, tc = tree_targets(labels)
        wa, wb, wc = self.weights
        return loss(out["logits_a"], ta, wa) + loss(out["logits_b"], tb, wb) + loss(out["logits_c"], tc, wc)
