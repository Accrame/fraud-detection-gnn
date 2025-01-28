"""Custom losses for imbalanced fraud detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss — down-weights easy examples so the model focuses on hard ones.
    Made a bigger difference than class weights for our imbalanced data.

    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma=2.0, alpha=None, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")

        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCrossEntropy(nn.Module):
    """CE with automatic class weight computation from batch frequencies."""

    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.weight is None:
            # compute weights on the fly from this batch
            unique, counts = torch.unique(targets, return_counts=True)
            weights = 1.0 / counts.float()
            weights = weights / weights.sum() * len(unique)

            weight = torch.ones(inputs.size(1), device=inputs.device)
            for u, w in zip(unique, weights):
                weight[u] = w
        else:
            weight = self.weight.to(inputs.device)

        return F.cross_entropy(inputs, targets, weight=weight, reduction=self.reduction)


class LabelSmoothingLoss(nn.Module):
    """Label smoothing to prevent overconfident predictions."""

    def __init__(self, smoothing=0.1, reduction="mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)

        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(inputs, dim=1)
        loss = -(smooth_targets * log_probs).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
