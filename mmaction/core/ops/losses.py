import torch
import torch.nn.functional as F
import numpy as np


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0.0, p * p.log(), torch.zeros_like(p)).sum(dim=dim, keepdim=keepdim)


def focal_loss(input_values, gamma):
    return (1.0 - torch.exp(-input_values)) ** gamma * input_values


class MaxEntropyLoss:
    def __init__(self, scale=1.0):
        super(MaxEntropyLoss, self).__init__()

        self.scale = scale
        assert self.scale > 0.0

    def forward(self, cos_theta):
        probs = F.softmax(self.scale * cos_theta, dim=-1)

        entropy_values = entropy(probs, dim=-1)
        losses = np.log(cos_theta.size(-1)) - entropy_values

        return losses.mean()


class CrossEntropy:
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, logits, target):
        return self.weight * F.cross_entropy(logits, target, reduction='none')


class NormalizedCrossEntropy:
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, logits, target):
        log_softmax = F.log_softmax(logits, dim=-1)

        target_log_softmax = log_softmax[torch.arange(target.size(0), device=target.device), target]
        sum_log_softmax = log_softmax.sum(dim=-1)
        losses = self.weight * target_log_softmax / sum_log_softmax

        return losses


class ReverseCrossEntropy:
    def __init__(self, scale=4.0, weight=1.0):
        self.weight = weight * abs(float(scale))

    def __call__(self, logits, target):
        all_probs = F.softmax(logits, dim=-1)
        target_probs = all_probs[torch.arange(target.size(0), device=target.device), target]
        losses = self.weight * (1.0 - target_probs)

        return losses


class SymmetricCrossEntropy:
    def __init__(self, alpha=1.0, beta=1.0):
        self.ce = CrossEntropy(weight=alpha)
        self.rce = ReverseCrossEntropy(weight=beta)

    def __call__(self, logits, target):
        return self.ce(logits, target) + self.rce(logits, target)


class ActivePassiveLoss:
    def __init__(self, alpha=100.0, beta=1.0):
        self.active_loss = NormalizedCrossEntropy(weight=alpha)
        self.passive_loss = ReverseCrossEntropy(weight=beta)

    def __call__(self, logits, target):
        return self.active_loss(logits, target) + self.passive_loss(logits, target)


def build_classification_loss(name):
    if name == 'ce':
        return CrossEntropy()
    elif name == 'nce':
        return NormalizedCrossEntropy()
    elif name == 'rce':
        return ReverseCrossEntropy()
    elif name == 'sl':
        return SymmetricCrossEntropy()
    elif name == 'apl':
        return ActivePassiveLoss()
    else:
        raise AttributeError('Unknown name of loss: {}'.format(name))
