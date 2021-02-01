import torch
import torch.nn as nn
from torch.nn import Parameter

from .math import normalize


def kernel_prod(norm_x, norm_y, alpha, eps=1e-6):
    scores = norm_x.mm(norm_y)

    num_components = alpha.size(0)
    components = [alpha[i] * torch.pow(scores.unsqueeze(1), i)
                  for i in range(1, num_components - 2)]

    with torch.no_grad():
        high_mask = (scores > 1.0 - eps).float()
        low_mask = (scores < -1.0 + eps).float()

    odd_component = alpha[-2] * (high_mask - low_mask)
    even_component = alpha[-1] * (high_mask + low_mask)

    main_components = components + [odd_component.unsqueeze(1), even_component.unsqueeze(1)]
    main_component = torch.cat(main_components, dim=1).sum(dim=1)

    return alpha[0] + main_component


class KernelizedClassifier(nn.Module):
    def __init__(self, features_dim, num_classes, num_centers=1, num_components=8, eps=1e-6):
        super(KernelizedClassifier, self).__init__()

        assert num_centers == 1
        assert num_components >= 2

        assert features_dim > 0
        self.features_dim = features_dim
        assert num_classes >= 2
        self.num_classes = num_classes
        assert eps > 0.0
        self.eps = eps

        self.weight = Parameter(torch.Tensor(features_dim, num_classes))
        self.alpha = Parameter(torch.Tensor(num_components + 2))

        self._init_weights()

    def _init_weights(self):
        self.weight.data.normal_().renorm_(2, 1, 1e-5).mul_(1e5)
        self.alpha.data.fill_(5.0)

    def forward(self, normalized_x):
        normalized_x = normalized_x.view(-1, self.features_dim)
        normalized_weights = normalize(self.weight, dim=0)
        normalized_alpha = torch.sigmoid(self.alpha)
        # print(self.alpha)
        # print(normalized_alpha)
        # exit()

        scores = kernel_prod(normalized_x, normalized_weights, normalized_alpha, self.eps)

        return scores
