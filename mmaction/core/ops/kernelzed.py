import torch
import torch.nn as nn
from torch.nn import Parameter

from .math import normalize


def kernel_prod(norm_x, norm_y, alpha, use_end_components=True, eps=1e-6):
    scores = norm_x.mm(norm_y)

    num_main_components = alpha.size(0) - (2 if use_end_components else 0)
    main_components = [torch.pow(scores.unsqueeze(1), i) for i in range(num_main_components)]

    if use_end_components:
        with torch.no_grad():
            high_mask = (scores > 1.0 - eps).float()
            low_mask = (scores < -1.0 + eps).float()

        odd_component = high_mask - low_mask
        even_component = high_mask + low_mask

        all_components = main_components + [odd_component.unsqueeze(1), even_component.unsqueeze(1)]
    else:
        all_components = main_components

    all_components = torch.cat(all_components, dim=1)
    out = torch.sum(alpha.view(1, -1, 1) * all_components, dim=1)

    return out


class KernelizedClassifier(nn.Module):
    def __init__(self, features_dim, num_classes, num_centers=1, num_components=5, eps=1e-6):
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
        self.alpha.data.fill_(1.0)

    def forward(self, normalized_x):
        normalized_x = normalized_x.view(-1, self.features_dim)
        normalized_weights = normalize(self.weight, dim=0)
        normalized_alpha = torch.sigmoid(self.alpha)
        # print(self.alpha.detach().cpu().numpy(), normalized_alpha.detach().cpu().numpy())
        # exit()

        scores = kernel_prod(normalized_x, normalized_weights, normalized_alpha, eps=self.eps)

        return scores
