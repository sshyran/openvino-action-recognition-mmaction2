import torch
import torch.nn as nn
import torch.nn.functional as F


class Dropout(nn.Module):
    DISTRIBUTIONS = ['bernoulli', 'gaussian', 'infodrop']

    def __init__(self, p=0.5, mu=0.5, sigma=0.2, dist='bernoulli', kernel=3, temperature=0.05):
        super(Dropout, self).__init__()

        self.dist = dist
        assert self.dist in Dropout.DISTRIBUTIONS

        self.p = float(p)
        assert 0. <= self.p <= 1.

        self.mu = float(mu)
        self.sigma = float(sigma)
        assert self.sigma > 0.

        self.kernel = kernel
        assert self.kernel >= 3
        self.temperature = temperature
        assert self.temperature > 0.0

    def forward(self, x, x_original=None):
        if not self.training:
            return x

        if self.dist == 'bernoulli':
            out = F.dropout(x, self.p, self.training)
        elif self.dist == 'gaussian':
            with torch.no_grad():
                soft_mask = x.new_empty(x.size()).normal_(self.mu, self.sigma).clamp_(0., 1.)

            scale = 1. / self.mu
            out = scale * soft_mask * x
        elif self.dist == 'infodrop':
            assert x_original is not None

            out = info_dropout(x_original, self.kernel, x, self.p, self.temperature)
        else:
            out = x

        return out


def info_dropout(in_features, kernel, out_features, drop_rate, temperature=0.05, eps=1e-12):
    assert isinstance(kernel, int)
    assert kernel % 2 == 1

    in_shape = in_features.size()
    assert len(in_shape) in (4, 5)

    with torch.no_grad():
        if len(in_shape) == 5:
            b, c, t, h, w = in_shape
            out_mask_shape = b, 1, t, h, w

            in_features = in_features.permute(0, 2, 1, 3, 4)
            b *= t
        else:
            b, c, h, w = in_shape
            out_mask_shape = b, 1, h, w
        in_features = in_features.reshape(-1, c, h, w)

        padding = (kernel - 1) // 2
        unfolded_features = F.unfold(in_features, kernel, padding=padding)
        unfolded_features = unfolded_features.view(b, c, kernel * kernel, -1)

        distances = ((unfolded_features - in_features.view(-1, c, 1, h * w)) ** 2).sum(dim=1)
        weights = (0.5 * distances / distances.mean(dim=(1, 2), keepdim=True).clamp_min(eps)).neg().exp()

        middle = kernel * kernel // 2
        log_info = (weights[:, :middle].sum(dim=1) + weights[:, (middle + 1):].sum(dim=1) + eps).log()

        prob_weights = (1. / float(temperature) * log_info).exp() + eps
        probs = prob_weights / prob_weights.sum(dim=-1, keepdim=True)

        drop_num_samples = max(1, int(drop_rate * float(probs.size(-1))))
        drop_indices = torch.multinomial(probs, num_samples=drop_num_samples, replacement=True)

        out_mask = torch.ones_like(probs)
        out_mask[torch.arange(out_mask.size(0), device=out_mask.device).view(-1, 1), drop_indices] = 0.0

    out_scale = 1.0 / (1.0 - drop_rate)
    out = out_scale * out_features * out_mask.view(out_mask_shape)

    return out
