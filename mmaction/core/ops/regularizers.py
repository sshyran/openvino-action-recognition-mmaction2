import torch
import torch.nn as nn


class NormRegularizer(nn.Module):
    def __init__(self, max_ratio=10.0, min_norm=0.5, weight=1.0, eps=1e-5):
        super(NormRegularizer, self).__init__()

        self.max_ratio = float(max_ratio)
        self.min_norm = float(min_norm)
        self.weight = float(weight)
        self.eps = float(eps)

    def forward(self, net):
        conv_layers = self._collect_conv_layers(net, self.eps)

        num_losses = 0
        accumulator = torch.tensor(0.0).cuda()
        for conv in conv_layers:
            loss = self._loss(conv['weight'], self.max_ratio, self.min_norm)
            if loss > 0:
                accumulator += loss.to(accumulator.device)
                num_losses += 1

        return self.weight * accumulator / float(max(1, num_losses))

    @staticmethod
    def _collect_conv_layers(net, eps):
        conv_layers = []
        for name, m in net.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                conv_layers.append(dict(
                    name=name,
                    weight=m.weight,
                    updated=False,
                ))
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                assert len(conv_layers) > 0

                last_conv = conv_layers[-1]
                assert not last_conv['updated']

                alpha = m.weight
                running_var = m.running_var.detach()

                scales = alpha / torch.sqrt(running_var + eps)
                scale_shape = [-1] + [1] * len(last_conv['weight'].size())
                scales = scales.view(*scale_shape)

                last_conv['weight'] = scales * last_conv['weight']
                last_conv['updated'] = True

        return conv_layers

    @staticmethod
    def _loss(weights_matrix, max_ratio, min_norm):
        num_filters = weights_matrix.size(0)
        if num_filters <= 1:
            return 0.0

        weights_matrix = weights_matrix.view(num_filters, -1)
        norms = torch.sqrt(torch.sum(weights_matrix ** 2, dim=-1))

        with torch.no_grad():
            norm_ratio = torch.max(norms) / torch.min(norms)
            median_norm = torch.median(norms)

        if norm_ratio < max_ratio and median_norm > min_norm:
            return 0.0

        trg_norm = max(min_norm, median_norm)
        losses = (norms - trg_norm) ** 2
        loss = losses.mean()

        return loss
