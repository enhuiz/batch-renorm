import torch
from torch import nn, Tensor


class BN(nn.Module):
    def __init__(self, dim, momentum=0.1, eps=0):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones([dim]))
        self.bias = nn.Parameter(torch.zeros([dim]))
        self.running_mean: Tensor
        self.running_var: Tensor
        self.register_buffer("running_mean", torch.full([dim], 0.0))
        self.register_buffer("running_var", torch.full([dim], 1.0))

    @torch.no_grad()
    def _ema(self, a: Tensor, x: Tensor):
        return (1 - self.momentum) * a + self.momentum * x

    @torch.no_grad()
    def _update_stats(self, x: Tensor):
        dims = [i for i in range(x.dim()) if i > 0]
        mean = x.mean(dim=dims)
        # PyTorch uses unbiased var for running_var update
        var = x.var(dim=dims, unbiased=True)
        self.running_mean[:] = self._ema(self.running_mean, mean)
        self.running_var[:] = self._ema(self.running_var, var)

    @staticmethod
    def _get_batch_stats(x):
        dims = [i for i in range(x.dim()) if i > 0]
        mean = x.mean(dim=dims, keepdim=True)
        # However, for normalization, PyTorch uses biased var
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        return mean, var

    def _get_running_stats(self, x):
        mean, var = self.running_mean, self.running_var
        while mean.dim() < x.dim():
            mean = mean.unsqueeze(-1)
            var = var.unsqueeze(-1)
        return mean, var

    def _get_stats(self, x):
        if self.training:
            mean, var = self._get_batch_stats(x)
        else:
            mean, var = self._get_running_stats(x)
        return mean, var

    def _inv_sqrt(self, var):
        return 1 / (var + self.eps).sqrt()

    def _norm(self, x):
        mean, var = self._get_stats(x)
        return (x - mean) * self._inv_sqrt(var)

    def forward(self, x):
        x = x.transpose(0, 1)  # (b c ...) -> (c b ...)
        if self.training:
            self._update_stats(x)
        weight = self.weight
        bias = self.bias
        while weight.dim() < x.dim():
            weight = weight.unsqueeze(-1)
            bias = bias.unsqueeze(-1)
        x = weight * self._norm(x) + bias
        x = x.transpose(0, 1)  # (c b ...) -> (b c ...)
        return x


class BNRS(BN):
    """
    A batch norm but use running stats during training
    """

    def _get_stats(self, x):
        return self._get_running_stats(x)


class BRN(BN):
    """
    Batch renormalization: https://arxiv.org/pdf/1702.03275.pdf

    Not clipping for simplcity.
    """

    def _norm(self, x):
        if not self.training:
            return super()._norm(x)

        μb, σb2 = self._get_batch_stats(x)
        inv_σb = self._inv_sqrt(σb2)

        x = (x - μb) * inv_σb

        μ, σ2 = self._get_running_stats(x)
        inv_σ = self._inv_sqrt(σ2)

        with torch.no_grad():
            r = σb2.sqrt() * inv_σ
            d = (μb - μ) * inv_σ

        return r * x + d
