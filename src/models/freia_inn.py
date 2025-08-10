import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=128, scale_factor_s=1.5, scale_factor_t=1.5):
        super().__init__()
        self.dim = dim // 2
        hidden_dim = 90

        self.sub_nn1 = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.dim * 2)
        )

        self.sub_nn2 = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.dim * 2)
        )

        nn.init.zeros_(self.sub_nn1[-1].weight)
        nn.init.zeros_(self.sub_nn1[-1].bias)
        nn.init.zeros_(self.sub_nn2[-1].weight)
        nn.init.zeros_(self.sub_nn2[-1].bias)

        self.scale_factor_s = scale_factor_s
        self.scale_factor_t = scale_factor_t

    def forward(self, x, reverse=False):
        x1, x2 = x[:, :self.dim], x[:, self.dim:]

        if not reverse:
            # Step 1: x2 → s2, t2 → transform x1 → y1
            log_s2, t2 = self.sub_nn2(x2).chunk(2, dim=1)
            log_s2 = torch.tanh(log_s2) * self.scale_factor_s
            t2 = torch.tanh(t2) * self.scale_factor_t
            y1 = x1 * torch.exp(log_s2) + t2

            # Step 2: y1 → s1, t1 → transform x2 → y2
            log_s1, t1 = self.sub_nn1(y1).chunk(2, dim=1)
            log_s1 = torch.tanh(log_s1) * self.scale_factor_s
            t1 = torch.tanh(t1) * self.scale_factor_t
            y2 = x2 * torch.exp(log_s1) + t1

            log_det_jacobian = (log_s1 + log_s2).sum(dim=1)
        else:
            y1, y2 = x[:, :self.dim], x[:, self.dim:]

            # Step 1: y1 → s1, t1 → recover x2
            log_s1, t1 = self.sub_nn1(y1).chunk(2, dim=1)
            log_s1 = torch.tanh(log_s1) * self.scale_factor_s
            t1 = torch.tanh(t1) * self.scale_factor_t
            x2 = (y2 - t1) * torch.exp(-log_s1)

            # Step 2: x2 → s2, t2 → recover x1
            log_s2, t2 = self.sub_nn2(x2).chunk(2, dim=1)
            log_s2 = torch.tanh(log_s2) * self.scale_factor_s
            t2 = torch.tanh(t2) * self.scale_factor_t
            x1 = (y1 - t2) * torch.exp(-log_s2)

            log_det_jacobian = -(log_s1 + log_s2).sum(dim=1)

        y = torch.cat([y1, y2], dim=1) if not reverse else torch.cat([x1, x2], dim=1)
        logs = {
            'log_s1': torch.max(torch.abs(log_s1)).item(),
            'log_s2': torch.max(torch.abs(log_s2)).item(),
            't1': torch.max(torch.abs(t1)).item(),
            't2': torch.max(torch.abs(t2)).item(),
        }
        return y, log_det_jacobian, logs



class PermutationLayer(nn.Module):
    def __init__(self, dim, seed=1):
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.register_buffer("perm", torch.randperm(dim, generator=rng))
        self.register_buffer("inv_perm", torch.argsort(self.perm))

    def forward(self, x, reverse=False):
        if reverse:
            return x[:, self.inv_perm]
        else:
            return x[:, self.perm]


class Freia_INN(nn.Module):
    def __init__(self, dim, hidden_dim=128, num_blocks=6, prior=None):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            mask_type = 'even' if i % 2 == 0 else 'odd'
            self.blocks.append(AffineCoupling(dim, hidden_dim))
            self.blocks.append(PermutationLayer(dim, seed=i))  # Random but fixed permutation

        self.prior = prior if prior is not None else torch.distributions.Normal(0, 1)

    def f(self, x):
        log_det = torch.zeros(x.size(0), device=x.device)
        logs_s_acc = 0
        t_acc = 0

        z = x
        for block in self.blocks:
            if isinstance(block, AffineCoupling):
                z, ld, logs_block = block(z)
                log_det += ld
                logs_s_acc += logs_block['log_s1'] + logs_block['log_s2']
                t_acc += logs_block['t1'] + logs_block['t2']
            else:
                z = block(z)  # permutation

        logs = {
            'log_s': logs_s_acc,
            't': t_acc,
        }
        return z, log_det, logs

    def g(self, z):
        log_det = torch.zeros(z.size(0), device=z.device)
        x = z
        for block in reversed(self.blocks):
            if isinstance(block, AffineCoupling):
                x, ld, _ = block(x, reverse=True)
                log_det += ld
            else:
                x = block(x, reverse=True)  # reverse permutation
        return x, log_det

    def log_prob(self, z, log_diag_J):
        # Element-wise log prob summed over dims, batch-wise
        log_prior_prob = self.prior.log_prob(z).sum(dim=1)  # shape: (B,)
        return log_prior_prob + log_diag_J  # shape: (B,)

    def forward(self, x):
        z, log_det, logs = self.f(x)
        return z, log_det, logs