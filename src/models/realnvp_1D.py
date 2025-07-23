import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=128, mask_type='even'):
        super().__init__()
        self.dim = dim
        self.mask = self.create_mask(dim, mask_type)

        self.nn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, dim * 2),  # Outputs [log_s, t]
        )

        nn.init.zeros_(self.nn[-1].weight)
        nn.init.zeros_(self.nn[-1].bias)

        # Last linear layer: n.Linear(hidden_dim, dim * 2) initialization of this layer's weights and biases to zero
        # s = 0 and t = 1 -> then z is equal to x

    def create_mask(self, dim, mask_type):
        mask = torch.zeros(dim)
        if mask_type == 'even':
            mask[::2] = 1
        else:
            mask[1::2] = 1
        return mask

    def forward(self, x, reverse=False):
        mask = self.mask.to(x.device)

        x_masked = x * mask
        st = self.nn(x_masked)
        log_s, t = st.chunk(2, dim=1)

        scale_factor = 2.0
        scale_factor_t = 2.0
        log_s = torch.tanh(log_s) * scale_factor
        t = torch.tanh(t) * scale_factor_t

        if reverse:
            # y = x * mask + (x - (1-mask) * t) * torch.exp(-log_s)  * (1 - mask)  
            y = x * mask + (1 - mask) * ((x - t) * torch.exp(-log_s))
            log_det_jacobian = -torch.sum((1 - mask) * log_s, dim=1)
        else:
            # y = x * (mask + (1 - mask) * torch.exp(log_s)) + (1 - mask) * t   
            y = x * mask + (1 - mask) * (x * torch.exp(log_s) + t)
            log_det_jacobian = torch.sum((1 - mask) * log_s, dim=1)

        logs = {
            'log_s': torch.max(torch.abs(log_s)),
            't': torch.max(torch.abs(t)),
        }
        return y, log_det_jacobian, logs


class RealNVP1D(nn.Module):
    def __init__(self, dim, hidden_dim=128, num_blocks=6, prior=None):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            mask_type = 'even' if i % 2 == 0 else 'odd'
            self.blocks.append(AffineCoupling(dim, hidden_dim, mask_type))

        self.prior = prior if prior is not None else torch.distributions.Normal(0, 1)

    def f(self, x):
        log_det = torch.zeros(x.size(0), device=x.device)
        logs_s_acc = 0
        t_acc = 0
        
        z = x
        for block in self.blocks:
            z, ld, logs_block = block(z)
            log_det += ld
            logs_s_acc += logs_block['log_s']
            t_acc += logs_block['t']
        logs = {
            'log_s': logs_s_acc,
            't': t_acc,
        }
        return z, log_det, logs

    def g(self, z):
        log_det = torch.zeros(z.size(0), device=z.device)
        x = z
        for block in reversed(self.blocks):
            x, ld, _ = block(x, reverse=True)
            log_det += ld
        return x, log_det

    def log_prob(self, z, log_diag_J):
        log_det_J = torch.sum(log_diag_J)
        log_prior_prob = torch.sum(self.prior.log_prob(z))
        return log_prior_prob + log_det_J

    def forward(self, x):
        z, log_det, logs = self.f(x)
        return z, log_det, logs
    
if __name__ == "__main__":
    import torch

    # Set seeds for reproducibility (optional)
    # torch.manual_seed(42)

    # Create model with default random initialization
    model = RealNVP1D(dim=4, hidden_dim=128, num_blocks=6)
    model.eval()  # Make sure dropout/batchnorm (if any) are off

    # Create input
    x = torch.randn(128, 4)

    # Forward to latent space
    z, _ = model.f(x)

    # Inverse back to x
    x_recon, _ = model.g(z)

    # Check reconstruction error
    l2_error = torch.norm(x - x_recon, p=2).item()
    max_error = torch.max(torch.abs(x - x_recon)).item()

    print("L2 Reconstruction Error:", l2_error)
    print("Max Absolute Error per Element:", max_error)

    if torch.allclose(x, x_recon, atol=1e-5):
        print("Reconstruction successful: x and x_recon are close enough.")
        print(f"difference: {torch.norm(x - x_recon, p=2).item()}")
    else:
        print("Reconstruction failed: x and x_recon are not close enough.")
        print(f"difference: {torch.norm(x - x_recon, p=2).item()}")
        raise ValueError("Reconstruction error exceeds tolerance.")