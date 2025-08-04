import torch.nn as nn
import numpy as np
import torch 
import torch.nn.functional as F

class BaseMapping(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return  self.block(x)
    
class ReversibleBlock(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask
        nr_ones = int(self.mask.sum())
        nr_zeros = int((~self.mask).sum())
        
        self.s2 = BaseMapping(nr_zeros, nr_ones)
        self.t2 = BaseMapping(nr_zeros, nr_ones)

        self.s1 = BaseMapping(nr_ones, nr_zeros)
        self.t1 = BaseMapping(nr_ones, nr_zeros)

    def forward(self, u):
        if u.shape[1] != len(self.mask):
            raise ValueError("Input and Mask don't have the same size")
        else:
            u1 = u[:, self.mask]
            u2 = u[:, ~self.mask]

            v1 = u1 * torch.exp(self.s2(u2)) + self.t2(u2)
            v2 = u2 * torch.exp(self.s1(v1)) + self.t1(v1)

            v = torch.zeros_like(u)
            v[:, self.mask] = v1
            v[:, ~self.mask] = v2
            return v
    
    def inverse(self, v):
        if v.shape[1] != len(self.mask):
            raise ValueError("Input and Mask don't have the same size")
        else:
            v1 = v[:, self.mask]
            v2 = v[:, ~self.mask]

            u2 = (v2 - self.t1(v1)) * torch.exp(-self.s1(v1))
            u1 = (v1 - self.t2(u2)) * torch.exp(-self.s2(u2))

            u = torch.zeros_like(v)
            u[:, self.mask] = u1
            u[:, ~self.mask] = u2
            return u

class Model(nn.Module):
    def __init__(self, nr_blocks, masks):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.masks = masks
        self.nr_blocks = nr_blocks
        for i in range(nr_blocks):
            self.blocks.append(ReversibleBlock(masks[i]))
    
    # The spliting in y, z should happen here.
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
    def inverse(self, v):
        for block in reversed(self.blocks):
            v = block.inverse(v)
        return v
        