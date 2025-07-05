import torch

log_s = torch.tensor(2.0, dtype=torch.float32)
s = torch.exp(log_s)
inv_s = torch.exp(-log_s)
product = s * inv_s

print("s * inv_s (raw):", product.item())
print("Is exactly 1.0?", product.item() == 1.0)
print("Difference from 1.0:", abs(product.item() - 1.0))
