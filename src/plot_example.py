import torch
from utils.dataloader import FKIKDataset
from models.realnvp import RealNVP  # <-- replace with actual import path
import torch.nn.functional as F

# 1. Load the dataset and take one sample
dataset = FKIKDataset('fk_ik_dataset.pt')
sample = dataset[0]

train_direction = 'ik'  # or 'fk'

input_key = 'ee_pose' if train_direction == 'ik' else 'joint_config'
output_key = 'joint_config' if train_direction == 'ik' else 'ee_pose'

x_vec = sample[input_key].unsqueeze(0)  # (1, D)
y_vec = sample[output_key].unsqueeze(0)  # (1, D)

print(f"Input vector ({input_key}):", x_vec)
print(f"Target output vector ({output_key}):", y_vec)

# 2. Pad and reshape to 2×2
def pad_and_reshape(vec, target_dim=64):
    padded = F.pad(vec, (0, target_dim - vec.shape[1]))
    return padded.view(-1, 1, 8, 8)  # (B, C=1, H=8, W=8)

x_tensor = pad_and_reshape(x_vec)  # (1, 1, 2, 2)

# 3. Dummy prior and datainfo for RealNVP setup
class DummyPrior:
    def log_prob(self, z): return torch.zeros_like(z)
    def sample(self, shape): return torch.randn(shape)

class DummyDataInfo:
    def __init__(self):
        self.channel = 1
        self.size = 8  # instead of 2
        self.name = 'custom'

class DummyHParams:
    def __init__(self):
        self.res_blocks = 2
        self.bottleneck = False
        self.skip = False
        self.weight_norm = False
        self.coupling_bn = False
        self.affine = True
        self.base_dim = 64  # used internally

prior = DummyPrior()
datainfo = DummyDataInfo()
hps = DummyHParams()

# 4. Load RealNVP and run forward/inverse
device = 'cuda' if torch.cuda.is_available() else 'cpu'
realnvp = RealNVP(datainfo, prior, hps).to(device)
realnvp.eval()  # no batchnorm updates

x_tensor = x_tensor.to(device)

with torch.no_grad():
    if train_direction == 'ik':
        z_tensor, _ = realnvp.f(x_tensor)  # (1, 1, 2, 2)
        z_pred = z_tensor.view(1, -1)[:, :y_vec.shape[1]]
    else:
        y_tensor = realnvp.g(x_tensor)
        z_pred = y_tensor.view(1, -1)[:, :y_vec.shape[1]]

y_vec = y_vec.to(z_pred.device)  # <-- FIX

print(f"Predicted {output_key}:", z_pred)
print(f"Ground truth {output_key}:", y_vec)
print(f"L2 error: {torch.norm(z_pred - y_vec):.6f}")


import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# ===============================
# Invertibility Sanity Check: q0 → z → q1
# ===============================
q0 = sample['joint_config'].unsqueeze(0).to(device)  # (1, 4)
q0_tensor = pad_and_reshape(q0)                      # (1, 1, 8, 8)

# f(q0) → z
with torch.no_grad():
    z_tensor, _ = realnvp.f(q0_tensor)

# g(z) → q1
with torch.no_grad():
    q1_tensor = realnvp.g(z_tensor)
    q1 = q1_tensor.view(1, -1)[:, :4]  # crop padded

# ===============================
# Plot both joint configurations
# ===============================
def plot_arm(q, label, color, alpha=1.0, linestyle='-'):
    q = q.squeeze().cpu().numpy()
    x1 = q[0]
    theta1 = q[1]
    theta2 = q[2]
    theta3 = q[3]
    l1, l2, l3 = 0.5, 0.5, 1.0

    # Cumulative angles
    theta_12 = theta2
    theta_23 = theta3 - theta2
    theta_34 = theta3 + theta3 - theta2 - theta2

    # Positions
    x0 = np.array([x1, 0])
    x1 = x0 + np.array([l1 * np.sin(theta_12), l1 * np.cos(theta_12)])
    x2 = x1 + np.array([l2 * np.sin(theta_23), l2 * np.cos(theta_23)])
    x3 = x2 + np.array([l3 * np.sin(theta_34), l3 * np.cos(theta_34)])

    xs = [x0[0], x1[0], x2[0], x3[0]]
    ys = [x0[1], x1[1], x2[1], x3[1]]

    plt.plot(xs, ys, '-o', label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=4)
    plt.scatter([x0[0]], [x0[1]], marker='s', color=color, label=f"{label} Base", s=60)
    plt.scatter([x3[0]], [x3[1]], marker='*', color=color, label=f"{label} EE", s=100)

    return np.array([x0, x1, x2, x3])  # (4, 2)

# Plot arms
plt.figure(figsize=(6, 6))
pos_q0 = plot_arm(q0, 'Original q₀', 'blue')
pos_q1 = plot_arm(q1, 'Reconstructed q₁', 'red', alpha=0.75, linestyle='--')

plt.legend()
plt.axis("equal")
plt.grid(True)
plt.title("Invertibility Check: g(f(q₀)) ≈ q₀")
plt.savefig("sanity_check_plot.png")
print("Plot saved as sanity_check_plot.png")

# ===============================
# Distance Metrics
# ===============================
distances = np.linalg.norm(pos_q0 - pos_q1, axis=1)
print("Euclidean distances between corresponding joints:")
for i, d in enumerate(distances):
    print(f"  Joint {i}: {d:.6f}")
print(f"End-effector position error: {distances[-1]:.6f}")

# MSE error
mse = F.mse_loss(q1, q0)
print(f"MSE error between q₀ and g(f(q₀)): {mse.item():.6f}")
