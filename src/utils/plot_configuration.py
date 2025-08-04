import torch
import matplotlib.pyplot as plt


def plot_configuration(x):
    if x.ndim == 1:
        x = x.reshape(1, x.shape[0])  # Fix: assign the result back to x
    if x.ndim == 3:
        raise ArithmeticError("x can't have so many dimensions")  # Fix: actually raise the error
    
    # Validate that we have at least 4 columns for the 4 joint angles
    if x.shape[1] < 4:
        raise ValueError(f"Expected at least 4 joint angles, got {x.shape[1]}")
    
    x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    l1, l2, l3 = 0.5, 0.5, 1.0

    y1 = x1 + l1 * torch.sin(x2) + l2 * torch.sin(x3 - x2) + l3 * torch.sin(x4 - x2 - x3) # vertical component
    y2 = l1 * torch.cos(x2) + l2 * torch.cos(x3 - x2) + l3 * torch.cos(x4 - x2 - x3) # horizontal component
    y = torch.stack([y1, y2], dim=1)  # y: (N, 2)

    j1_x = torch.zeros(x.shape[0])
    j1_y = x1

    j2_x = l1 * torch.cos(x2)
    j2_y = x1 + l1 * torch.sin(x2)

    j3_x = j2_x + l2 * torch.cos(x3 - x2)
    j3_y = j2_y + l2 * torch.sin(x3 - x2)

    j4_x = j3_x + l3 * torch.cos(x4 - x2 - x3)
    j4_y = j3_y + l3 * torch.sin(x4 - x2 - x3)  # Fix: should be j3_y, not j4_x

    # Convert to numpy arrays safely
    def safe_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach().numpy()
        else:
            return tensor.numpy()

    j1 = torch.stack([j1_y, j1_x], dim=1)
    j2 = torch.stack([j2_y, j2_x], dim=1)
    j3 = torch.stack([j3_y, j3_x], dim=1)
    j4 = torch.stack([j4_y, j4_x], dim=1)
    
    j1_np = safe_numpy(j1)
    j2_np = safe_numpy(j2)
    j3_np = safe_numpy(j3)
    j4_np = safe_numpy(j4)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot points and connect them with lines
    for i in range(j1_np.shape[0]):
        plt.plot([j1_np[i, 1], j2_np[i, 1], j3_np[i, 1], j4_np[i, 1]],
                 [j1_np[i, 0], j2_np[i, 0], j3_np[i, 0], j4_np[i, 0]],
                 'o-', linewidth=2, markersize=6,
                 color='deepskyblue', markerfacecolor='red', markeredgecolor='black')
    
    # Customize the plot
    plt.title('Arm Configuration Plot', fontsize=16, fontweight='bold')
    plt.xlabel('X Position', fontsize=12)
    plt.ylabel('Y Position', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # Equal aspect ratio
    
    # Mark the first point (base) with green and the last point (end effector) with red
    for i in range(j1_np.shape[0]):
        plt.scatter(j1_np[i, 1], j1_np[i, 0], color='green', s=80, zorder=3, label='Base' if i == 0 else "")
        plt.scatter(j4_np[i, 1], j4_np[i, 0], color='red', s=80, zorder=3, label='End Effector' if i == 0 else "")
    # Optionally add legend only once
    if j1_np.shape[0] > 0:
        plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    return y

# Example usage
if __name__ == "__main__":
    # Create some sample data
    n_samples = 10
    x_sample = torch.randn(n_samples, 4)  # Random joint configurations
    
    # Plot the configuration
    result = plot_configuration(x_sample)
    print(f"Plotted {n_samples} configurations")
    print(f"Result shape: {result.shape}")
