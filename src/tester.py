import torch
import numpy as np
import matplotlib.pyplot as plt
from models.ardizzone import Model
from utils.dataloader import FKIKDataset
from torch.utils.data import DataLoader

def test_inverse(model: Model):

    x = torch.rand((64, 4))
    pred = model(x)
    
    x_rec = model.inverse(pred)

    if torch.allclose(x, x_rec, atol=1e-5):
        print("Reconstruction successful: x and x_rec are close enough")
        print(f"Difference: {torch.norm(x-x_rec, p=2).item()}")
    else:
        print("Reconstruction unsuccessful: x and x_rec are not close enough")
        print(f"Difference: {torch.norm(x-x_rec, p=2).item()}")
    
def test_model(model: Model, test_dataset, device):

    """
    Test the trained model on test dataset.
    - Evaluate y (ee_pose) prediction error
    - Check if z follows normal distribution
    - Create 2 plots: y comparison and z distribution
    """


    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_y_true = []
    all_y_pred = []
    all_z_pred = []
    total_mse = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for test_batch in test_loader:
            x = test_batch["joint_config"].to(device)
            y_true = test_batch["ee_pose"].to(device)
            
            # Forward pass through model
            pred = model(x)
            
            # Split output into y and z components
            y_pred = pred[:, :2]  # First 2 dimensions (ee_pose prediction)
            z_pred = pred[:, 2:]  # Remaining dimensions (latent variables)
            
            # Calculate MSE for y (ee_pose) - sum over batch, not average
            batch_mse_sum = torch.nn.functional.mse_loss(y_pred, y_true, reduction='sum')
            total_mse += batch_mse_sum.item()
            num_samples += y_true.shape[0]  # Add number of samples in this batch
            
            # Store for plotting
            all_y_true.append(y_true.cpu())
            all_y_pred.append(y_pred.cpu())
            all_z_pred.append(z_pred.cpu())
    
    # Convert to numpy for plotting
    y_true_all = torch.cat(all_y_true, dim=0).numpy()
    y_pred_all = torch.cat(all_y_pred, dim=0).numpy()
    z_pred_all = torch.cat(all_z_pred, dim=0).numpy()
    
    # Calculate metrics
    avg_mse = total_mse / num_samples  # Average MSE per sample
    print(f"Average MSE per sample for y (ee_pose): {avg_mse:.6f}")
    print(f"Total samples evaluated: {num_samples}")
    
    # Test normality of z
    test_normality(z_pred_all)
    
    # Create plots
    plot_y_comparison(y_true_all, y_pred_all)
    plot_z_distribution(z_pred_all)
    
    return avg_mse, z_pred_all

def test_normality(z_values):
    """
    Test if z values follow a standard normal distribution N(0,1).
    Z should already be standard normal if the model is working correctly.
    """
    print("\n" + "="*50)
    print("STANDARD NORMALITY TEST FOR Z (LATENT VARIABLES)")
    print("Expected: Mean ≈ 0, Std ≈ 1, N(0,1) distribution")
    print("="*50)
    
    # Test each dimension of z
    for dim in range(z_values.shape[1]):
        z_dim = z_values[:, dim]
        
        # Calculate statistics (should be close to 0 and 1)
        mean = np.mean(z_dim)
        std = np.std(z_dim, ddof=1)  # Sample standard deviation
        
        # Test if z_dim follows standard normal (no normalization!)
        within_1_std = np.sum(np.abs(z_dim) <= 1.0) / len(z_dim)
        within_2_std = np.sum(np.abs(z_dim) <= 2.0) / len(z_dim)
        within_3_std = np.sum(np.abs(z_dim) <= 3.0) / len(z_dim)
        
        print(f"Dimension {dim}:")
        print(f"  Mean: {mean:.4f} (should be ≈ 0)")
        print(f"  Std:  {std:.4f} (should be ≈ 1)")
        print(f"  Within 1σ: {within_1_std:.3f} (expected: 0.683)")
        print(f"  Within 2σ: {within_2_std:.3f} (expected: 0.954)")
        print(f"  Within 3σ: {within_3_std:.3f} (expected: 0.997)")
        
        # Check if reasonably standard normal
        mean_ok = abs(mean) < 0.2  # Mean should be close to 0
        std_ok = 0.8 <= std <= 1.2  # Std should be close to 1
        dist_ok = (0.6 <= within_1_std <= 0.75) and (within_2_std >= 0.9)
        
        is_standard_normal = mean_ok and std_ok and dist_ok
        
        print(f"  Mean ≈ 0: {'✓' if mean_ok else '✗'}")
        print(f"  Std ≈ 1:  {'✓' if std_ok else '✗'}")
        print(f"  Distribution shape: {'✓' if dist_ok else '✗'}")
        print(f"  Standard Normal N(0,1): {'✓' if is_standard_normal else '✗'}")
        print()
    
    return is_standard_normal

def plot_y_comparison(y_true, y_pred):
    """
    Plot 2D comparison between true and predicted end-effector positions.
    Shows both points in workspace with lines connecting corresponding pairs.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot true positions (blue) and predicted positions (red)
    ax.scatter(y_true[:, 0], y_true[:, 1], color='blue', alpha=0.7, s=30, 
               label='True EE Position', edgecolors='darkblue', linewidth=0.5)
    ax.scatter(y_pred[:, 0], y_pred[:, 1], color='red', alpha=0.7, s=30, 
               label='Predicted EE Position', edgecolors='darkred', linewidth=0.5)
    
    # Draw lines connecting corresponding true and predicted points
    for i in range(len(y_true)):
        ax.plot([y_true[i, 0], y_pred[i, 0]], [y_true[i, 1], y_pred[i, 1]], 
                'k-', alpha=0.3, linewidth=0.5)
    
    # Calculate relative distances between corresponding points
    # Absolute Euclidean distance between each corresponding pair
    absolute_distances = np.sqrt((y_true[:, 0] - y_pred[:, 0])**2 + (y_true[:, 1] - y_pred[:, 1])**2)
    
    # Relative distance: normalize by the magnitude of true position
    true_magnitudes = np.sqrt(y_true[:, 0]**2 + y_true[:, 1]**2)
    
    # Only avoid division by zero for true points exactly at origin (very rare)
    # If true_magnitude ≈ 0, then absolute_distance should also be ≈ 0 if model is good
    relative_distances = np.where(true_magnitudes > 1e-10, 
                                  absolute_distances / true_magnitudes, 
                                  0.0)  # Set to 0 for points exactly at origin
    
    # Calculate average and max relative distance
    avg_relative_distance = np.mean(relative_distances)
    max_relative_distance = np.max(relative_distances)
    avg_absolute_distance = np.mean(absolute_distances)
    max_absolute_distance = np.max(absolute_distances)
    
    print(f"Average relative distance between corresponding points: {avg_relative_distance:.6f}")
    print(f"Maximum relative distance between corresponding points: {max_relative_distance:.6f}")
    print(f"Average absolute distance between corresponding points: {avg_absolute_distance:.6f}")
    print(f"Maximum absolute distance between corresponding points: {max_absolute_distance:.6f}")
    
    # Set labels and title
    ax.set_xlabel('End-Effector X Position')
    ax.set_ylabel('End-Effector Y Position')
    ax.set_title('End-Effector Position: True vs Predicted')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add statistics text box
    stats_text = f'Avg Rel Dist: {avg_relative_distance:.4f}\n'
    stats_text += f'Max Rel Dist: {max_relative_distance:.4f}\n'
    stats_text += f'Avg Abs Dist: {avg_absolute_distance:.4f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
            verticalalignment='top', fontsize=10)
    
    # Make axes equal for proper spatial representation
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()

def plot_z_distribution(z_values):
    """
    Plot distribution of z values (latent variables).
    """
    num_dims = z_values.shape[1]
    fig, axes = plt.subplots(1, num_dims, figsize=(5*num_dims, 4))
    
    if num_dims == 1:
        axes = [axes]
    
    for dim in range(num_dims):
        z_dim = z_values[:, dim]
        
        # Histogram
        axes[dim].hist(z_dim, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Overlay normal distribution
        mean, std = np.mean(z_dim), np.std(z_dim)
        x_range = np.linspace(z_dim.min(), z_dim.max(), 100)
        normal_curve = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std)**2)
        axes[dim].plot(x_range, normal_curve, 'r-', lw=2, label=f'Normal(μ={mean:.3f}, σ={std:.3f})')
        
        axes[dim].set_xlabel(f'Z[{dim}] values')
        axes[dim].set_ylabel('Density')
        axes[dim].set_title(f'Distribution of Z[{dim}]')
        axes[dim].legend()
        axes[dim].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Z (Latent Variables) Distribution', y=1.02)
    plt.show()


if __name__=="__main__":
    dataset = FKIKDataset("fk_ik_dataset.pt")
    mask = torch.zeros(dataset[0]['joint_config'].shape[0])
    mask[:len(mask)//2] = 1
    mask = mask.bool()

    model = Model(nr_blocks=2, masks=[mask, mask])

    test_inverse(model)
    
    test_dataset = FKIKDataset('fk_ik_dataset_test.pt')
    test_model(model, test_dataset, 'cpu')

    

