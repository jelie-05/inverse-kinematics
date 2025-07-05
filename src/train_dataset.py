import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import torch.distributions as distributions
from utils.dataloader import FKIKDataset
from torch.utils.data import DataLoader, random_split
from models.realnvp_1D import RealNVP1D

# add argparse for save_name
import argparse
parser = argparse.ArgumentParser(description='Train RealNVP model for FK/IK')
parser.add_argument('--save_name', type=str, required=True, help='Name to save the model and logs')


def compute_ee_pose(x, l):
    """
    x: (N, 4) tensor, where:
        x[:, 0] = base x translation
        x[:, 1], x[:, 2], x[:, 3] = theta1, theta2, theta3 (global angles)
    l: (3,) tensor of link lengths
    """
    x_trans = x[:, 0]
    theta1, theta2, theta3 = x[:, 1], x[:, 2], x[:, 3]
    l1, l2, l3 = l[0], l[1], l[2]

    y1 = l1 * torch.cos(theta1) + l2 * torch.cos(theta2) + l3 * torch.cos(theta3)
    y2 = x_trans + l1 * torch.sin(theta1) + l2 * torch.sin(theta2) + l3 * torch.sin(theta3)  # Allowing translation in y direction by x_trans

    y = torch.stack([y1, y2], dim=1)    # (x,y) end-effector position
    return y


def plot_arm(x, l, title="Robot Arm", color='blue', y_sample=None, x_gt=None):
    """
    x: joint configuration tensor of shape (1, 4)
    l: tensor of link lengths (3,)
    y_sample: target end-effector position to overlay as a blue dot, shape (1, 2) or (2,)
    """
    # x from sampling
    x = x.squeeze().cpu().numpy()
    l1, l2, l3 = l[0].item(), l[1].item(), l[2].item()
    
    base_x, base_y = 0, x[0]
    theta1, theta2, theta3 = x[1], x[2], x[3]

    joint1 = (base_x, base_y)
    joint2 = (base_x + l1 * np.cos(theta1), base_y + l1 * np.sin(theta1))
    joint3 = (joint2[0] + l2 * np.cos(theta2), joint2[1] + l2 * np.sin(theta2))
    ee     = (joint3[0] + l3 * np.cos(theta3), joint3[1] + l3 * np.sin(theta3))

    fig, ax = plt.subplots()

    # Plot arm
    xs = [joint1[0], joint2[0], joint3[0], ee[0]]
    ys = [joint1[1], joint2[1], joint3[1], ee[1]]
    ax.plot(xs, ys, 'o-', color=color, linewidth=3, label="Arm")

    # Plot base triangle
    base_triangle = plt.Polygon([
        (base_x - 0.1, base_y),
        (base_x + 0.1, base_y),
        (base_x, base_y - 0.15)
    ], closed=True, color='gray')
    ax.add_patch(base_triangle)

    # Mark EE
    ax.plot(ee[0], ee[1], 'ro', markersize=8, label="Predicted EE")

    # x from ground truth
    if x_gt is not None:
        x_gt = x_gt.squeeze().cpu().numpy()
        
        base_x_gt, base_y_gt = 0, x_gt[0]
        theta1_gt, theta2_gt, theta3_gt = x_gt[1], x_gt[2], x_gt[3]

        joint1_gt = (base_x_gt, base_y_gt)
        joint2_gt = (base_x_gt + l1 * np.cos(theta1_gt), base_y_gt + l1 * np.sin(theta1_gt))
        joint3_gt = (joint2_gt[0] + l2 * np.cos(theta2_gt), joint2_gt[1] + l2 * np.sin(theta2_gt))
        ee_gt     = (joint3_gt[0] + l3 * np.cos(theta3_gt), joint3_gt[1] + l3 * np.sin(theta3_gt))

        # Plot ground truth arm
        xs_gt = [joint1_gt[0], joint2_gt[0], joint3_gt[0], ee_gt[0]]
        ys_gt = [joint1_gt[1], joint2_gt[1], joint3_gt[1], ee_gt[1]]
        ax.plot(xs_gt, ys_gt, 'o--', color='orange', linewidth=3, label="GT Arm")

    # Plot y_sample if provided
    if y_sample is not None:
        y_sample = y_sample.squeeze().cpu().numpy()
        ax.plot(y_sample[0], y_sample[1], 'bo', markersize=8, label="Target EE")

    ax.set_title(title)
    ax.set_xlim(-1, 4)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()

    return fig


def plot_y_distribution(y_all, title="EE Position Distribution"):
    y_all = y_all.cpu().numpy()
    plt.figure(figsize=(20, 20))
    fig, ax = plt.subplots()
    ax.scatter(y_all[:, 0], y_all[:, 1], s=10, alpha=0.5, c='blue', label='EE positions')
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1, 4)
    ax.set_ylim(-2, 2)
    ax.grid(True)
    ax.set_aspect('equal')
    ax.legend()
    return fig


def train(model, train_loader, val_loader, train_direction='fk', device='cpu', n_epochs=10, lr=1e-3, writer=None):
    model = model.to(device)

    input_key = 'joint_config' if train_direction == 'fk' else 'ee_pose'
    output_key = 'ee_pose' if train_direction == 'fk' else 'joint_config'

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # Robot arm data:
    l = torch.tensor([0.5, 0.5, 1.0], device=device)  # Link lengths
    
    # Miscallaneous
    collected_y = []
    max_y_samples = 15000
    noise_scale = 0.001  # Scale for noise added to sampling of latent space
    mle_scale = 0.01
    count_avg = 0

    for epoch in range(n_epochs):
        total_train_loss = 0.0
        total_train_recon = 0.0
        total_train_pred = 0.0
        total_train_mle = 0.0
        total_val_loss = 0.0
        total_val_recon = 0.0
        total_val_pred = 0.0
        total_val_mle = 0.0
        
        for batch in train_loader:
            model.train()
            x = batch[input_key].to(device)
            y = batch[output_key].to(device)

            # Forward pass through the model
            z, log_det = model(x)
            z_ee, z_latent = z[:, :y.shape[1]], z[:, y.shape[1]:]   # Split z into ee and latent parts

            # Sanity check
            z_sanity_check = torch.cat((z_ee.detach(), z_latent.detach()), dim=1)  # Concatenate y and z_latent
            x_sanity_check, _ = model.g(z_sanity_check)  # Reconstruct x from z_ee and z_latent
            # assert torch.allclose(x_sanity_check, x, atol=1e-5), "Sanity check failed: Reconstructed x does not match original x"

            # Compute losses
            # 1. Reconstruction loss
            latent_samples = 'noise'  # 'normal' or 'noise'
            if latent_samples == 'normal':
                # Option 1: z_samples comes is sampled from normal distribution
                z_samples = torch.randn_like(z_latent)  # Sample from standard normal distribution
            elif latent_samples == 'noise':
                # Option 2: z_samples is a minor noise added to z_latent -> the lantent space should be normal too <-> from MLE/MMD losses
                z_samples = z_latent + noise_scale * torch.randn_like(z_latent)  # Add minor noise to z_latent
            z_recon = torch.cat((y, z_samples), dim=1)  # Reconstruct z from z_ee and z_latent
            x_recon, _ = model.g(z_recon) 
            loss_recon = loss_fn(x_recon, x)

            # 2. Prediction loss:
            # Loss between the GT y and the predicted end-effector position z_ee
            loss_pred = loss_fn(z_ee, y)

            # 3. MLE loss:
            # Loss between the latent z_latent and a standard normal distribution
            # log_p_z = -0.5 * torch.sum(z_latent ** 2, dim=1)
            # loss_mle = -torch.mean(log_p_z + log_det)
            loss_mle = -torch.mean(-0.5 * torch.sum(z_latent**2, dim=1) + log_det)  

            # 4. MMD loss


            loss = (loss_recon + loss_pred) + loss_mle * mle_scale

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()
            total_train_recon += loss_recon.item()
            total_train_pred += loss_pred.item()
            total_train_mle += loss_mle.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_recon = total_train_recon / len(train_loader)
        avg_train_pred = total_train_pred / len(train_loader)
        avg_train_mle = total_train_mle / len(train_loader)


        # === Save y samples for visualization ===
        if len(collected_y) < max_y_samples:
            collected_y.append(y.detach().cpu())

        # === Evaluation ===
        model.eval()

        # === Visualization ===
        if epoch == 0:
            x_gt = x[10].detach().unsqueeze(0)  # Save a sample for visualization
            y_eval = y[10].detach().unsqueeze(0) # save as a sample for visualization
            z_sample_dim = z_latent.shape[1]  # Dimension of the latent space
        elif epoch % 2 == 0 and epoch != 0:
            # Prepare input to inverse pass
            z_latent_sample = torch.randn((1, z_sample_dim), device=device)  # Random latent vector
            z_input = torch.cat([y_eval, z_latent_sample], dim=1)  # Concatenate target with latent

            # Inverse pass to get joint config
            with torch.no_grad():
                x_pred, _ = model.g(z_input)  # Shape (1, 4)

            # Plot the robot arm from x_pred 
            fig = plot_arm(x_pred, l, title="Eval Arm at Epoch {}".format(epoch), color='green', y_sample=y_eval, x_gt=x_gt)

            # Log to TensorBoard
            writer.add_figure("EvalArm/y_eval", fig, global_step=epoch)
            plt.close(fig)


        # === Validation Loop ===
        with torch.no_grad():
            for batch in val_loader:
                x = batch[input_key].to(device)
                y = batch[output_key].to(device)

                z, log_det = model(x)

                # Split z into ee and latent parts
                z_ee, z_latent = z[:, :y.shape[1]], z[:, y.shape[1]:]

                # Compute losses
                # 1. Reconstruction loss
                latent_samples = 'noise'  # 'normal' or 'noise'
                if latent_samples == 'normal':
                    # Option 1: z_samples comes is sampled from normal distribution
                    z_samples = torch.randn_like(z_latent)  # Sample from standard normal distribution
                elif latent_samples == 'noise':
                    # Option 2: z_samples is a minor noise added to z_latent -> the lantent space should be normal too <-> from MLE/MMD losses
                    z_samples = z_latent + noise_scale * torch.randn_like(z_latent)  # Add minor noise to z_latent
                z_recon = torch.cat((y, z_samples), dim=1)  # Reconstruct z from z_ee and z_latent
                x_recon, _ = model.g(z_recon) 
                loss_recon = loss_fn(x_recon, x)

                # 2. Prediction loss:
                # Loss between the GT y and the predicted end-effector position z_ee
                loss_pred = loss_fn(z_ee, y)

                # 3. MLE loss:
                # Loss between the latent z_latent and a standard normal distribution
                # log_p_z = -0.5 * torch.sum(z_latent ** 2, dim=1)
                # loss_mle = -torch.mean(log_p_z + log_det)
                loss_mle = -torch.mean(-0.5 * torch.sum(z_latent**2, dim=1) + log_det)  

                # 4. MMD loss


                loss = (loss_recon + loss_pred) + loss_mle * mle_scale

                total_val_loss += loss.item()
                total_val_recon += loss_recon.item()
                total_val_pred += loss_pred.item()
                total_val_mle += loss_mle.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_recon = total_val_recon / len(val_loader)
        avg_val_pred = total_val_pred / len(val_loader)
        avg_val_mle = total_val_mle / len(val_loader)

        # === Logging to TensorBoard ===
        print(f"Epoch {epoch+1}/{n_epochs} | "
                f"Train Loss: {avg_train_loss:.4f}| Val Loss: {avg_val_loss:.6f} | "
                f"Train Recon: {avg_train_recon:.4f} | Val Recon: {avg_val_recon:.6f} | "
                f"Train Pred: {avg_train_pred:.4f} | Val Pred: {avg_val_pred:.6f} | "
                f"Train MLE: {avg_train_mle:.4f} | Val MLE: {avg_val_mle:.6f}")
        
        if writer:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Loss/train_recon", avg_train_recon, epoch)
            writer.add_scalar("Loss/val_recon", avg_val_recon, epoch)
            writer.add_scalar("Loss/train_pred", avg_train_pred, epoch)
            writer.add_scalar("Loss/val_pred", avg_val_pred, epoch)
            writer.add_scalar("Loss/train_mle", avg_train_mle, epoch)
            writer.add_scalar("Loss/val_mle", avg_val_mle, epoch)

        writer.flush()

    # === Save the model ===
    torch.save(model.state_dict(), f"results/models/realnvp_fkik_{runname}.pth")
    print(f"Model saved to results/models/realnvp_fkik_{runname}.pth")

    collected_y = torch.cat(collected_y, dim=0)[:max_y_samples]  # Shape: (N, 2)
    fig = plot_y_distribution(collected_y)
    fig.savefig(f"results/plots/y_distribution_{runname}.png")


if __name__ == "__main__":
    torch.manual_seed(42)
    # Load dataset
    full_dataset = FKIKDataset('data/fk_ik_dataset.pt')

    # Tensorboard logging
    args = parser.parse_args()
    runname = args.save_name
    print(f"Run name: {runname}")
    writer = SummaryWriter(log_dir=f"runs/realnvp_fkik{runname}")

    # Split into train/val (60/40)
    n_total = len(full_dataset)
    n_train = int(0.6 * n_total)
    n_val = n_total - n_train
    # train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    train_dataset = full_dataset  # Use the full dataset for training
    val_dataset = full_dataset  # Use the full dataset for validation

    # Hyperparameters
    batch_size = 64
    epoch = 500
    num_blocks = 6  # Number of blocks in the RealNVP model
    hidden_dim = 128  # Hidden dimension for the affine coupling layers
    learning_rate = 5e-3  # Learning rate for the optimizer

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Direction
    train_direction = 'fk'  # or 'ik'

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Model
    dim = full_dataset[0]['joint_config'].shape[0] if train_direction == 'fk' else full_dataset[0]['ee_pose'].shape[0]
    prior = distributions.Normal(   # isotropic standard normal distribution
            torch.tensor(0.).to(device), torch.tensor(1.).to(device))
    model = RealNVP1D(dim=dim, hidden_dim=hidden_dim, num_blocks=num_blocks, prior=prior)

    # Train
    train(model, train_loader, val_loader, train_direction=train_direction, device=device, n_epochs=epoch, lr=learning_rate, writer=writer)
    writer.close()
