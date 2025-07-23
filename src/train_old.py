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
import math
from surface_contact import RobotArm2D_CP, RobotArm2D
import argparse

from utils.losses import forward_mmd, backward_mmd

parser = argparse.ArgumentParser(description='Train RealNVP model for FK/IK')
parser.add_argument('--save_name', type=str, required=True, help='Name to save the model and logs')


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

def get_warmup_cosine_schedule(warmup_steps, total_steps, min_lr_ratio=0.05):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))  # Linear warmup
        else:
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay  # Cosine decay to min_lr_ratio
    return lr_lambda
    

def train(model, experiment='contact_point', device='cpu', n_train=10, lr=1e-3, batch_size=128, writer=None, num_cp_per_ee=1, distribution_loss='mmd'):
    i = 0
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) 
    warmup_steps = int(0.05 * n_train)         # 5% of total steps as warmup
    total_steps = n_train                      
    lr_schedule_fn = get_warmup_cosine_schedule(warmup_steps, total_steps, min_lr_ratio=0.05)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule_fn)

    loss_fn = torch.nn.MSELoss()

    # Robot arm data:
    l = torch.tensor([0.5, 0.5, 1.0], device=device)  # Link lengths

    if experiment == 'contact_point':
        arm = RobotArm2D_CP(link_lengths=l)
        assert batch_size % num_cp_per_ee == 0, "Batch size must be divisible by num_samples_per_cp"
        batch_size_tmp = batch_size // num_cp_per_ee  # Adjust batch size for contact points
    elif experiment == 'ee_point':
        arm = RobotArm2D(link_lengths=l)
        num_cp_per_ee = 1  # For end-effector point, we sample only one point per joint configuration
        batch_size_tmp = batch_size  # Use full batch size for end-effector points
    else:
        raise ValueError("Experiment must be either 'contact_point' or 'ee_point'")
    
    # Miscallaneous
    collected_y = []
    max_y_samples = 15000
    noise_scale = 0.001  # Scale for noise added to sampling of latent space
    scale = 0.05
    count_avg = 0

    model.train()

    for i in range(n_train):
        # Average losses over n iterations
        n_avg = 5000    # total samples: n_avg * batch_size
        if i % n_avg == 0:
            if i != 0:
                # Log average losses
                avg_train_loss = total_train_loss / n_avg
                avg_val_loss = total_val_loss / n_avg
                avg_train_recon = total_train_recon / n_avg
                avg_val_recon = total_val_recon / n_avg
                avg_train_pred = total_train_pred / n_avg
                avg_val_pred = total_val_pred / n_avg
                avg_train_mle = total_train_mle / n_avg
                avg_val_mle = total_val_mle / n_avg

                print(f"Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f} | "
                      f"Avg Recon L2 (T/V): {avg_train_recon:.4f} / {avg_val_recon:.4f} | "
                      f"Avg Pred L2 (T/V): {avg_train_pred:.4f} / {avg_val_pred:.4f} | "
                      f"Avg MLE Loss (T/V): {avg_train_mle:.4f} / {avg_val_mle:.4f}")
                if writer:
                    # write in Avg/
                    writer.add_scalar('Avg/train_avg_total', avg_train_loss, count_avg)
                    writer.add_scalar('Avg/val_avg_total', avg_val_loss, count_avg)
                    writer.add_scalar('Avg/train_avg_recon', avg_train_recon, count_avg)
                    writer.add_scalar('Avg/val_avg_recon', avg_val_recon, count_avg)
                    writer.add_scalar('Avg/train_avg_pred', avg_train_pred, count_avg)
                    writer.add_scalar('Avg/val_avg_pred', avg_val_pred, count_avg)
                    writer.add_scalar('Avg/train_avg_mle', avg_train_mle, count_avg)
                    writer.add_scalar('Avg/val_avg_mle', avg_val_mle, count_avg)
                    writer.flush()
                count_avg += 1

            total_train_loss = 0.0
            total_val_loss = 0.0
            total_train_recon = 0.0
            total_val_recon = 0.0
            total_train_pred = 0.0
            total_val_pred = 0.0
            total_train_mle = 0.0
            total_val_mle = 0.0


        # Generate a batch of x and corresponding y
        sigma = torch.tensor([0.25, 0.5, 0.5, 0.5]).to(device)  # Standard deviations for joint configurations
        x = torch.randn(batch_size_tmp, 4).to(device) * sigma

        if experiment == 'ee_point':
            # End-effector position is a point in 2D space
            y = arm.compute_ee_pose(x)  # Compute end-effector positions from joint configurations
        elif experiment == 'contact_point':
            outputs = arm.sample_contact_surface_global_batch(x, n_samples_per_el=num_cp_per_ee)  # Sample contact surface points
            y = outputs['global_pts']  # Shape: (B * n_samples_per_el, 2)
            x = outputs['x_batch']  # Joint configurations corresponding to the sampled points

        # Forward pass through the model
        z, log_det = model(x)
        z_ee, z_latent = z[:, :y.shape[1]], z[:, y.shape[1]:]   # Split z into ee and latent parts

        # Sanity check
        z_sanity_check = z.detach()  # Use the full z for sanity check
        x_sanity_check, _ = model.g(z_sanity_check)  # Reconstruct x from z_ee and z_latent
        if not torch.allclose(x_sanity_check, x, atol=1e-5, rtol=1e-5):
            print(f"Sanity check failed: L2={torch.norm(x_sanity_check - x):.3e}, Max={torch.max((x_sanity_check - x).abs()):.3e}")
            input("Press Enter to continue...")

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
        loss_mle = -torch.mean(-0.5 * torch.sum(z_latent**2, dim=1) + log_det)  

        # 4. MMD loss 
        z_mme = torch.cat((z_ee.detach(), z_latent), dim=1)     # Prevent gradient flow to z_ee
        loss_mmd = torch.mean(forward_mmd(z_mme, z_recon))

        if distribution_loss == 'mle':
            loss = (loss_recon + loss_pred) + loss_mle * scale
        elif distribution_loss == 'mmd':
            loss = (loss_recon + loss_pred) + loss_mmd * scale

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        curr_train_loss = loss.item() 
        curr_train_recon = loss_recon.item()
        curr_train_pred = loss_pred.item()
        curr_train_mle = loss_mle.item()

        # === Average Logging ===
        total_train_loss += curr_train_loss
        total_train_recon += curr_train_recon
        total_train_pred += curr_train_pred
        total_train_mle += curr_train_mle

        # === Save y samples for visualization ===
        if len(collected_y) < max_y_samples:
            collected_y.append(y.detach().cpu())

        # === Evaluation ===
        model.eval()

        # === Visualization ===
        if i == 0:
            y_eval = y[10].detach().unsqueeze(0) # save as a sample for visualization
            z_sample_dim = z_latent.shape[1]  # Dimension of the latent space
        elif i % 5000 == 0 and i != 0:
            # Prepare input to inverse pass
            z_latent_sample = torch.randn((1, z_sample_dim), device=device)  # Random latent vector
            z_input = torch.cat([y_eval, z_latent_sample], dim=1)  # Concatenate target with latent
    
            # Inverse pass to get joint config
            with torch.no_grad():
                x_pred, _ = model.g(z_input)  # Shape (1, 4)

            # Plot the robot arm from x_pred 
            if experiment == 'ee_point':
                fig = arm.plot_arm(x_pred, l, title="Eval Arm at Epoch {}".format(i), color='green', y_sample=y_eval)
            elif experiment == 'contact_point':
                fig = arm.plot_arm(x_pred.squeeze(0), l, title="Eval Arm at Epoch {}".format(i), color='green', y_sample=y_eval)

            # Log to TensorBoard
            writer.add_figure("EvalArm/y_eval", fig, global_step=i)
            plt.close(fig)


        # === Validation Loop ===
        with torch.no_grad():
            model.eval()

            sigma = torch.tensor([0.25, 0.5, 0.5, 0.5]).to(device)  # Standard deviations for joint configurations
            x = torch.randn(batch_size_tmp, 4).to(device) * sigma

            if experiment == 'ee_point':
                # End-effector position is a point in 2D space
                y = arm.compute_ee_pose(x, l)  # Compute end-effector positions from joint configurations
            elif experiment == 'contact_point':
                outputs = arm.sample_contact_surface_global_batch(x, n_samples_per_el=num_cp_per_ee)  # Sample contact surface points
                y = outputs['global_pts']  # Shape: (B * n_samples_per_el, 2)
                x = outputs['x_batch']  # Joint configurations corresponding to the sampled points

                # Ensuring the order of x and y is shuffled after repeating by num_cp_per_ee
                perm = torch.randperm(x.shape[0])
                x = x[perm]
                y = y[perm]

            z, log_det = model(x)

            # Split z into ee and latent parts
            z_ee, z_latent = z[:, :y.shape[1]], z[:, y.shape[1]:]

            # Compute losses
            # 1. Reconstruction loss
            latent_samples = 'add_noise'  # 'normal' or 'noise'
            if latent_samples == 'pure_noise':
                # Option 1: z_samples comes is sampled from normal distribution
                z_samples = torch.randn_like(z_latent)  # Sample from standard normal distribution
            elif latent_samples == 'add_noise':
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
            z_mme = torch.cat((z_ee.detach(), z_latent), dim=1)     # Prevent gradient flow to z_ee
            loss_mmd = torch.mean(forward_mmd(z_mme, z_recon))

            if distribution_loss == 'mle':
                loss = (loss_recon + loss_pred) + loss_mle * scale
            elif distribution_loss == 'mmd':
                loss = (loss_recon + loss_pred) + loss_mmd * scale

            # val_loss += loss.item()
            # val_recon_l2 += torch.norm(x_recon - x, p=2).item()
            # val_pred_l2 += torch.norm(z_ee - y, p=2).item()

        curr_val_loss = loss.item() 
        curr_val_recon = loss_recon.item() 
        curr_val_pred = loss_pred.item() 
        curr_val_mle = loss_mle.item()

        # === Average Validation Logging ===
        total_val_loss += curr_val_loss
        total_val_recon += curr_val_recon
        total_val_pred += curr_val_pred
        total_val_mle += curr_val_mle

        # === Logging ===
        i += 1
        print(f"Train {i}/{n_train} | "
              f"Train Loss: {curr_train_loss:.6f} | Val Loss: {curr_val_loss:.6f} | "
              f"Recon L2 (T/V): {curr_train_recon:.4f} / {curr_val_recon:.4f} | "
              f"Pred L2 (T/V): {curr_train_pred:.4f} / {curr_val_pred:.4f} | "
              f"MLE Loss (T/V): {curr_train_mle:.4f} / {curr_val_mle:.4f}")
        
        current_lr = optimizer.param_groups[0]['lr']
        # print(f"Learning Rate: {current_lr:.6e}")
          

        if writer:
            writer.add_scalar('Loss/train_total', curr_train_loss, i)
            writer.add_scalar('Loss/val_total', curr_val_loss, i)
            writer.add_scalar('Detail/train_recon', curr_train_recon, i)
            writer.add_scalar('Detail/val_recon', curr_val_recon, i)
            writer.add_scalar('Detail/train_pred', curr_train_pred, i)
            writer.add_scalar('Detail/val_pred', curr_val_pred, i)
            writer.add_scalar('Detail/train_mle', curr_train_mle, i)
            writer.add_scalar('Detail/val_mle', curr_val_mle, i)
            writer.add_scalar("LearningRate", current_lr, i)

        writer.flush()

    collected_y = torch.cat(collected_y, dim=0)[:max_y_samples]  # Shape: (N, 2)
    fig = plot_y_distribution(collected_y)
    fig.savefig(f"results/plots/y_distribution_{runname}.png")


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(42)

    # Tensorboard logging
    runname = args.save_name
    print(f"Run name: {runname}")
    writer = SummaryWriter(log_dir=f"runs/realnvp_fkik{runname}")

    # Hyperparameters
    batch_size = 128
    i = 75000 # * 4
    num_blocks = 8  # Number of blocks in the RealNVP model
    hidden_dim = 128  # Hidden dimension for the affine coupling layers
    learning_rate = 5e-3  # Learning rate for the optimizer
    num_cp_per_ee = 2  # Number of samples per contact point for the contact point experiment

    # Problem parameters
    dim = 4

    # Direction
    experiment = 'contact_point'  # 'ee_point' or 'contact_point'

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Model 
    prior = distributions.Normal(   # isotropic standard normal distribution
            torch.tensor(0.).to(device), torch.tensor(1.).to(device))
    model = RealNVP1D(dim=dim, hidden_dim=hidden_dim, num_blocks=num_blocks, prior=prior, writer=writer)

    # Train
    train(model, experiment=experiment, device=device, n_train=i, lr=learning_rate, batch_size=batch_size, writer=writer, num_cp_per_ee=num_cp_per_ee)
    writer.close()
