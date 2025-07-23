import torch
import torch.nn as nn
import math
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
from surface_contact import RobotArm2D, RobotArm2D_CP
from utils.losses import *
from tqdm import tqdm
import yaml


class Trainer():
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            config,
            ):
        self.model = model
        self.device = device
        self.config = config

        self.training_config = config.training
        self.experiment_config = config.experiment

        self.num_sampling = self.training_config.num_sampling

        # Initialize optimizer
        if self.training_config.optimizer.name == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.training_config.optimizer.lr, 
                                              weight_decay=self.training_config.optimizer.weight_decay)
        elif self.training_config.optimizer.name == 'adamw':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.training_config.optimizer.lr,
                                                weight_decay=self.training_config.optimizer.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.training_config.optimizer.name} is not implemented.")
        
        # Initialize learning rate scheduler
        if self.training_config.scheduler.name == 'cosine':
            warmup_steps = int(self.training_config.scheduler.warmup_steps * self.num_sampling)
            lr_schedule_fn = self.get_warmup_cosine_schedule(warmup_steps, self.num_sampling, min_lr_ratio=0.05)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_schedule_fn)
        else:
            raise NotImplementedError(f"Scheduler {self.training_config.scheduler.name} is not implemented.")
        
        # Initialize loss function
        self.recon_loss_fn = torch.nn.MSELoss()  
        self.pred_loss_fn = torch.nn.MSELoss()
        if self.training_config.loss_fn.distribution_loss == 'mmd':
            self.distribution_loss_fn = self.mmd_loss
        elif self.training_config.loss_fn.distribution_loss == 'mle':
            self.distribution_loss_fn = self.mle_loss
        else:
            raise NotImplementedError(f"Loss function {self.training_config.loss_fn.distribution_loss} is not implemented.")
        
        # Logging setup
        self.experiment_dir = Path(f"{os.getcwd()}/outputs/{self.experiment_config.experiment_name}")
        print(f"Experiment directory: {self.experiment_dir}")
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.output_dir = self.experiment_dir / f"run_{self.experiment_config.run_id}"
        if self.output_dir.exists():
            suffix = time.strftime("%Y%m%d-%H%M%S")
            print(f"Output directory {self.output_dir} already exists. Adding suffix: {suffix}")
            self.output_dir = self.experiment_dir / f"run_{self.experiment_config.run_id}_{suffix}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        tensorboard_dir = self.output_dir / "tensorboard"
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_dir)

        # Sampling and Robot parameters
        self.sigma = torch.tensor([0.25, 0.5, 0.5, 0.5]).to(self.device)  # Standard deviation for noise
        self.l = torch.tensor([0.5, 0.5, 1.0], device=self.device)

        # Calculate effective batch size
        if self.experiment_config.experiment == 'contact_point':
            self.arm = RobotArm2D_CP(link_lengths=self.l)
            self.num_cp_per_ee = self.experiment_config.num_cp_per_ee
            assert self.training_config.batch_size % self.num_cp_per_ee == 0, "Batch size must be divisible by num_samples_per_cp"
            self.batch_size_tmp = self.training_config.batch_size // self.num_cp_per_ee  # Adjust batch size for contact points
        elif self.experiment_config.experiment == 'ee_point':
            self.arm = RobotArm2D(link_lengths=self.l)
            self.num_cp_per_ee = 1  # For end-effector point, we sample only one point per joint configuration
            self.batch_size_tmp = self.training_config.batch_size  # Use full batch size for end-effector points
        else:
            raise ValueError("Experiment must be either 'contact_point' or 'ee_point'")
        
    def train(self):
        print(f"Starting training for {self.num_sampling} iterations...")
        start_time = time.time()
        count_avg = 0
        total_train_loss = 0.0
        total_val_loss = 0.0
        total_train_recon = 0.0
        total_val_recon = 0.0
        total_train_pred = 0.0
        total_val_pred = 0.0
        total_train_distr = 0.0
        total_val_distr = 0.0
        best_val_loss = float('inf')

        pbar = tqdm(range(self.num_sampling))

        for iter_sampling, _ in enumerate(pbar):
            # Logging/Writing the training information
            avg_num_samples = 5000
            if iter_sampling % avg_num_samples == 0 and iter_sampling > 0:
                # Log average losses
                avg_train_loss = total_train_loss / avg_num_samples
                avg_val_loss = total_val_loss / avg_num_samples
                avg_train_recon = total_train_recon / avg_num_samples
                avg_val_recon = total_val_recon / avg_num_samples
                avg_train_pred = total_train_pred / avg_num_samples
                avg_val_pred = total_val_pred / avg_num_samples
                avg_train_distr = total_train_distr / avg_num_samples
                avg_val_distr = total_val_distr / avg_num_samples

                print(f"Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f} | "
                      f"Avg Recon L2 (T/V): {avg_train_recon:.4f} / {avg_val_recon:.4f} | "
                      f"Avg Pred L2 (T/V): {avg_train_pred:.4f} / {avg_val_pred:.4f} | "
                      f"Avg Distribution Loss (T/V): {avg_train_distr:.4f} / {avg_val_distr:.4f}")
                if self.writer:
                    # write in Avg/
                    self.writer.add_scalar('Avg/train_avg_total', avg_train_loss, count_avg)
                    self.writer.add_scalar('Avg/val_avg_total', avg_val_loss, count_avg)
                    self.writer.add_scalar('Avg/train_avg_recon', avg_train_recon, count_avg)
                    self.writer.add_scalar('Avg/val_avg_recon', avg_val_recon, count_avg)
                    self.writer.add_scalar('Avg/train_avg_pred', avg_train_pred, count_avg)
                    self.writer.add_scalar('Avg/val_avg_pred', avg_val_pred, count_avg)
                    self.writer.add_scalar('Avg/train_avg_distr', avg_train_distr, count_avg)
                    self.writer.add_scalar('Avg/val_avg_distr', avg_val_distr, count_avg)
                    self.writer.flush()
                
                # Update and Reset accumulators
                count_avg += 1
                total_train_loss = 0.0
                total_val_loss = 0.0
                total_train_recon = 0.0
                total_val_recon = 0.0
                total_train_pred = 0.0
                total_val_pred = 0.0
                total_train_distr = 0.0
                total_val_distr = 0.0

            # Training step
            train_metrics = self._train_step(iter_sampling)

            # Example Sampling for Visualization
            if iter_sampling % avg_num_samples == 0 and iter_sampling > 0:
                self._logging_sampling(iter_sampling)
            
            # Validation step
            val_metrics = self._validate_step(iter_sampling)

            # Accumulate losses
            total_train_loss += train_metrics['loss']
            total_val_loss += val_metrics['loss']
            total_train_recon += train_metrics['recon_loss']
            total_val_recon += val_metrics['recon_loss']
            total_train_pred += train_metrics['pred_loss']
            total_val_pred += val_metrics['pred_loss']
            total_train_distr += train_metrics['dist_loss']
            total_val_distr += val_metrics['dist_loss']  

            # Log s and t to TensorBoard
            if self.writer:
                self.writer.add_scalar('Scalar/log_s_max', train_metrics['logs']['log_s'], iter_sampling)
                self.writer.add_scalar('Scalar/log_t_max', train_metrics['logs']['t'], iter_sampling)

            # Update progress bar
            if iter_sampling % 500 == 0:
                pbar.set_postfix({
                    "Iter": f"{iter_sampling+1}/{self.num_sampling}",
                    "TrainLoss": f"{total_train_loss / (iter_sampling + 1):.6f}",
                    "ValLoss": f"{total_val_loss / (iter_sampling + 1):.6f}"
                })

            # Save model periodically
            if iter_sampling % (avg_num_samples * 3) == 0 and iter_sampling > 0:
                model_save_path = self.output_dir / f"model_epoch_{iter_sampling}.pth"
                torch.save(self.model.state_dict(), model_save_path)
                print(f"Model saved at {model_save_path}")

            # Save best model based on validation loss
            if val_metrics['loss'] < best_val_loss and iter_sampling % avg_num_samples == 0 and iter_sampling > 0:
                best_val_loss = val_metrics['loss']
                best_model_path = self.output_dir / "best_model.pth"
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Best model saved at {best_model_path} with validation loss: {best_val_loss:.6f}")

        # Save the model
        final_model_path = self.output_dir / "final_model.pth"
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Final model saved at {final_model_path}")

    def _train_step(self, iter_sampling):
        self.model.train()
        self.optimizer.zero_grad()

        # Forward Kinematics using random sampling x
        x = torch.randn(self.batch_size_tmp, self.config.model.io_dimension).to(self.device) * self.sigma
        if self.experiment_config.experiment == 'ee_point':
            # End-effector position is a point in 2D space
            y = self.arm.compute_ee_pose(x)  # Compute end-effector positions from joint configurations
        elif self.experiment_config.experiment == 'contact_point':
            outputs = self.arm.sample_contact_surface_global_batch(x, n_samples_per_el=self.num_cp_per_ee)  # Sample contact surface points
            y = outputs['global_pts']  # Shape: (B * n_samples_per_el, 2)
            x = outputs['x_batch']  # Joint configurations corresponding to the sampled points

        # Forward pass through the model
        z, log_det, logs = self.model(x)
        z_ee, z_latent = z[:, :y.shape[1]], z[:, y.shape[1]:]   # Split z into ee and latent parts

        # Compute Losses
        # Reconstruction Loss
        if self.experiment_config.sampling_latent == 'pure_noise':
            z_samples = torch.randn_like(z_latent)  # Sample from standard normal distribution
        elif self.experiment_config.sampling_latent == 'add_noise':
            z_samples = z_latent + self.training_config.loss_fn.noise_scale * torch.randn_like(z_latent)  # Add minor noise to z_latent
        z_recon = torch.cat((y, z_samples), dim=1)  # Concatenate GT y and z_samples
        x_recon, _ = self.model.g(z_recon)
        loss_recon = self.recon_loss_fn(x_recon, x)

        # Prediction Loss
        loss_pred = self.pred_loss_fn(z_ee, y)

        # Distribution Loss
        if self.training_config.loss_fn.distribution_loss == 'mle':
            loss_dist = self.distribution_loss_fn(z_latent, log_det)
        elif self.training_config.loss_fn.distribution_loss == 'mmd':
            loss_dist = self.mmd_loss(z_ee, z_latent, x)

        loss = loss_recon + loss_pred + loss_dist
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
        self.optimizer.step()
        self.scheduler.step()

        # Store samples for evaluation
        if iter_sampling == 0:
            self.n_samples = 500
            self.y_eval_each = y[10].detach().unsqueeze(0) # save as a sample for visualization
            self.y_eval = self.y_eval_each.repeat(self.n_samples, 1)  # Repeat self.n_samples times for evaluation
            self.z_sample_dim = z_latent.shape[1]
        
        train_metrics = {
            'loss': loss.item(),
            'recon_loss': loss_recon.item(),
            'pred_loss': loss_pred.item(),
            'dist_loss': loss_dist.item(),
            'logs': logs,
        }


        return train_metrics
    
    def _validate_step(self, iter_sampling):
        self.model.eval()

        with torch.no_grad():
            x = torch.randn(self.batch_size_tmp, self.config.model.io_dimension).to(self.device) * self.sigma

            if self.experiment_config.experiment == 'ee_point':
                y = self.arm.compute_ee_pose(x)
            elif self.experiment_config.experiment == 'contact_point':
                outputs = self.arm.sample_contact_surface_global_batch(x, n_samples_per_el=self.num_cp_per_ee)
                y = outputs['global_pts']
                x = outputs['x_batch']

                # Ensuring the order of x and y is shuffled after repeating by num_cp_per_ee
                perm = torch.randperm(x.shape[0])
                x = x[perm]
                y = y[perm]

            z, log_det, _ = self.model(x)
            z_ee, z_latent = z[:, :y.shape[1]], z[:, y.shape[1]:]

            # Compute Losses
            # Reconstruction Loss
            if self.experiment_config.sampling_latent == 'pure_noise':
                z_samples = torch.randn_like(z_latent)
            elif self.experiment_config.sampling_latent == 'add_noise':
                z_samples = z_latent + self.training_config.loss_fn.noise_scale * torch.randn_like(z_latent)
            z_recon = torch.cat((y, z_samples), dim=1)
            x_recon, _ = self.model.g(z_recon)
            loss_recon = self.recon_loss_fn(x_recon, x)

            # Prediction Loss
            loss_pred = self.pred_loss_fn(z_ee, y)

            # Distribution Loss
            if self.training_config.loss_fn.distribution_loss == 'mle':
                loss_dist = self.distribution_loss_fn(z_latent, log_det)
            elif self.training_config.loss_fn.distribution_loss == 'mmd':
                loss_dist = self.mmd_loss(z_ee, z_latent, x)

            loss = loss_recon + loss_pred + loss_dist

            # Sanity Check
            z_sanity_check = z.detach()  # Use the full z for sanity check
            x_sanity_check, _ = self.model.g(z_sanity_check)  # Reconstruct x from z_ee and z_latent
            if not torch.allclose(x_sanity_check, x, atol=1e-5, rtol=1e-5):
                print(f"Sanity check failed: L2={torch.norm(x_sanity_check - x):.3e}, Max={torch.max((x_sanity_check - x).abs()):.3e}")
                input("Press Enter to continue...")

            val_metrics = {
                'loss': loss.item(),
                'recon_loss': loss_recon.item(),
                'pred_loss': loss_pred.item(),
                'dist_loss': loss_dist.item()
            }

            return val_metrics

    def get_warmup_cosine_schedule(self, warmup_steps, total_steps, min_lr_ratio=0.05):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))  # Linear warmup
            else:
                progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay  # Cosine decay to min_lr_ratio
        return lr_lambda
    
    def _logging_sampling(self, iter_sampling):
        self.model.eval()
        with torch.no_grad():
            # Sample from the model
            z_samples = torch.randn(self.n_samples, self.z_sample_dim).to(self.device)

            z_input = torch.cat((self.y_eval, z_samples), dim=1)

            x_recon, _ = self.model.g(z_input)

            if self.experiment_config.experiment == 'ee_point':
                fig = self.arm.plot_arm_batch(x_recon,
                                              title="Eval Arm at Epoch {}".format(iter_sampling), 
                                              color='green', y_sample=self.y_eval)
            elif self.experiment_config.experiment == 'contact_point':
                fig = self.arm.plot_arm_batch(x_recon, 
                                              title="Eval Arm at Epoch {}".format(iter_sampling), 
                                              color='green', y_sample=self.y_eval)
                
            # Log to TensorBoard
            if self.writer:
                self.writer.add_figure("EvalArm/y_eval", fig, global_step=iter_sampling)
                plt.close(fig)

    def save_run_config(self, config):
        try:
            config_dict = self._config_to_dict(config)
            config_path = self.output_dir / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

                print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Failed to save configuration: {e}")

    def _config_to_dict(self, obj):
        if hasattr(obj, '__dict__'):
            # If it's an object with attributes, convert to dict
            result = {}
            for key, value in vars(obj).items():
                # Skip private attributes (starting with _)
                if not key.startswith('_'):
                    result[key] = self._config_to_dict(value)
            return result
        elif isinstance(obj, (list, tuple)):
            # If it's a list or tuple, convert each element
            return [self._config_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            # If it's already a dict, convert its values
            return {key: self._config_to_dict(value) for key, value in obj.items()}
        elif isinstance(obj, (int, float, str, bool, type(None))):
            # Primitive types can be used as-is
            return obj
        else:
            # For any other types, convert to string
            return str(obj)
        
    def mle_loss(self, z_latent, log_det):
        return -torch.mean(-0.5 * torch.sum(z_latent**2, dim=1) + log_det) * self.training_config.loss_fn.scale_mle

    def mmd_loss(self, z_ee, z_latent, x):
        loss_config = self.training_config.loss_fn
        mmd_forward = loss_config.scale_mmd_forw * torch.mean(
            forward_mmd(z_ee.detach(), z_latent, loss_config.mmd_forw_kernels, self.device)) 

        z_sampling = torch.cat((z_ee, z_latent), dim=1)
        x_sampling, _ = self.model.g(z_sampling)
        mmd_backward = backward_mmd(x, x_sampling, loss_config.mmd_back_kernels, self.device)
        if loss_config.mmd_back_weighted:
            mmd_backward *= torch.exp(- 0.5 / loss_config.y_uncertainty_sigma**2 *
                                      l2_dist_matrix(z_sampling, z_sampling))
            
        mmd_backward = loss_config.scale_mmd_back * torch.mean(mmd_backward)

        return mmd_forward + mmd_backward

