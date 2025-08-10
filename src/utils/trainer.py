from matplotlib import scale
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
            test_size=10000,
            ):
        self.model = model
        self.device = device
        self.config = config
        self.test_size = test_size

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
            raise NotImplementedError(f"Loss function {self.training_config.loss_fn.distribution_loss} not implemented.")

        
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

        # Last test evaluation
        self.evaluation_test()

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

        # Store samples for evaluation
        if iter_sampling == 0:
            self.n_samples = 1000  # Number of samples for evaluation
            self.y_eval_each = y[10].detach().unsqueeze(0) # save as a sample for visualization
            self.y_eval = self.y_eval_each.repeat(self.n_samples, 1)  # Repeat self.n_samples times for evaluation
            self.ee_dim = z_ee.shape[1]
            self.latent_dim = z_latent.shape[1]

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
            loss_dist = self.distribution_loss_fn(z_ee, z_latent, x, y)

        loss = loss_recon + loss_pred + loss_dist
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
        self.optimizer.step()
        self.scheduler.step()

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
                loss_dist = self.distribution_loss_fn(z_ee, z_latent, x, y)

            loss = loss_recon + loss_pred + loss_dist

            # Sanity Check
            z_sanity_check = z.detach()  # Use the full z for sanity check
            x_sanity_check, _ = self.model.g(z_sanity_check)  # Reconstruct x from z_ee and z_latent
            # if not torch.allclose(x_sanity_check, x, atol=1e-4, rtol=1e-4):
            #     print(f"Sanity check failed: L2={torch.norm(x_sanity_check - x):.3e}, Max={torch.max((x_sanity_check - x).abs()):.3e}")
            #     input("Press Enter to continue...")

            val_metrics = {
                'loss': loss.item(),
                'recon_loss': loss_recon.item(),
                'pred_loss': loss_pred.item(),
                'dist_loss': loss_dist.item()
            }

            return val_metrics
        
    def evaluation_test(self):
        self.model.eval()
        with torch.no_grad():
            # Random sampling of joint configurations x
            x = torch.randn(self.test_size, self.config.model.io_dimension).to(self.device) * self.sigma

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

            # Losses
            loss_pred = self.pred_loss_fn(z_ee, y)
            loss_recon = self.recon_loss_fn(x_recon, x)

            # Rejection Sampling evaluated at y = (1.5, 0)
            results = self.rejection_sampling(y_target=torch.tensor([1.5, 0.0]).to(self.device),
                                              n_samples=self.test_size,
                                              tolerances=[0.01, 0.05, 0.1])
            
            # Calculate inference time
            start_time = time.time()
            if self.experiment_config.experiment == 'ee_point':
                for i in range(self.test_size):
                    self.arm.compute_ee_pose(x[i].unsqueeze(0))
            elif self.experiment_config.experiment == 'contact_point':
                for i in range(self.test_size):
                    self.arm.sample_contact_surface_global_batch(x[i].unsqueeze(0), n_samples_per_el=self.num_cp_per_ee)
            inference_time = time.time() - start_time
            
            # Save certain logs for later analysis in .txt file
            log_file_path = self.output_dir / "training_eval_log.txt"
            with open(log_file_path, 'w') as log_file:
                log_file.write(f"Experiment: {self.experiment_config.experiment}\n")
                log_file.write(f"Experiment Name: {self.experiment_config.experiment_name}\n")
                log_file.write(f"Run ID: {self.experiment_config.run_id}\n")
                log_file.write(f"Test Size: {self.test_size}\n")
                log_file.write(f"Reconstruction Loss: {loss_recon.item()}\n")
                log_file.write(f"Prediction Loss: {loss_pred.item()}\n")
                log_file.write(f"Inference Time for {self.test_size} samples: {inference_time:.4f} seconds; Avg: {inference_time / self.test_size:.4f} seconds\n")
                log_file.write(f"Results:\n")
                for tol, res in results.items():
                    log_file.write(f"Tolerance {tol}:\n")
                    log_file.write(f"  Acceptance Rate: {res['acceptance_rate']:.4f}\n")
                    if res['mean_error'] is not None:
                        log_file.write(f"  Mean Error: {res['mean_error']:.4f}\n")
                        log_file.write(f"  Std Error: {res['std_error']:.4f}\n")
                    else:
                        log_file.write(f"  Mean Error: None (no accepted samples)\n")
                        log_file.write(f"  Std Error: None (no accepted samples)\n")
                    if res['variance_x'] is not None:
                        log_file.write(f"  Variance of x: {res['variance_x'].cpu().numpy().tolist()}\n")

                log_file.write("===================================================\n")
                log_file.write("Configuration:\n")
                config_dict = self._config_to_dict(self.config)
                yaml.dump(config_dict, log_file, default_flow_style=False, sort_keys=False)


    def rejection_sampling(self,
                           y_target,
                           n_samples=10000,
                           tolerances=[0.01, 0.05, 0.1],
                           ):
        self.model.eval()

        y_target = y_target.unsqueeze(0)  # Shape: [1, ndim_y]
        results = {}

        z_samples = torch.randn(n_samples, self.latent_dim).to(self.device)
        y_all = y_target.expand(n_samples, -1)  # Repeat y_target n_samples times
        z_input = torch.cat([y_all, z_samples], dim=1)

        with torch.no_grad():
            x_all, _ = self.model.g(z_input)
            if self.experiment_config.experiment == 'ee_point':
                y_pred_all = self.arm.compute_ee_pose(x_all)
            elif self.experiment_config.experiment == 'contact_point':
                outputs = self.arm.sample_contact_surface_global_batch(x_all, n_samples_per_el=self.num_cp_per_ee)
                y_pred_all = outputs['global_pts']
            
            error_all = torch.norm(y_pred_all - y_target, dim=1)  # Shape: [n_samples]

        for tol in tolerances:
            accept_mask = error_all < tol
            x_accepted = x_all[accept_mask]
            y_pred_accepted = y_pred_all[accept_mask]
            err_accepted = error_all[accept_mask]

            accepted_count = x_accepted.shape[0]
            acceptance_rate = accepted_count / n_samples
            mean_error = err_accepted.mean().item() if accepted_count > 0 else None
            std_error = err_accepted.std().item() if accepted_count > 0 else None
            var_x = x_accepted.var(dim=0) if accepted_count > 0 else None

            results[tol] = {
                "accepted_x": x_accepted,
                "accepted_y_pred": y_pred_accepted,
                "acceptance_rate": acceptance_rate,
                "mean_error": mean_error,
                "std_error": std_error,
                "variance_x": var_x,
            }

            # Initialize figure for plotting
            if self.experiment_config.experiment == 'ee_point':
                fig = self.arm.plot_arm_batch(x_accepted, color='skyblue', y_sample=y_target)
            elif self.experiment_config.experiment == 'contact_point':
                fig = self.arm.plot_arm_batch(x_accepted, color='skyblue', y_sample=y_target)
            
            print(f"Accepted {accepted_count} samples out of {n_samples} for tolerance {tol}")

            # Save the figure
            idx = f"{tol:2f}".split('.')[1]
            fig_path = self.output_dir / f"rejection_sampling_plot_{idx}.png"
            fig.savefig(fig_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            print("Figure saved at:", fig_path)

        return results

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
            z_samples = torch.randn(self.n_samples, self.latent_dim).to(self.device)

            z_input = torch.cat((self.y_eval, z_samples), dim=1)

            # print(f"[LOGGING] y_eval shape: {self.y_eval.shape}, z_input shape: {z_input.shape}")
            # print(f"[LOGGING] y_eval [0]: {self.y_eval[0]}")

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
        B, D = z_latent.shape
        log_pz = -0.5 * torch.sum(z_latent ** 2, dim=1) - 0.5 * D * math.log(2 * math.pi)
        nll = -log_pz - log_det  # shape (B,)
        return torch.mean(nll) * self.training_config.loss_fn.scale_mle

    def noise_batch(self, dim, batch_size):
        noise_batch = torch.randn(batch_size, dim).to(self.device) 
        return noise_batch

    def mmd_loss(self, z_ee, z_latent, x, y):
        loss_config = self.training_config.loss_fn

        # Notes:comparing var in original repo vs this repo
        # out_y: whole output of the model <-> z_ee, z_latent
        # y: GT y + noise <-> y + noise_batch(self.latent_dim, self.batch_size_tmp)
        # x: x

        # Forward MMD loss
        output = torch.cat((z_ee.detach(), z_latent), dim=1)    # Remove gradient wrt z_ee
        y_noised = torch.cat((y, self.noise_batch(self.latent_dim, self.batch_size_tmp)), dim=1)    # Add noise to y from the target distribution

        loss_fit_forw = loss_config.lambd_fit_forw * l2_fit(z_latent, y_noised[:, :self.latent_dim], self.batch_size_tmp)
        loss_mmd_forw = loss_config.lambd_mmd_forw * torch.mean(forward_mmd(output, y_noised, loss_config.mmd_forw_kernels, self.device))

        # Backward MMD loss
        x_samples, _ = self.model.g(y_noised)
        loss_mmd_back = backward_mmd(x, x_samples, loss_config.mmd_back_kernels, self.device)
        if loss_config.mmd_back_weighted:
            loss_mmd_back *= torch.exp(-0.5 / loss_config.y_uncertainty_sigma**2 * l2_dist_matrix(y_noised, y_noised))
        
        loss_mmd_back = loss_config.lambd_mmd_back * torch.mean(loss_mmd_back)

        scaledown_mmd = 15
        return (loss_fit_forw + loss_mmd_forw + loss_mmd_back) / scaledown_mmd
