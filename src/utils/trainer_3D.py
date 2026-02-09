import torch
import torch.nn as nn
import math
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
from utils.panda_model import PandaArm
from utils.losses import *
from tqdm import tqdm
import yaml

class Trainer3D(object):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            config,
            dataloader=None,
            test_dataloader=None
            ):
        self.model = model
        self.device = device
        self.config = config
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        
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
        
        # Initialize Panda Arm model for validation
        self.arm = PandaArm(device=self.device)
        self.latent_dim = self.config.model.latent_dim
        self.batch_size_tmp = self.training_config.batch_size # needed for mmd loss

    def train(self):
        print(f"Starting training for {self.num_sampling} iterations...")
        count_avg = 0
        total_train_loss = 0.0
        total_train_recon = 0.0
        total_train_pred = 0.0
        total_train_distr = 0.0
        
        # Create an iterator from the dataloader that cycles endlessly
        def cycle(iterable):
            while True:
                for x in iterable:
                    yield x
        
        train_iterator = cycle(self.dataloader)
        
        pbar = tqdm(range(self.num_sampling))

        for iter_sampling in pbar:
            # Get next batch
            batch = next(train_iterator)
            
            # Training step
            train_metrics = self._train_step(batch, iter_sampling)
            
            # Accumulate losses
            total_train_loss += train_metrics['loss']
            total_train_recon += train_metrics['recon_loss']
            total_train_pred += train_metrics['pred_loss']
            total_train_distr += train_metrics['dist_loss']
            
            # Logging
            avg_num_samples = 100
            if iter_sampling % avg_num_samples == 0 and iter_sampling > 0:
                avg_train_loss = total_train_loss / avg_num_samples
                avg_train_recon = total_train_recon / avg_num_samples
                avg_train_pred = total_train_pred / avg_num_samples
                avg_train_distr = total_train_distr / avg_num_samples
                
                print(f"Iter {iter_sampling} | Avg Train Loss: {avg_train_loss:.6f} | "
                      f"Recon: {avg_train_recon:.4f} | Pred: {avg_train_pred:.4f} | Distr: {avg_train_distr:.4f}")
                
                if self.writer:
                    self.writer.add_scalar('Avg/train_avg_total', avg_train_loss, iter_sampling)
                    self.writer.add_scalar('Avg/train_avg_recon', avg_train_recon, iter_sampling)
                    self.writer.add_scalar('Avg/train_avg_pred', avg_train_pred, iter_sampling)
                    self.writer.add_scalar('Avg/train_avg_distr', avg_train_distr, iter_sampling)
                    self.writer.flush()
                
                # Reset accumulators
                total_train_loss = 0.0
                total_train_recon = 0.0
                total_train_pred = 0.0
                total_train_distr = 0.0
                
            # Update progress bar
            if iter_sampling % 100 == 0:
                pbar.set_postfix({"Loss": f"{train_metrics['loss']:.6f}"})

            # Save model periodically
            if iter_sampling % 5000 == 0 and iter_sampling > 0:
                model_save_path = self.output_dir / f"model_step_{iter_sampling}.pth"
                torch.save(self.model.state_dict(), model_save_path)
                print(f"Model saved at {model_save_path}")

        # Save the final model
        final_model_path = self.output_dir / "final_model.pth"
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Final model saved at {final_model_path}")
        
    def _train_step(self, batch, iter_sampling):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get data from batch
        x = batch['joint_config'].to(self.device).float() # (B, 7)
        y = batch['ee_pose'].to(self.device).float() # (B, 7)
        
        current_batch_size = x.shape[0]
        # Update batch size tmp for MMD loss if it changes (usually last batch)
        if current_batch_size != self.batch_size_tmp:
             self.batch_size_tmp = current_batch_size
        
        # Pad x to match io_dimension (14)
        # x is (B, 7). We need (B, 14).
        # We use noise for padding as recommended in INN literature for stability
        padding = torch.randn(current_batch_size, self.config.model.io_dimension - x.shape[1]).to(self.device) * self.training_config.loss_fn.noise_scale
        x_input = torch.cat((x, padding), dim=1)

        # Forward pass through the model (Inversion: x -> z)
        # The model takes x (joints) and outputs z (latent)
        # We split z into z_ee (corresponding to y) and z_latent (remaining dimensions)
        z, log_det, logs = self.model(x_input)
        
        # z should have dimension 14 (7+7)
        # y has dimension 7
        # z_ee corresponds to the first 7 dims, z_latent to the rest
        z_ee, z_latent = z[:, :y.shape[1]], z[:, y.shape[1]:]

        # Compute Losses
        
        # Reconstruction Loss (Generation: z -> x)
        if self.experiment_config.sampling_latent == 'pure_noise':
            z_samples = torch.randn_like(z_latent)
        elif self.experiment_config.sampling_latent == 'add_noise':
            z_samples = z_latent + self.training_config.loss_fn.noise_scale * torch.randn_like(z_latent)
            
        z_recon = torch.cat((y, z_samples), dim=1) # (B, 14)
        x_recon, _ = self.model.g(z_recon) # (B, 14)
        
        # We only care about reconstructing the REAL joints x (first 7 dims)
        # But for stability we might want to reconstruct the whole thing? 
        # Usually we only care about x.
        loss_recon = self.recon_loss_fn(x_recon[:, :7], x) * self.training_config.loss_fn.lambd_reconstruct

        # Prediction Loss (Consistency in latent space for EE)
        # z_ee should match y
        loss_pred = self.pred_loss_fn(z_ee, y)

        # Distribution Loss
        if self.training_config.loss_fn.distribution_loss == 'mle':
            loss_dist = self.distribution_loss_fn(z_latent, log_det)
        elif self.training_config.loss_fn.distribution_loss == 'mmd':
            # MMD needs same dim inputs. 
            # z_ee (7), z_latent (7), x (7), y (7).
            # original MMD loss signature: (z_ee, z_latent, x, y)
            # here x is 7 dim. 
            loss_dist = self.distribution_loss_fn(z_ee, z_latent, x_input, y)

        loss = loss_recon + loss_pred + loss_dist
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        train_metrics = {
            'loss': loss.item(),
            'recon_loss': loss_recon.item(),
            'pred_loss': loss_pred.item(),
            'dist_loss': loss_dist.item(),
        }
        return train_metrics

    # Reuse MMD loss and other utilities from original Trainer if possible, 
    # but since I am creating a new class, I'll copy the necessary methods.
    
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

        # Forward MMD loss
        output = torch.cat((z_ee.detach(), z_latent), dim=1)
        y_noised = torch.cat((y, self.noise_batch(self.latent_dim, self.batch_size_tmp)), dim=1)

        loss_fit_forw = loss_config.lambd_fit_forw * l2_fit(z_latent, y_noised[:, :self.latent_dim], self.batch_size_tmp)
        loss_mmd_forw = loss_config.lambd_mmd_forw * torch.mean(forward_mmd(output, y_noised, loss_config.mmd_forw_kernels, self.device))

        # Backward MMD loss
        x_samples, _ = self.model.g(y_noised)
        loss_mmd_back = backward_mmd(x, x_samples, loss_config.mmd_back_kernels, self.device)
        
        # Weighted MMD back (optional, simplified here)
        loss_mmd_back = loss_config.lambd_mmd_back * torch.mean(loss_mmd_back)

        scaledown_mmd = 1.0 # Removed the arbitrary 15 scaling
        return (loss_fit_forw + loss_mmd_forw + loss_mmd_back) / scaledown_mmd

    def get_warmup_cosine_schedule(self, warmup_steps, total_steps, min_lr_ratio=0.05):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
        return lr_lambda