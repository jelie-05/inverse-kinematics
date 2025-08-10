import torch
from surface_contact import RobotArm2D, RobotArm2D_CP
import matplotlib.pyplot as plt
from models.realnvp_1D import RealNVP1D
from models.freia_inn import Freia_INN
from utils.load_config import load_config_from_args
import torch.distributions as distributions
import yaml
import time
import os
from pathlib import Path


class Inference():
    def __init__(self,
                 model,
                 device='cpu',
                 test_size=1000,
                 config=None,):
        self.model = model.to(device)
        self.device = device
        self.test_size = test_size
        
        self.config = config
        self.training_config = config.training
        self.experiment_config = config.experiment if config else None

        # Logging setup
        self.experiment_dir = Path(f"{os.getcwd()}/outputs/{self.experiment_config.experiment_name}")
        print(f"Experiment directory: {self.experiment_dir}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.output_dir = self.experiment_dir / f"evaluation_inference/run_{self.experiment_config.run_id}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.z_sample_dim = 2

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
        
        self.recon_loss_fn = torch.nn.MSELoss()  
        # self.pred_loss_fn = torch.nn.MSELoss()

    def evaluation_test(self):
        self.model.eval()
        with torch.no_grad():
            # Random sampling of joint configurations x
            self.x = torch.randn(self.test_size, self.config.model.io_dimension).to(self.device) * self.sigma

            if self.experiment_config.experiment == 'ee_point':
                self.y = self.arm.compute_ee_pose(self.x)
            elif self.experiment_config.experiment == 'contact_point':
                outputs = self.arm.sample_contact_surface_global_batch(self.x, n_samples_per_el=self.num_cp_per_ee)
                self.y = outputs['global_pts']
                self.x = outputs['x_batch']

                # Ensuring the order of x and y is shuffled after repeating by num_cp_per_ee
                perm = torch.randperm(self.x.shape[0])
                self.x = self.x[perm]
                self.y = self.y[perm]

            z, log_det, _ = self.model(self.x)
            z_ee, self.z_latent = z[:, :self.y.shape[1]], z[:, self.y.shape[1]:]

            # Compute Losses
            # Reconstruction Loss
            if self.experiment_config.sampling_latent == 'pure_noise':
                z_samples = torch.randn_like(self.z_latent)
            elif self.experiment_config.sampling_latent == 'add_noise':
                z_samples = self.z_latent + self.training_config.loss_fn.noise_scale * torch.randn_like(self.z_latent)
            z_recon = torch.cat((self.y, z_samples), dim=1)
            x_recon, _ = self.model.g(z_recon)

            # # Losses
            # loss_pred = torch.norm(z_ee - self.y, dim=1, p=2).mean()  # Using L2 norm for prediction loss
            # loss_recon = self.recon_loss_fn(x_recon, self.x)

            # Evaluation metrics
            mean_error, std_error = self.distance_to_target()

            results = self.rejection_sampling()
            
            # Calculate inference time -> iterate instead of batch processing
            start_time = time.time()
            if self.experiment_config.experiment == 'ee_point':
                for i in range(self.test_size):
                    self.arm.compute_ee_pose(self.x[i].unsqueeze(0))
            elif self.experiment_config.experiment == 'contact_point':
                for i in range(self.test_size):
                    self.arm.sample_contact_surface_global_batch(self.x[i].unsqueeze(0), n_samples_per_el=self.num_cp_per_ee)
            inference_time = time.time() - start_time
            
            # Save certain logs for later analysis in .txt file
            log_file_path = self.output_dir / "training_eval_log.txt"
            with open(log_file_path, 'w') as log_file:
                log_file.write(f"Experiment: {self.experiment_config.experiment}\n")
                log_file.write(f"Experiment Name: {self.experiment_config.experiment_name}\n")
                log_file.write(f"Run ID: {self.experiment_config.run_id}\n")
                log_file.write(f"Test Size: {self.test_size}\n")
                # log_file.write(f"Reconstruction Loss: {loss_recon.item()}\n")
                # log_file.write(f"Prediction Loss: {loss_pred.item()}\n")
                log_file.write(f"Avg L2 Error: {mean_error:.4f}\n")
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
                log_file.write(f"Model Name: {self.config.model.name}\n")
                log_file.write(f"Number of parameters in model: {sum(p.numel() for p in self.model.parameters())}\n")
                log_file.write("===================================================\n")
                log_file.write("Configuration:\n")
                config_dict = self._config_to_dict(self.config)
                yaml.dump(config_dict, log_file, default_flow_style=False, sort_keys=False)

    def distance_to_target(self):
        # Sample n_samples y_target randomly
        z_samples = torch.randn_like(self.z_latent)
        z_input = torch.cat((self.y, z_samples), dim=1)

        with torch.no_grad():
            x_all, _ = self.model.g(z_input)
            if self.experiment_config.experiment == 'ee_point':
                y_pred_all = self.arm.compute_ee_pose(x_all)
                error_all = torch.norm(y_pred_all - self.y, dim=1, p=2)  # Shape: [n_samples]

            elif self.experiment_config.experiment == 'contact_point':
                n_samples_per_el = 100
                outputs = self.arm.sample_contact_surface_global_batch(x_all, n_samples_per_el=n_samples_per_el)
                y_pred_all = outputs['global_pts'].view(self.test_size, n_samples_per_el, 2)
                error_all_points = torch.norm(y_pred_all - self.y.unsqueeze(1), dim=2, p=2)  # Shape: [n_samples, n_samples_per_el]
                error_all = torch.min(error_all_points, dim=1).values

            return error_all.mean().item(), error_all.std().item()

    def rejection_sampling(self,
                        tolerances=[0.01, 0.05, 0.1, 0.15, 0.2]):
        import matplotlib.pyplot as plt
        from collections import defaultdict
        import gc  # for manual memory management

        self.model.eval()
        n_samples = 500
        y_targets = self.y.to(self.device)

        # Add a test target [1.5, 0.0]
        y_targets = torch.cat((y_targets, torch.tensor([[1.5, 0.0]], device=self.device)), dim=0)

        results_avg = {tol: defaultdict(float) for tol in tolerances}
        results_avg_counts = {tol: 0 for tol in tolerances}

        for idx_y, y_target in enumerate(y_targets):
            y_target = y_target.unsqueeze(0)  # [1, ndim_y]
            y_all = y_target.expand(n_samples, -1)

            z_samples = torch.randn(n_samples, self.z_sample_dim, device=self.device)
            z_input = torch.cat((y_all, z_samples), dim=1)

            with torch.no_grad():
                x_all, _ = self.model.g(z_input)

                if self.experiment_config.experiment == 'ee_point':
                    y_pred_all = self.arm.compute_ee_pose(x_all)  # (n_samples, 2)
                    error_all = torch.norm(y_pred_all - y_all, dim=1, p=2)

                elif self.experiment_config.experiment == 'contact_point':
                    n_samples_per_el = 100
                    outputs = self.arm.sample_contact_surface_global_batch(
                        x_all, n_samples_per_el=n_samples_per_el)
                    y_pred_all = outputs['global_pts'].view(n_samples, n_samples_per_el, 2)
                    error_all_points = torch.norm(y_pred_all - y_all.unsqueeze(1), dim=2, p=2)
                    error_all = torch.min(error_all_points, dim=1).values

                # Plot only for the special case [1.5, 0.0]
                if idx_y == len(y_targets) - 1:
                    fig = self.arm.plot_arm_batch(x_all.cpu(), color='skyblue', y_sample=y_all.cpu())
                    fig_path = self.output_dir / f"rejection_sampling_initial_plot_target_{idx_y}.png"
                    fig.savefig(fig_path, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    print("Initial figure saved at:", fig_path)

            # Evaluate each tolerance
            for tol in tolerances:
                accept_mask = error_all < tol
                x_accepted = x_all[accept_mask]

                err_accepted = error_all[accept_mask]
                accepted_count = x_accepted.shape[0]

                acceptance_rate = accepted_count / n_samples
                mean_error = err_accepted.mean().item() if accepted_count > 0 else 0.0
                std_error = err_accepted.std().item() if accepted_count > 0 else 0.0
                var_x = x_accepted.var(dim=0) if accepted_count > 0 else torch.zeros_like(x_all[0])

                # Accumulate
                results_avg[tol]["acceptance_rate"] += acceptance_rate
                results_avg[tol]["mean_error"] += mean_error
                results_avg[tol]["std_error"] += std_error
                results_avg[tol]["variance_x"] += var_x
                results_avg_counts[tol] += 1

                # print(f"Accepted {accepted_count} samples out of {n_samples} for tolerance {tol:.3f}")

                # Plot accepted only for the test target
                if idx_y == len(y_targets) - 1:
                    fig = self.arm.plot_arm_batch(x_accepted.cpu(), color='skyblue', y_sample=y_target.cpu())
                    tol_id = f"{tol:0.2f}".replace(".", "_")
                    fig_path = self.output_dir / f"rejection_sampling_plot_tol_{tol_id}_target_{idx_y}.png"
                    fig.savefig(fig_path, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    print("Figure saved at:", fig_path)

            # Clean up memory
            del x_all, y_pred_all, error_all, z_samples, z_input
            if self.experiment_config.experiment == 'contact_point':
                del outputs
            gc.collect()
            torch.cuda.empty_cache()

        # Average the results
        for tol in tolerances:
            count = results_avg_counts[tol]
            if count > 0:
                results_avg[tol]["acceptance_rate"] /= count
                results_avg[tol]["mean_error"] /= count
                results_avg[tol]["std_error"] /= count
                results_avg[tol]["variance_x"] /= count
            else:
                results_avg[tol]["acceptance_rate"] = 0.0
                results_avg[tol]["mean_error"] = 0.0
                results_avg[tol]["std_error"] = 0.0
                results_avg[tol]["variance_x"] = torch.zeros_like(y_targets[0])

        print("\n=== Final Averaged Results ===")
        for tol in tolerances:
            print(f"Tolerance {tol:.3f} -> "
                f"Acceptance Rate: {results_avg[tol]['acceptance_rate']:.4f}, "
                f"Mean Error: {results_avg[tol]['mean_error']:.5f}, "
                f"Std Error: {results_avg[tol]['std_error']:.5f}")

        return results_avg


    def rejection_sampling_old(self,
                           y_target,
                           n_samples=10000,
                           tolerances=[0.01, 0.05, 0.1, 0.15, 0.2],
                           ):
        self.model.eval()

        y_target = y_target.unsqueeze(0)  # Shape: [1, ndim_y]
        y_all = y_target.repeat(n_samples, 1)  # Repeat y_target n_samples times

        z_samples = torch.randn(n_samples, self.z_sample_dim).to(self.device)
        z_input = torch.cat((y_all, z_samples), dim=1)

        results = {}

        with torch.no_grad():
            x_all, _ = self.model.g(z_input)
            if self.experiment_config.experiment == 'ee_point':
                y_pred_all = self.arm.compute_ee_pose(x_all)
                error_all = torch.norm(y_pred_all - y_all, dim=1, p=2)  # Shape: [n_samples]

            elif self.experiment_config.experiment == 'contact_point':
                n_samples_per_el = 100
                outputs = self.arm.sample_contact_surface_global_batch(x_all, n_samples_per_el=n_samples_per_el)
                y_pred_all = outputs['global_pts'].view(n_samples, n_samples_per_el, 2)
                print(f"y_pred_all shape: {y_pred_all.shape}, y_all shape: {y_all.shape}")
                error_all_points = torch.norm(y_pred_all - y_all.unsqueeze(1), dim=2, p=2)  # Shape: [n_samples, n_samples_per_el]
                error_all = torch.min(error_all_points, dim=1).values

            if self.experiment_config.experiment == 'ee_point':
                fig = self.arm.plot_arm_batch(x_all, color='skyblue', y_sample=y_all)
            elif self.experiment_config.experiment == 'contact_point':
                fig = self.arm.plot_arm_batch(x_all, color='skyblue', y_sample=y_all)

            fig_path = self.output_dir / "rejection_sampling_initial_plot.png"
            fig.savefig(fig_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print("Initial figure saved at:", fig_path)

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

def main():
    config = load_config_from_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init seed for reproducibility
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.training.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config.model.name == 'realnvp1d':
        model = RealNVP1D(
            dim=config.model.io_dimension,
            hidden_dim=config.model.hidden_dim,
            num_blocks=config.model.num_blocks,
            prior=distributions.Normal(
                torch.tensor(0.).to(device), 
                torch.tensor(1.).to(device)
            ),
        ).to(device)

        model_path = "outputs/" + config.experiment.experiment_name + f"/run_{config.experiment.run_id}/best_model.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
    elif config.model.name == 'freia_inn':
        model = Freia_INN(
            dim=config.model.io_dimension,
            hidden_dim=config.model.hidden_dim,
            num_blocks=config.model.num_blocks,
            prior=distributions.Normal(
                torch.tensor(0.).to(device), 
                torch.tensor(1.).to(device)
            ),
        ).to(device)
        model_path = "outputs/" + config.experiment.experiment_name + f"/run_{config.experiment.run_id}/best_model.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
    else:
        raise NotImplementedError(f"Model {config.model.name} is not implemented.")

    inference = Inference(
        model=model,
        device=device,
        config=config,
        test_size=5000,
    )

    inference.evaluation_test()  # Run the evaluation test

    print("Inference completed.")


if __name__ == "__main__":
    main()