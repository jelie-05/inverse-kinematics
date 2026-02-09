from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class OptimizerConfig:
    name: str = 'adamw'
    lr: float = 0.001
    weight_decay: float = 0.0001
    betas: tuple = (0.9, 0.999)

@dataclass
class SchedulerConfig:
    name: str = 'lambdalr'
    warmup_steps: float = 0.05

@dataclass
class LossConfig:
    distribution_loss: str = 'mmd'  # Options: 'mle' or 'mmd'
    scale_mle: Optional[float] = 1.0  # Scaling factor for the distribution loss
    noise_scale: Optional[float] = 0.01  # Scale of noise added to z_latent
    lambd_mmd_forw: Optional[float] = 50.
    lambd_mmd_back: Optional[float] = 500.0
    lambd_reconstruct: Optional[float] = 1.0
    lambd_fit_forw: Optional[float] = 1.0
    y_uncertainty_sigma: Optional[float] = 0.12 * 4
    mmd_forw_kernels: Optional[List[Tuple[float, float]]] = None
    mmd_back_kernels: Optional[List[Tuple[float, float]]] = None
    mmd_back_weighted: Optional[bool] = False

@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_sampling: int = 1000
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    loss_fn: LossConfig = LossConfig()
    seed: int = 42  # Random seed for reproducibility

@dataclass
class ModelConfig:
    name: str = 'realnvp1d'
    num_blocks: int = 8
    hidden_dim: int = 128
    io_dimension: int = 4  # Input/Output dimension for the model
    latent_dim: int = 2

@dataclass
class ExperimentMeta:
    experiment_name: str = 'mmd_training'
    experiment: str = 'contact_point'
    num_cp_per_ee: int = 2
    sampling_latent: str = 'add_noise'  # Options: 'add_noise' or 'pure_noise'
    run_id: str = '20250723_001'  # Unique identifier for the run

@dataclass
class ExperimentConfig:
    experiment: ExperimentMeta = ExperimentMeta()
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
