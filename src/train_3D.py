import torch
import torch.distributions as distributions
from torch.utils.data import DataLoader
from models.realnvp_1D import RealNVP1D
from utils.load_config import load_config_from_args
from utils.trainer_3D import Trainer3D
from utils.dataloader import Panda3DDataset
import numpy as np
import random
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    config = load_config_from_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.training.seed)

    # Initialize Model
    if config.model.name == 'realnvp1d':
        model = RealNVP1D(
            dim=config.model.io_dimension, # 14
            hidden_dim=config.model.hidden_dim,
            num_blocks=config.model.num_blocks,
            prior=distributions.Normal(
                torch.tensor(0.).to(device), 
                torch.tensor(1.).to(device)
            ),
        ).to(device)
    else:
        raise NotImplementedError(f"Model {config.model.name} is not implemented.")
    
    print(f"Model {config.model.name} created with {sum(p.numel() for p in model.parameters())} parameters.")

    # Load Data
    dataset_path = "panda_ik_data.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run src/utils/3d_generator.py first.")
    
    dataset = Panda3DDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, drop_last=True)
    
    print(f"Dataset loaded from {dataset_path} with {len(dataset)} samples.")

    # Initialize Trainer
    trainer = Trainer3D(
        model=model,
        device=device,
        config=config,
        dataloader=dataloader
    )

    # Start Training
    trainer.train()

if __name__ == "__main__":
    main()
