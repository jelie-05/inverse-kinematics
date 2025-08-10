import torch
import torch.distributions as distributions
from models.realnvp_1D import RealNVP1D
from models.freia_inn import Freia_INN
from utils.load_config import load_config_from_args
from utils.trainer import Trainer
from torchinfo import summary
import numpy as np
import random
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    config = load_config_from_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init seed for reproducibility
    # torch.manual_seed(config.training.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(config.training.seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    set_seed(config.training.seed)

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

    else:
        raise NotImplementedError(f"Model {config.model.name} is not implemented.")
    
    # print summary of the model
    try:
        summary(model)
        print("Model created successfully!")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        print(f"Could not print model summary: {e}")
    
    trainer = Trainer(
        model=model,
        device=device,
        config=config
    )

    trainer.train()  # Assuming the Trainer class has a train method

    print("Training completed.")


if __name__ == "__main__":
    main()
