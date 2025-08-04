import torch
from models.ardizzone import Model
from utils.plot_configuration import plot_configuration


def plot_distribution(model: Model,  ee_pos, device):
    n_samples = 10

    # ee_pos assumed to have shape (1, D)
    y_size = ee_pos.shape[1]
    x_size = len(model.masks[0])

    z_size = x_size - y_size

    z_latent = torch.randn(n_samples, z_size).to(device)
    z_with_ee = torch.cat([ee_pos.repeat(n_samples, 1), z_latent], dim=1)

    print(ee_pos)
    print(z_with_ee)

    x_pred = model.inverse(z_with_ee)

    print(x_pred)

    plot_configuration(x_pred)


if __name__ == '__main__':

    PATH = 'checkpoints/final_model.pth'
    device = 'cpu'

    saved_model = torch.load(PATH)
    model = Model(saved_model['nr_blocks'], saved_model['masks'])
    model.load_state_dict(saved_model['model_state_dict'])
    model.eval()

    # ee_pos = torch.rand(1, 2)
    ee_pos = torch.tensor([[3, 4]], dtype=torch.float32)
    print(ee_pos.shape)

    plot_distribution(model, ee_pos, device)