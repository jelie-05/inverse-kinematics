import torch

def generate_fk_ik_dataset(n_samples, save_path='fk_ik_dataset.pt'):
    """
    Generate and save FK-IK dataset.
    Saves both x (IK target) and y (FK output) in one file.
    """
    sigma = torch.tensor([0.25, 0.5, 0.5, 0.5])
    x = torch.randn(n_samples, 4) * sigma  # x: (N, 4)
    x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    l1, l2, l3 = 0.5, 0.5, 1.0

    y1 = x1 + l1 * torch.sin(x2) + l2 * torch.sin(x3 - x2) + l3 * torch.sin(x4 - x2 - x3)
    y2 = l1 * torch.cos(x2) + l2 * torch.cos(x3 - x2) + l3 * torch.cos(x4 - x2 - x3)
    y = torch.stack([y1, y2], dim=1)  # y: (N, 2)

    # Save both directions in one file
    torch.save({'joint_config': x, 'ee_pose': y}, save_path)


if __name__ == "__main__":
    n_train_val_samples = 500  # Number of samples to generate
    save_path_train_val = 'fk_ik_dataset.pt'  # Path to save the dataset
    generate_fk_ik_dataset(n_train_val_samples, save_path_train_val)

    n_train_test_samples = 100 # Number of samples to generate for testing
    save_path_test = 'fk_ik_dataset_test.pt'

    print(f"Training and Vaidation Dataset saved to {save_path_train_val}")
    print(f"Test Dataset saved to {save_path_test}")
