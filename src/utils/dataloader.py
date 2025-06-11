import torch
from torch.utils.data import Dataset

class FKIKDataset(Dataset):
    def __init__(self, path):
        """
        Always loads both joint_config and ee_pose.
        Direction of training is handled outside the dataset.
        """
        data = torch.load(path)
        self.joint_config = data['joint_config']
        self.ee_pose = data['ee_pose']

    def __len__(self):
        return len(self.joint_config)

    def __getitem__(self, idx):
        return {
            "joint_config": self.joint_config[idx],
            "ee_pose": self.ee_pose[idx]
        }
