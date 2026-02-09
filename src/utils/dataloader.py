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

class Panda3DDataset(Dataset):
    def __init__(self, path):
        """
        Loads joint_config and ee_pose from a CSV file.
        CSV Columns: q1, q2, q3, q4, q5, q6, q7, x, y, z, qw, qx, qy, qz
        """
        import pandas as pd
        df = pd.read_csv(path)
        
        # Extract joint configurations (q1-q7)
        self.joint_config = torch.tensor(df.iloc[:, :7].values, dtype=torch.float32)
        
        # Extract EE pose (x, y, z, qw, qx, qy, qz)
        self.ee_pose = torch.tensor(df.iloc[:, 7:].values, dtype=torch.float32)

    def __len__(self):
        return len(self.joint_config)

    def __getitem__(self, idx):
        return {
            "joint_config": self.joint_config[idx],
            "ee_pose": self.ee_pose[idx]
        }

