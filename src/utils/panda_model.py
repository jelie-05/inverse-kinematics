import torch
import numpy as np
import mujoco
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class PandaArm:
    def __init__(self, xml_path='mujoco_menagerie/franka_emika_panda/panda.xml', device='cpu'):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.device = device
        
        # Identify EE body ID
        self.ee_body_name = "hand"
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)
        if self.ee_id == -1:
             # Fallback
            self.ee_body_name = "link7"
            self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)

        # Joint IDs
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]

    def compute_ee_pose(self, q):
        """
        Computes the end-effector pose for a batch of joint configurations.
        q: (B, 7) tensor
        Returns: (B, 7) tensor [x, y, z, qw, qx, qy, qz]
        """
        q_np = q.detach().cpu().numpy()
        batch_size = q_np.shape[0]
        ee_poses = []

        for i in range(batch_size):
            self.data.qpos[:7] = q_np[i]
            mujoco.mj_forward(self.model, self.data)
            
            pos = self.data.xpos[self.ee_id].copy()
            quat = self.data.xquat[self.ee_id].copy()
            
            ee_poses.append(np.concatenate([pos, quat]))
        
        return torch.tensor(np.array(ee_poses), dtype=torch.float32).to(self.device)

    def plot_arm_batch(self, q_batch, title=None, color='blue', y_sample=None):
        """
        Plots the arm configurations. 
        Since 3D plotting is complex and slow with matplotlib, we might just print info or do a simple 3D scatter of EE.
        For now, let's implement a simple 3D scatter plot of the End Effectors.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Compute EE positions
        ee_poses = self.compute_ee_pose(q_batch).cpu().numpy()
        
        ax.scatter(ee_poses[:, 0], ee_poses[:, 1], ee_poses[:, 2], c=color, marker='o', label='Predicted EE')
        
        if y_sample is not None:
             y_sample_np = y_sample.detach().cpu().numpy()
             if y_sample_np.ndim == 1:
                 y_sample_np = y_sample_np.reshape(1, -1)
             ax.scatter(y_sample_np[:, 0], y_sample_np[:, 1], y_sample_np[:, 2], c='red', marker='x', s=100, label='Target EE')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if title:
            ax.set_title(title)
        ax.legend()
        
        return fig
