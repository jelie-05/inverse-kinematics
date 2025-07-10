import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class RobotArm2D:
    def __init__(self, link_lengths, surface_radius=0.05):
        self.l = link_lengths  # torch.tensor([l1, l2, l3])
        self.surface_radius = surface_radius

    def compute_ee_pose_and_orientation(self, x):
        """
        x: (B, 4) tensor
        Returns:
          ee_pos: (B, 2)
          ee_angle: (B,)
        """
        x_trans = x[:, 0]
        theta1, theta2, theta3 = x[:, 1], x[:, 2], x[:, 3]
        l1, l2, l3 = self.l

        y1 = l1 * torch.cos(theta1) + l2 * torch.cos(theta2) + l3 * torch.cos(theta3)
        y2 = x_trans + l1 * torch.sin(theta1) + l2 * torch.sin(theta2) + l3 * torch.sin(theta3)
        ee_pos = torch.stack([y1, y2], dim=1)
        ee_angle = theta3
        return ee_pos, ee_angle

    def sample_surface_local(self, n_samples=10):
        """
        Returns: (n_samples, 2)
        """
        theta_full = torch.linspace(-torch.pi / 2, torch.pi / 2, 100)
        x_full = self.surface_radius * torch.cos(theta_full)
        y_full = self.surface_radius * torch.sin(theta_full)
        points_full = torch.stack([x_full, y_full], dim=1)
        indices = torch.randperm(100)[:n_samples]
        sampled_points = points_full[indices]
        return sampled_points  # (n_samples, 2)

    def transform_to_global_batch(self, local_points, ee_pos_batch, ee_angle_batch):
        """
        local_points: (P, 2)
        ee_pos_batch: (B, 2)
        ee_angle_batch: (B,)
        Returns: (B, P, 2)
        """
        B = ee_pos_batch.shape[0]
        P = local_points.shape[0]

        cos = torch.cos(ee_angle_batch).unsqueeze(1)  # (B, 1)
        sin = torch.sin(ee_angle_batch).unsqueeze(1)  # (B, 1)

        R = torch.zeros((ee_angle_batch.shape[0], 2, 2), dtype=ee_angle_batch.dtype)

        R[:, 0, 0] = torch.cos(ee_angle_batch)
        R[:, 0, 1] = -torch.sin(ee_angle_batch)
        R[:, 1, 0] = torch.sin(ee_angle_batch)
        R[:, 1, 1] = torch.cos(ee_angle_batch)

        local_expanded = local_points.unsqueeze(0).expand(B, -1, -1)  # (B, P, 2)
        rotated = torch.bmm(local_expanded, R).to(ee_pos_batch.device)  # (B, P, 2)
        translated = rotated + ee_pos_batch.unsqueeze(1)  # (B, P, 2)
        return translated

    def compute_joint_positions(self, x):
        """
        x: (4,)
        Returns: list of 4 joint positions
        """
        x_trans, theta1, theta2, theta3 = x[0], x[1], x[2], x[3]
        l1, l2, l3 = self.l

        base = torch.tensor([0.0, x_trans])
        joint1 = base + torch.tensor([l1 * torch.cos(theta1), l1 * torch.sin(theta1)])
        joint2 = joint1 + torch.tensor([l2 * torch.cos(theta2), l2 * torch.sin(theta2)])
        ee = joint2 + torch.tensor([l3 * torch.cos(theta3), l3 * torch.sin(theta3)])
        return [base, joint1, joint2, ee]

    def sample_contact_surface_global_batch(self, x_batch, n_samples_per_el=10):
        """
        x_batch: (B, 4)
        n_samples_per_el: number of contact surface samples per element in the batch
        Returns:
            global_pts: (B, n_samples, 2)
            ee_pos: (B, 2)
            ee_angle: (B,)
        """
        ee_pos, ee_angle = self.compute_ee_pose_and_orientation(x_batch)  # (B, 2), (B,)
        local_pts = self.sample_surface_local(n_samples_per_el)  # (P, 2)
        global_pts = self.transform_to_global_batch(local_pts, ee_pos, ee_angle)  # (B, P, 2)

        # flatten to (B * n_samples, 2)
        global_pts_flat = global_pts.view(-1, 2)
        ee_pos_ = ee_pos.repeat_interleave(n_samples_per_el, dim=0)  # (B * n_samples, 2)
        ee_angle_ = ee_angle.repeat_interleave(n_samples_per_el)  # (B * n_samples,)

        # rearrange x_batch to match the global_pts shape
        x_batch = x_batch.repeat_interleave(n_samples_per_el, dim=0)

        outputs = {
            'global_pts': global_pts_flat,
            'x_batch': x_batch,
            'ee_pos': ee_pos_,
            'ee_angle': ee_angle_,
        }

        return outputs

    def plot(self, joint_positions, surface_points, ee_pos, ee_angle):
        joints = torch.stack(joint_positions, dim=0)

        plt.figure(figsize=(6, 6))
        plt.plot(joints[:, 0], joints[:, 1], '-+', label='Robot Arm', color='blue')
        plt.scatter(surface_points[0], surface_points[1], color='red', marker='x', label='Sampled Contact Points')

        theta_full = torch.linspace(-torch.pi / 2, torch.pi / 2, 100)
        x_full = self.surface_radius * torch.cos(theta_full)
        y_full = self.surface_radius * torch.sin(theta_full)
        semicircle_local = torch.stack([x_full, y_full], dim=1)
        R = torch.tensor([
            [torch.cos(ee_angle), -torch.sin(ee_angle)],
            [torch.sin(ee_angle),  torch.cos(ee_angle)]
        ])
        semicircle_global = (R @ semicircle_local.T).T + ee_pos
        plt.plot(semicircle_global[:, 0], semicircle_global[:, 1], color='black', label='EE Surface (Semicircle)')

        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.title('Robot Arm with Contact Surface')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(-0.25, 4)
        plt.ylim(-2, 2)
        plt.savefig("robot_arm_with_surface.png")
        print("Plot saved to 'robot_arm_with_surface.png'")
        plt.close()

    def plot_arm(self, x, l, title="Robot Arm", color='blue', y_sample=None):
        """
        x: joint configuration tensor of shape (1, 4)
        l: tensor of link lengths (3,)
        y_sample: target end-effector position to overlay as a green star, shape (1, 2) or (2,)
        ee_pose: tensor of shape (2,), global EE position for surface plot
        ee_angle: float or tensor scalar, EE orientation in radians
        """
        joint_positions = self.compute_joint_positions(x)
        joint_positions = torch.stack(joint_positions, dim=0)  # (num_joints+1, 2)

        ee_pose = joint_positions[-1] 
        ee_angle = x[-1]

        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot arm
        xs = joint_positions[:, 0].detach().cpu().numpy()
        ys = joint_positions[:, 1].detach().cpu().numpy()
        base_x, base_y = xs[0], ys[0]
        ee = joint_positions[-1].detach().cpu().numpy()
        ax.plot(xs, ys, '+-', color=color, linewidth=3, label="Arm")

        # Plot semicircle for end-effector surface if data is available
        if ee_pose is not None and ee_angle is not None:
            theta_full = torch.linspace(-torch.pi / 2, torch.pi / 2, 100)
            x_full = self.surface_radius * torch.cos(theta_full)
            y_full = self.surface_radius * torch.sin(theta_full)
            semicircle_local = torch.stack([x_full, y_full], dim=1)  # (100, 2)

            # Ensure ee_angle is a float
            if isinstance(ee_angle, torch.Tensor):
                ee_angle = ee_angle.item()
            R = torch.tensor([
                [torch.cos(torch.tensor(ee_angle)), -torch.sin(torch.tensor(ee_angle))],
                [torch.sin(torch.tensor(ee_angle)),  torch.cos(torch.tensor(ee_angle))]
            ])
            semicircle_global = (R @ semicircle_local.T).T + ee_pose  # (100, 2)
            semicircle_global = semicircle_global.detach().cpu().numpy()
            ax.plot(semicircle_global[:, 0], semicircle_global[:, 1], color='black', label='EE Surface (Semicircle)')

        # Plot base triangle
        base_triangle = plt.Polygon([
            (base_x - 0.1, base_y),
            (base_x + 0.1, base_y),
            (base_x, base_y - 0.15)
        ], closed=True, color='gray')
        ax.add_patch(base_triangle)

        # Mark EE
        # ax.plot(ee[0], ee[1], 'ro', markersize=8, label="Predicted EE")

        # Plot target EE if given
        if y_sample is not None:
            y_sample = y_sample.squeeze()
            if isinstance(y_sample, torch.Tensor):
                y_sample = y_sample.detach().cpu().numpy()
            ax.plot(y_sample[0], y_sample[1], 'rx', markersize=10, label="Target EE")

        ax.set_title(title)
        ax.set_xlim(-0.25, 3.5)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()

        return fig



if __name__ == '__main__':
    torch.manual_seed(0)
    sigma = torch.tensor([0.25, 0.5, 0.5, 0.5])
    x_batch = torch.randn(2, 4) * sigma  # (B, 4)

    arm = RobotArm2D(link_lengths=torch.tensor([0.5, 0.5, 1.0]))

    outputs = arm.sample_contact_surface_global_batch(x_batch, n_samples_per_el=3)

    # Plot only first sample in batch
    global_pts = outputs['global_pts'][0]
    x_batch = outputs['x_batch'][0]
    ee_pos = outputs['ee_pos'][0]
    ee_angle = outputs['ee_angle'][0]

    joints = arm.compute_joint_positions(x_batch)

    arm.plot(joints, global_pts, ee_pos, ee_angle)

    fig = arm.plot_arm(x_batch, arm.l, title="Robot Arm with Contact Surface", y_sample=global_pts)
    fig.savefig("robot_arm_plot.png")
