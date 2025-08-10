import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class RobotArm2D_CP:
    def __init__(self, link_lengths, surface_radius=0.05):
        self.l = link_lengths  # torch.tensor([l1, l2, l3])
        self.surface_radius = surface_radius
    
    def compute_ee_pose_and_orientation(self, x):
        """
        x: (N, n) tensor: 
            - x[:, 0] = base y translation
            - x[:, 1:] = joint angles (theta1, theta2, ..., theta_(n-1))    -> in global frame!
        l: (n-1,) tensor: link lengths for each joint (l1, l2, ..., l_(n-1))

        Returns the (x, y) position and angle of the end-effector.
        """
        # Base translation along y-axis
        y_trans = x[:, 0]
        
        # Joint angles (thetas)
        thetas = x[:, 1:]
        
        # Link lengths (l1, l2, ..., l_(n-1))
        link_lengths = self.l
        
        # Initialize the end-effector position components (x, y)
        x_pos = torch.zeros(x.size(0)).to(device=link_lengths.device)  # x position of the end-effector
        y_pos = y_trans.clone()         # y position starts with base translation

        # Iterate through the links to compute the end-effector position
        for i in range(thetas.size(1)):
            x_pos += link_lengths[i] * torch.cos(thetas[:, i])  # Add x displacement from the link
            y_pos += link_lengths[i] * torch.sin(thetas[:, i])  # Add y displacement from the link
        
        # Stack the x and y positions into a tensor to represent the end-effector's position
        ee_pos = torch.stack([x_pos, y_pos], dim=1)
        ee_angle = thetas[:, -1]
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
        x = x.squeeze()  # Directly use PyTorch tensor operations
        
        # Extract base translation and joint angles
        y_trans = x[0]  # Base translation along y-axis
        thetas = x[1:]  # Joint angles (theta1, theta2, ..., theta_(n-1))
        
        # Link lengths
        link_lengths = self.l

        # Initialize the first joint at the base position
        joint_positions = [torch.tensor([0, y_trans])]
        
        # Calculate positions of joints and the end-effector
        for i in range(len(thetas)):
            x_pos = joint_positions[-1][0] + link_lengths[i] * torch.cos(thetas[i])
            y_pos = joint_positions[-1][1] + link_lengths[i] * torch.sin(thetas[i])
            joint_positions.append(torch.tensor([x_pos, y_pos]))
        return joint_positions

    def sample_contact_surface_global_batch(self, x_batch, n_samples_per_el=10):

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

    def plot_arm(self, x, l, title=None, color='blue', y_sample=None):
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

        if title is not None:
            ax.set_title(title)
        ax.set_xlim(-0.25, 3.5)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()

        return fig
    
    def plot_arm_batch(self, x_batch, title=None, color='blue', y_sample=None):
        """
        x_batch: tensor of shape (B, 4), where B is the number of joint configurations
        l: tensor of link lengths (3,)
        y_sample: target end-effector position to overlay as a green star, shape (1, 2) or (2,)
        """

        if isinstance(x_batch, torch.Tensor):
            x_batch = x_batch.cpu()

        if x_batch.ndim == 1:
            x_batch = x_batch.unsqueeze(0)

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_axes([0, 0, 1, 1]) 

        for i, x in enumerate(x_batch):
            joint_positions = self.compute_joint_positions(x)
            joint_positions = torch.stack(joint_positions, dim=0)  # (num_joints+1, 2)

            ee_pose = joint_positions[-1]
            ee_angle = x[-1]

            xs = joint_positions[:, 0].detach().numpy()
            ys = joint_positions[:, 1].detach().numpy()
            base_x, base_y = xs[0], ys[0]
            ee = joint_positions[-1].detach().numpy()

            if i != len(x_batch) - 1:
                alpha = 0.1
                ax.plot(xs, ys, '+-', linewidth=8, alpha=alpha, color=color)
            else:
                alpha = 1.0
                ax.plot(xs, ys, '+-', linewidth=8, alpha=alpha, color='green', label="Arm" if i == len(x_batch) - 1 else None)

            # Plot semicircle for end-effector surface
            if ee_pose is not None and ee_angle is not None:
                theta_full = torch.linspace(-torch.pi / 2, torch.pi / 2, 100)
                x_full = self.surface_radius * torch.cos(theta_full)
                y_full = self.surface_radius * torch.sin(theta_full)
                semicircle_local = torch.stack([x_full, y_full], dim=1)

                if isinstance(ee_angle, torch.Tensor):
                    ee_angle = ee_angle.item()
                R = torch.tensor([
                    [torch.cos(torch.tensor(ee_angle)), -torch.sin(torch.tensor(ee_angle))],
                    [torch.sin(torch.tensor(ee_angle)),  torch.cos(torch.tensor(ee_angle))]
                ])
                semicircle_global = (R @ semicircle_local.T).T + ee_pose
                semicircle_global = semicircle_global.detach().numpy()
                ax.plot(semicircle_global[:, 0], semicircle_global[:, 1], color='black', alpha=alpha, label="EE Surface" if i == 0 else None)

            # Plot base triangle only for first arm
            if i == len(x_batch) - 1:
                base_triangle = plt.Polygon([
                    (base_x - 0.1, base_y),
                    (base_x + 0.1, base_y),
                    (base_x, base_y - 0.15)
                ], closed=True, color='gray')
                ax.add_patch(base_triangle)

        # Plot target EE if provided
        if y_sample is not None:
            target = y_sample[0].cpu().numpy()
            ax.plot(target[0], target[1], 'ro', markersize=20, label="Target EE")

        if title is not None:
            ax.set_title(title)
        ax.set_xlim(-0.25, 2.0)
        # ax.set_ylim(-2, 2)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=25)  # Bigger tick labels
        ax.legend(fontsize=25)

        return fig


class RobotArm2D:
    def __init__(self, link_lengths):
        self.l = link_lengths  # torch.tensor([l1, l2, l3])

    def compute_ee_pose(self, x):
        """
        x: (N, n) tensor: 
            - x[:, 0] = base y translation
            - x[:, 1:] = joint angles (theta1, theta2, ..., theta_(n-1))    -> in global frame!
        l: (n-1,) tensor: link lengths for each joint (l1, l2, ..., l_(n-1))

        Returns the (x, y) position of the end-effector.
        """       
        # Joint angles (thetas)
        thetas = x[:, 1:]
        
        # Link lengths (l1, l2, ..., l_(n-1))
        link_lengths = self.l
        
        # Initialize the end-effector position components (x, y)
        x_trans = torch.zeros(x.size(0)).to(device=x.device)  # x position of the end-effector
        y_trans = x[:, 0]         # Base translation along y-axis

        # Iterate through the links to compute the end-effector position
        # for i in range(thetas.size(1)):
        #     x_pos += link_lengths[i] * torch.cos(thetas[:, i])  # Add x displacement from the link
        #     y_pos += link_lengths[i] * torch.sin(thetas[:, i])  # Add y displacement from the link
        x_pos = x_trans + (link_lengths * torch.cos(thetas)).sum(dim=1)
        y_pos = y_trans + (link_lengths * torch.sin(thetas)).sum(dim=1)

        # Stack the x and y positions into a tensor to represent the end-effector's position
        ee_pos = torch.stack([x_pos, y_pos], dim=1)
        return ee_pos


    def plot_arm(self, x, title=None, color='blue', y_sample=None):
        """
        x: joint configuration tensor of shape (N, n) 
            - x[:, 0] = base y translation
            - x[:, 1:] = joint angles (theta1, theta2, ..., theta_(n-1))
        l: tensor of link lengths (n-1,)
        y_sample: target end-effector position to overlay as a blue dot, shape (1, 2) or (2,)
        """
        x = x.squeeze().cpu().numpy()
        
        # Extract base translation and joint angles
        y_trans = x[0]  # Base translation along y-axis
        thetas = x[1:]  # Joint angles (theta1, theta2, ..., theta_(n-1))
        thetas = torch.tensor(thetas, dtype=torch.float32)  # Ensure thetas is a tensor
        
        # Link lengths
        link_lengths = self.l.cpu().numpy()

        # Initialize the first joint at the base position
        joint_positions = [(0, y_trans)]  # Starting with base (x=0, y=y_trans)
        
        # Calculate positions of joints and the end-effector
        for i in range(len(thetas)):
            x_pos = joint_positions[-1][0] + link_lengths[i] * torch.cos(thetas[i])
            y_pos = joint_positions[-1][1] + link_lengths[i] * torch.sin(thetas[i])
            joint_positions.append((x_pos, y_pos))
        
        # Extract the end-effector position
        ee = joint_positions[-1]

        # Plotting
        fig, ax = plt.subplots()
        
        # Extract joint and end-effector x and y positions
        xs, ys = zip(*joint_positions)
        
        # Plot arm
        ax.plot(xs, ys, 'o-', color=color, linewidth=3, label="Arm")
        
        # Plot base triangle (symbolizing the base of the robot)
        base_triangle = plt.Polygon([
            (-0.1, y_trans), 
            (0.1, y_trans), 
            (0, y_trans - 0.15)
        ], closed=True, color='gray')
        ax.add_patch(base_triangle)

        # Mark the End-Effector (EE)
        # ax.plot(ee[0], ee[1], 'ro', markersize=8, label="Predicted EE")
        
        # Plot y_sample if provided
        if y_sample is not None:
            y_sample = y_sample.squeeze().cpu().numpy()
            ax.plot(y_sample[0], y_sample[1], 'bo', markersize=12, label="Target EE")
        
        # Set plot title and limits
        if title is not None:
            ax.set_title(title)
        ax.set_xlim(-1, np.sum(link_lengths) + 1)  # Adjust limits based on the total length
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend(fontsize=18)

        return fig
    
    def plot_arm_batch(self, x_list, title=None, color='blue', y_sample=None):
        """
        x_list: list or tensor of joint configurations of shape (B, n) or list of (n,)
            - x[:, 0] = base y translation
            - x[:, 1:] = joint angles (theta1, theta2, ..., theta_(n-1))
        l: tensor of link lengths (n-1,)
        y_sample: target end-effector position to overlay as a blue dot, shape (1, 2) or (2,)
        """
        # TODO: CORRECT THE PLOT FOR MULTIPLE CONFIGURATIONS

        # Convert to list of NumPy arrays for uniformity
        if torch.is_tensor(x_list):
            x_list = x_list.cpu().numpy()
        x_list = np.atleast_2d(x_list)  # Ensure shape (B, n)

        link_lengths = self.l.cpu().numpy()

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_axes([0, 0, 1, 1]) 

        for i, x in enumerate(x_list):
            y_trans = x[0]
            thetas = x[1:]

            joint_positions = [(0, y_trans)]
            for j in range(len(thetas)):
                x_pos = joint_positions[-1][0] + link_lengths[j] * np.cos(thetas[j])
                y_pos = joint_positions[-1][1] + link_lengths[j] * np.sin(thetas[j])
                joint_positions.append((x_pos, y_pos))

            xs, ys = zip(*joint_positions)
            # alpha = 1.0 if i == 0 else 0.3

            # ax.plot(xs, ys, 'o-', color=color, linewidth=3, alpha=alpha, label=f"Arm {i+1}" if i == 0 else None)
            
            if i != len(x_list) - 1:
                alpha = 0.1
                ax.plot(xs, ys, '+-', linewidth=8, alpha=alpha, color=color)
            else:
                alpha = 1.0
                ax.plot(xs, ys, '+-', linewidth=8, alpha=alpha, color='blue', label="Arm" if i == len(x_list) - 1 else None)

            # Plot base triangle for the first arm only
            if i == len(x_list) - 1:
                base_triangle = plt.Polygon([
                    (-0.1, y_trans), 
                    (0.1, y_trans), 
                    (0, y_trans - 0.15)
                ], closed=True, color='gray')
                ax.add_patch(base_triangle)

                # Mark the End-Effector
                ee = joint_positions[-1]
                ax.plot(ee[0], ee[1], 'ro', markersize=20, label="Predicted EE")

        if y_sample is not None:
            target = y_sample[0].cpu().numpy()
            ax.plot(target[0], target[1], 'bo', markersize=20, label="Target EE")

        if title is not None:
            ax.set_title(title)
        ax.set_xlim(-0.25, 2.0)
        # ax.set_ylim(-2, 2)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=25)  # Bigger tick labels
        ax.legend(fontsize=25)

        return fig


if __name__ == '__main__':
    torch.manual_seed(0)
    sigma = torch.tensor([0.25, 0.5, 0.5, 0.5])
    x_batch = torch.randn(1, 4) * sigma  # (B, 4)

    # arm = RobotArm2D_CP(link_lengths=torch.tensor([0.5, 0.5, 1.0]))

    # outputs = arm.sample_contact_surface_global_batch(x_batch, n_samples_per_el=3)

    # # Plot only first sample in batch
    # global_pts = outputs['global_pts'][0]
    # x_batch = outputs['x_batch'][0]
    # ee_pos = outputs['ee_pos'][0]
    # ee_angle = outputs['ee_angle'][0]

    # joints = arm.compute_joint_positions(x_batch)

    # arm.plot(joints, global_pts, ee_pos, ee_angle)

    # fig = arm.plot_arm(x_batch, arm.l, title="Robot Arm with Contact Surface", y_sample=global_pts)
    # fig.savefig("robot_arm_plot.png")

    arm = RobotArm2D(link_lengths=torch.tensor([0.5, 0.5, 1.0]))

    ee_pos = arm.compute_ee_pose(x_batch)

    fig = arm.plot_arm(x_batch[0], title="Robot Arm", color='blue', y_sample=ee_pos[0])
    fig.savefig("robot_arm_plot.png")
