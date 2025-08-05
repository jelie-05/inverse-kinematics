import math
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class FKIKDataset(Dataset):
    
    def __init__(self, forward_kinematics=None):
        if forward_kinematics:
            self.fw = forward_kinematics
        else:
            self.fw = self.default_forward_kinematics
        
    def default_forward_kinematics(self, x: torch.Tensor):
        # x has shape (B, 4)
        # y has shape (B, 2)
        l1, l2, l3 = 0.5, 0.5, 1.0

        B = x.shape[0]

        x_joint1 = torch.zeros(B)
        y_joint1 = x[:, 0]

        x_joint2 = x_joint1 + l1 * torch.cos(x[:, 1])
        y_joint2 = y_joint1 + l1 * torch.sin(x[:, 1])

        x_joint3 = x_joint2 + l2 * torch.cos(x[:, 2])
        y_joint3 = y_joint2 + l2 * torch.sin(x[:, 2])

        x_joint4 = x_joint3 + l3 * torch.cos(x[:, 3])
        y_joint4 = y_joint3 + l3 * torch.sin(x[:, 3])

        joint1 = torch.stack((x_joint1, y_joint1), dim=1)
        joint2 = torch.stack((x_joint2, y_joint2), dim=1)
        joint3 = torch.stack((x_joint3, y_joint3), dim=1)
        joint4 = torch.stack((x_joint4, y_joint4), dim=1)

        ee_pos = torch.stack((x_joint4, y_joint4), dim=1)

        return [joint1, joint2, joint3, joint4]
    
    def compute_ee_pos(self, x: torch.Tensor):
        joints = self.fw(x)
        return joints[-1]

    def test_compute_ee_pos(self):
        x = torch.Tensor([[2, 0, 0, torch.pi/4], [3, 0, torch.pi/4, torch.pi/4]])
        ee_pos = self.compute_ee_pos(x)
        
        exp_ee_pos = torch.Tensor([[1+math.cos(math.pi/4), 2+math.sin(math.pi/4)], [0.5 + 1.5 * math.cos(math.pi/4), 3 + 1.5 * math.sin(math.pi/4)]])
        if torch.allclose(ee_pos, exp_ee_pos):
            good = True
        else:
            good = False

        return x, ee_pos, exp_ee_pos, good
  
    def create_filtered_dataset(self, n_samples, save_path=None):
        sigma = torch.tensor([0.25, 0.5, 0.5, 0.5])
        self.joint_config = None
        self.ee_pos = None

        while n_samples > 0:
            sample = torch.randn(1, 4) * sigma

            if self.self_collision(sample) == True:
                # There is self_collision. Do nothing
                pass
            else:
                if self.joint_config == None:
                    self.joint_config = sample
                else:
                    self.joint_config = torch.cat([self.joint_config, sample], dim=0)

                n_samples -= 1

        self.ee_pos = self.compute_ee_pos(self.joint_config)

        data = {'joint_config': self.joint_config, 'ee_pos': self.ee_pos}
        if save_path:
            torch.save(data, save_path)
        
        return self
    
    def self_collision(self, x: torch.Tensor):
        # x of shape (1, DoF)
        joints = self.fw(x)
        nr_joints = len(joints)

        for i in range(nr_joints-2):
            joint_i_1 = joints[i].squeeze(0)
            joint_i_2 = joints[i+1].squeeze(0)

            for j in range(i+1, nr_joints-1):
                joint_j_1 = joints[j].squeeze(0)
                joint_j_2 = joints[j+1].squeeze(0)

                # Initialize info list if it doesn't exist
                if 'info' not in locals():
                    info = []
                info.append(f"Links {i}, {j} ")

                # Check if intersection
                if self.overlap(joint_i_1, joint_i_2, joint_j_1, joint_j_2):
                    info[-1] += "/ overlap "
                    if torch.all(joint_i_2 == joint_j_1):
                        # Case when the checked links are connected through an end point
                        # Check if the three points are collinear using the cross product
                        info[-1] += "/ common joint"
                        vec1 = joint_i_2 - joint_i_1
                        vec2 = joint_j_2 - joint_j_1
                        cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
                        if abs(cross) < 1e-9:
                            info[-1] += "/ collinear"
                            # Collinear, check if the segments overlap
                            # Use projection to check overlap
                            def on_segment(p, q, r):
                                return (min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9 and
                                        min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9)
                            if (on_segment(joint_i_1, joint_j_2, joint_i_2) or
                                on_segment(joint_j_1, joint_i_1, joint_j_2)):
                                # Collision, overlap of the segments
                                info[-1] += "/ same dir"
                                return True, info
                            else:
                                # Collinear, but not overlapping
                                pass
                                info[-1] += "/ diff dir"
                        else:
                            # Not collinear
                            pass
                            info[-1] += " / not collinear"
                    else:
                        # Case when the checked links are not connected
                        info[-1] += "/ no common joint"
                        return True, info
                else:
                    info[-1] += 'no_overlap'
        return False, info
    
    def test_self_collision(self, x, expRes):
        res, info = self.self_collision(x)
        if res == expRes:
            print("Test passed")
        else:
            print("Test failed")
        
        print(info)
        self.plot_config(x)

    def create_unfiltered_dataset(self, n_samples, save_path=None):
        sigma = torch.tensor([0.25, 0.5, 0.5, 0.5])
        x = torch.randn(n_samples, 4) * sigma  # x: (N, 4)
        ee_pos = self.compute_ee_pos(x)

        data = {'joint_config': x, 'ee_pos': ee_pos}
        if save_path:
            torch.save(data, save_path)
        
        self.joint_config = x
        self.ee_pos = ee_pos
        
        return self
    
    def __len__(self):
        return len(self.joint_config)

    def __getitem__(self, idx):
        return {
            "joint_config": self.joint_config[idx],
            "ee_pos": self.ee_pos[idx]
        }
        
    def overlap(self, joint_i_1, joint_i_2, joint_j_1, joint_j_2):
        def orientation(p, q, r):
            """0 = colinear, 1 = clockwise, 2 = counterclockwise"""
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < 1e-9:
                return 0
            return 1 if val > 0 else 2

        def on_segment(p, q, r):
            """Check if point q lies on segment pr"""
            return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                    min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

        A = joint_i_1
        B = joint_i_2
        C = joint_j_1
        D = joint_j_2

        o1 = orientation(A, B, C)
        o2 = orientation(A, B, D)
        o3 = orientation(C, D, A)
        o4 = orientation(C, D, B)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special cases – colinear and overlapping
        if o1 == 0 and on_segment(A, C, B):
            return True
        if o2 == 0 and on_segment(A, D, B):
            return True
        if o3 == 0 and on_segment(C, A, D):
            return True
        if o4 == 0 and on_segment(C, B, D):
            return True

        return False

    def test_overlap(self, joint_1, joint_2, joint_3, joint_4, expRes):
        if self.overlap(joint_1, joint_2, joint_3, joint_4) == expRes:
            print("Test passed")
        else:
            print("Test failed")

    def plot_dataset(self):
        ax = self.plot_config(self.joint_config, ax=None, show=True, label="Dataset")
        return ax

    def plot_config(self, x, ax=None, show=True, color='b', label=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        for i in range(len(x)):
            joints = self.fw(x[i].unsqueeze(0))
            xs = [joint[:, 0] for joint in joints]
            ys = [joint[:, 1] for joint in joints]
            ax.plot(xs, ys, marker='o')
            ax.plot(xs, ys, 'k-')
            # Plot a small triangle at the first joint
            triangle_size = 0.05
            x0, y0 = xs[0].item(), ys[0].item()
            triangle = plt.Polygon([
                (x0, y0 + triangle_size),
                (x0 - triangle_size * 0.866, y0 - triangle_size / 2),
                (x0 + triangle_size * 0.866, y0 - triangle_size / 2)
            ], color='red')
            ax.add_patch(triangle)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Robot Arm Configuration')
        ax.axis('equal')
        ax.grid(True)
        if label is not None:
            ax.legend()
        if show:
            plt.show()
        return ax
        

if __name__ == "__main__":
    dataset = FKIKDataset()
    
    print(dataset.test_compute_ee_pos())

    # Testing function overlap
    print("Testing overlap method")
    joint_1 = torch.Tensor([0, 0])
    joint_2 = torch.Tensor([2, 0])
    joint_3 = torch.Tensor([2, 2])
    joint_4 = torch.Tensor([0, 2])

    dataset.test_overlap(joint_1, joint_3, joint_2, joint_4, expRes=True)
    dataset.test_overlap(joint_1, joint_2, joint_3, joint_4, expRes=False)
    dataset.test_overlap(joint_1, joint_3, joint_3, joint_2, expRes=True)
    dataset.test_overlap(joint_1, joint_2, joint_2, joint_4, expRes=True)

    # Generating a dataset and plotting it
    print("Testing plot_config method")
    dataset.create_unfiltered_dataset(n_samples=4)
    joint_config1 = dataset.joint_config # (4, 4)
    dataset.create_unfiltered_dataset(n_samples=2)
    joint_config2 = dataset.joint_config # (2, 4)

    # ax1 = dataset.plot_config(joint_config1, ax=None, show=False)
    # ax2 = dataset.plot_config(joint_config2, ax=ax1, show=True)

    print("Testing plot_dataset method")
    dataset.create_unfiltered_dataset(n_samples=4)
    # dataset.plot_dataset()
    

    # Testing self_collision method
    print("Testing self_collision method")
    x = torch.Tensor([0, 0, math.pi/2, math.pi*1.4]).unsqueeze(0)
    dataset.test_self_collision(x, True)

    x = torch.Tensor([0, 0, math.pi/2, math.pi]).unsqueeze(0)
    dataset.test_self_collision(x, False)

    x = torch.Tensor([0, 0, 0, -math.pi]).unsqueeze(0)
    dataset.test_self_collision(x, True)

    x = torch.Tensor([0, 0, -math.pi/2, math.pi/2]).unsqueeze(0)
    dataset.test_self_collision(x, True)

    x = torch.Tensor([0, 0, -math.pi/2, math.pi/2 * 0.9]).unsqueeze(0)
    dataset.test_self_collision(x, False)

    x = torch.Tensor([0, 0, -math.pi, math.pi/2]).unsqueeze(0)
    dataset.test_self_collision(x, True)

    # Testing create_filtered_dataset
    dataset.create_filtered_dataset(10)
    dataset.plot_dataset()