import mujoco
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load the Panda model from Menagerie
model = mujoco.MjModel.from_xml_path('mujoco_menagerie/franka_emika_panda/panda.xml')
data = mujoco.MjData(model)

def generate_dataset(num_samples=10000, save_path="panda_ik_data.csv"):
    dataset = []
    
    # Identify joint and EE body IDs
    ee_body_name = "hand"
    try:
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
        if ee_id == -1:
            raise ValueError(f"Body '{ee_body_name}' not found.")
    except Exception:
        # Fallback to link7 if hand not found (e.g. if using panda_nohand.xml)
        ee_body_name = "link7"
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)

    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
    
    # Get joint limits from the model
    lower_limits = model.jnt_range[joint_ids, 0]
    upper_limits = model.jnt_range[joint_ids, 1]

    print(f"Generating {num_samples} valid samples using EE body: {ee_body_name}...")
    pbar = tqdm(total=num_samples)
    
    while len(dataset) < num_samples:
        # Randomly sample joint angles within limits
        q_rand = np.random.uniform(lower_limits, upper_limits)
        data.qpos[:7] = q_rand
        
        # Compute Forward Kinematics
        mujoco.mj_forward(model, data)
        
        # Check for self-collisions
        if data.ncon == 0:
            # Extract EE Position (x, y, z) and Orientation (Quat)
            ee_pos = data.xpos[ee_id].copy()
            ee_quat = data.xquat[ee_id].copy()
            
            # Save: [q1...q7, x, y, z, qw, qx, qy, qz]
            sample = np.concatenate([q_rand, ee_pos, ee_quat])
            dataset.append(sample)
            pbar.update(1)

    pbar.close()
    
    # Save to CSV
    cols = [f'q{i}' for i in range(1, 8)] + ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
    df = pd.DataFrame(dataset, columns=cols)
    df.to_csv(save_path, index=False)
    print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    generate_dataset(num_samples=1000)