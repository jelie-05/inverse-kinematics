import mujoco
import mujoco.viewer
import time

# 1. Load the model from the Menagerie folder
model = mujoco.MjModel.from_xml_path('mujoco_menagerie/franka_emika_panda/panda.xml')
data = mujoco.MjData(model)

# 2. Launch the passive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 3. Simulation loop
    while viewer.is_running():
        step_start = time.time()

        # Apply physics step
        mujoco.mj_step(model, data)

        # Sync the viewer with the new physics state
        viewer.sync()

        # Maintain real-time frequency (approx 500Hz-1000Hz)
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)