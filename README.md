# Team 5 - Invertible Neural Networks for Inverse Kinematics

This project implements Invertible Neural Networks (INNs) using the RealNVP framework to solve the inverse kinematics problem for a simulated 4-DoF planar robotic manipulator. The framework supports multiple end-effector geometries and includes evaluation tools for accuracy and solution diversity.

## Usage

### Training
```bash
python3 -m src.train --config <config-yaml-file>
```

### Inference and Evaluation
```bash
python3 -m src.inference_eval --config <config-yaml-file>
```
