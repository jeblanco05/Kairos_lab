# Kairos Lab

Kairos Lab is a robotics research project focused on whole-body coordinated control of a Kairos+ mobile manipulator using reinforcement learning. The project combines mobile navigation, manipulator control, and tray stability into a unified control problem, with the final goal of training policies that can operate robustly under realistic disturbances and payload variations.

## Overview

The repository is structured around two complementary parts:

- `train_lab/`: training environment and RL pipeline built with IsaacLab.
- `deploy_gz/`: deployment stack for running the trained policy in ROS 2 simulation and evaluating task performance.

Together, these modules allow the research workflow to progress from environment design and policy learning to deployment and validation.

## Main goals

- train a policy that tracks base velocity commands,
- coordinate the mobile base and UR5e arm together,
- maintain the end-effector or tray at a desired relative target position,
- keep the tray horizontal with respect to gravity,
- evaluate the resulting behavior in simulation before real-world transfer.

## Repository structure

```text
Kairos_lab/
├── README.md
├── deploy_gz/
│   └── ...
├── train_lab/
│   └── ...
├── ...
└── assets / generated resources
```

## Training workflow

The training workflow is centered on IsaacLab and uses reinforcement learning to optimize a policy for the robot. The system designs rewards around:

- base motion tracking,
- target tray placement,
- tray orientation stabilization,
- safe operation and collision avoidance.

The policy receives observations such as base velocity, arm state, tray pose, gravity alignment, and task commands, and outputs actions that coordinate the mobile platform and manipulator.

## Deployment workflow

Once a policy is trained, the deployment package loads the learned model and evaluates it inside a ROS 2 simulation environment. This stage is used to verify:

- inference behavior,
- tray stability in motion,
- comparison against fixed baseline arm poses,
- navigation and teleoperation integration.

## Typical development cycle

1. Prepare the robot asset and convert URDF to USD.
2. Define the environment, observations, and reward functions in the RL task.
3. Train a PPO policy in IsaacLab.
4. Save checkpoints and evaluate policy behavior in simulation.
5. Deploy the model with the ROS 2 stack and benchmark performance.

## Notes

This project is oriented toward research and experimentation, with a strong emphasis on sim-to-real transfer, robust manipulation, and coordinated whole-body control. It is intended to be extended with additional robot models, new tasks, and improved transfer strategies.
