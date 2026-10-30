# Kairos Lab

Kairos Lab is a research project focused on coordinated whole-body control using reinforcement learning for a Kairos+ mobile manipulator equipped with a UR5e arm. The goal is to train a policy that controls both the mobile base and the manipulator at the same time while keeping the tray stable and tracking motion commands.

The platform is designed around a three-part objective:

1. base velocity tracking,
2. end-effector or tray positioning,
3. global tray orientation stabilization.

The project follows a sim-to-real philosophy, with the training loop implemented in IsaacLab and the deployment workflow validated in ROS 2 simulation.

## Problem description

The robot consists of a mobile base and a UR5e arm carrying a tray or payload at the end effector. The policy must coordinate base and arm motion to satisfy task commands while remaining stable and avoiding collisions or excessive tilt.

The main challenges are:

- tracking commanded linear and angular base velocities,
- reaching and maintaining a target tray position relative to the robot,
- keeping the tray horizontal with respect to gravity despite base rotations and uneven terrain,
- dealing with varying payload mass and dynamic disturbances.

This is a real-world-inspired scenario where whole-body control is crucial.

## Project structure

```text
Kairos_lab/train_lab/
├── assets/                     # USD robot assets and generated models
├── cfg/
│   └── robotcfg.py             # Robot articulation configuration and actuator setup
├── controllers/
│   └── local_base_action.py     # Custom local / base action controller
├── scripts/
│   ├── train.py                # PPO training entry point
│   ├── play.py                 # Policy evaluation and visualization
│   ├── script.py               # Quick debugging and manual checks
│   └── fix_usd.py              # USD repair utilities for wheels and collisions
├── slurms/                     # Slurm files for cluster training
├── tasks/
│   ├── kairosplus_env_cfg.py   # Main task configuration
│   ├── ppo_cfg.py              # PPO hyperparameters and trainer settings
│   └── mdp/                    # Observations, rewards, resets, terminations, curriculum
├── utils/                      # Common utilities and helper functions
├── README.md                   # Project overview and usage notes
└── ...
```

The project is organized around a modular RL pipeline:

- `cfg/`: robot articulation and actuator setup,
- `controllers/`: command conversion from policy output to valid base commands,
- `tasks/`: environment logic, MDP, rewards, resets, and curriculum,
- `scripts/`: training, evaluation, and debugging flow,
- `slurms/`: job files for batch execution,
- `utils/`: support code used by the training pipeline.

## Dependencies

- Isaac Sim 4.5.0
- Isaac Lab v2.3.2
- unitree_rl_lab 0.2.1
- rsl-rl v3.1.2
- PyTorch v2.5.1
- NumPy

## URDF to USD conversion

The robot asset is typically generated from a URDF and then converted to USD for use in IsaacLab.

1. Modify the `ur_type` argument in `rbkairos_plus.urdf.xacro` and set it to `"ur5e"`.
2. Export the URDF from the ROS package:

```bash
xacro rbkairos_plus.urdf.xacro > rbkairos_plus.urdf
```

> Note: it may be necessary to remove generated build folders such as `build/robotnik_description` and `install/robotnik_description`, then rebuild the package with `colcon build`.

3. Convert the URDF to USD using the IsaacLab tool:

```bash
IsaacLab/scripts/tools/convert_urdf.py /path/to/your/new/rbkairos_plus.urdf /path/to/destination/rbkairos_plus.usd
```

> Note: in some environments it is necessary to adjust the package search paths so the conversion script can find `ur_description` and `robotnik_sensors`.

4. Fix the resulting USD asset to repair wheel and collision issues:

```bash
Kairos_lab/scripts/fix_usd.py --headless
```

## Training scripts

### `train.py`

This script starts the training pipeline for the Kairos+ task. It configures the environment, launches the learner, stores logs under the chosen directory, and optionally records videos during training.

```bash
Kairos_lab/scripts/train.py \
  --task Kairos-PPO-v1 \
  --log_dir my_projects/Kairos_lab/logs \
  --headless \
  --video \
  --video_length 200 \
  --video_interval 1000 \
  --enable_cameras
```

Typical use cases:
- train a new PPO policy from scratch,
- monitor metrics and checkpoints,
- capture periodic visual samples for debugging.

### `play.py`

This script loads a saved policy and runs it in simulation for evaluation or demonstration.

```bash
Kairos_lab/scripts/play.py \
  --task Kairos-PPO-v1 \
  --load_run Kairos_lab/logs/Kairos-PPO-v1/2026-08-28_14-10-31 \
  --checkpoint -1 \
  --num_envs 25 \
  --headless \
  --video \
  --video_length 500 \
  --enable_cameras
```

Typical use cases:
- visualize the learned behavior,
- evaluate a checkpoint in a fixed number of environments,
- record videos of successful or failing trajectories.

### `script.py`

This is a quick debugging script intended for lightweight testing and environment inspection without launching a full RL training run.

```bash
Kairos_lab/scripts/script.py --headless --record_video
```

Typical use cases:
- quick sanity checks,
- manual robot motion tests,
- short diagnostic recordings.

## MDP and task design

The project uses a reinforcement learning MDP with task-specific rewards and observations.

### Rewards

- `action_rate_penalty`: discourages abrupt action changes and promotes smooth motion.
- `arm_joint_torques`: penalizes excessive torque usage in the arm.
- `arm_joint_acc`: penalizes aggressive arm acceleration.
- `dof_pos_limits`: penalizes motion close to joint limits.
- `joint_vel`: penalizes excessive joint velocity.
- `track_lin_vel_xy`: rewards accurate tracking of commanded forward and lateral base velocity.
- `track_ang_vel_z`: rewards correct angular tracking.
- `ee_position_tracking`: rewards accurate tray positioning.
- `ee_flat_orientation`: strongly penalizes tray tilt and keeps the payload level.
- `ee_stability`: penalizes high tray vibration or unstable motion.

### Observation space

The policy receives multiple observations, including:

- `base_ang_vel`: base angular velocity,
- `base_projected_gravity`: gravity direction in the base frame,
- `velocity_commands`: target base motion commands,
- `tray_pose_command`: target tray setpoint,
- `arm_joint_pos`: current arm joint positions,
- `arm_joint_vel`: current arm joint velocities,
- `ee_position_b` and `ee_orientation_b`: tray pose in the base frame,
- `ee_projected_gravity`: measured tray tilt,
- `last_action`: previous action for temporal continuity,
- `base_lin_vel`: base linear velocity,
- `ee_pose_b`: ideal tray pose estimate,
- `tray_payload_mass`: mass of the payload in the current episode,
- `base_mass_offset`: base mass variation,
- `ee_velocity`: tray linear and angular velocity.

### Termination conditions

The environment can terminate when:

- the episode time limit is reached,
- the arm contacts the floor, object, or itself,
- the robot reaches an unsafe orientation,
- the tray exceeds a critical tilt threshold, roughly 35 degrees.

## Main task

- `Kairos-PPO-v1`

This task is the main RL benchmark used for training and evaluation in the current project pipeline.

## Research directions

This project is part of a broader robotics research effort exploring transfer learning and whole-body control. Some relevant directions include:

- transferring a policy from one robot to another,
- using a pretrained locomotion policy to bootstrap training,
- preparing a dexterous robot setup with a camera,
- replicating behavior in more advanced mobile manipulation scenarios.
