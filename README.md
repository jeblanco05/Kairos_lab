# Kairos Lab

Coordinated whole-body control (Whole-Body Control) using Reinforcement Learning for navigation and stabilization of a Kairos+ mobile manipulator with UR5e in IsaacLab.

**Problem Description:**
The goal of the project is to train a Reinforcement Learning (RL) policy for coordinated sim-to-real control of a mobile manipulator composed of a Kairos+ base and a UR5e robotic arm. The policy actuates the mobile base and the arm joints simultaneously to accomplish three concurrent tasks:

1. Base velocity command tracking: The policy must move the mobile base following linear and angular velocity commands introduced by the simulator, which change dynamically every $X$ seconds throughout the episode.

2. Relative positioning of the end effector: The end effector (acting as support/tray) must reach and maintain a given reference position relative to the robot, whose target is also updated periodically during the episode.

3. Global orientation stabilization (Tray balance): The end effector must continuously maintain a strictly horizontal orientation with respect to the global gravity reference frame. This implies canceling deviations in roll, pitch, and yaw angles caused by base rotations, accelerations, or transitions over ramps and uneven terrain.

**Boundary Conditions and Dynamics:**
- Simulation Environment: Training and the initial validation are carried out entirely on the IsaacLab platform.

- End effector with variable mass: To promote agent robustness to payload changes, the mass attached to the end effector will vary dynamically between episodes or training phases.

- Control without active gripper: The manipulation or closure of the gripper fingers is not included in the action space; the load is assumed fixed, and the required measurements are extracted directly from the end effector pose and sensors.

## Project Structure

```text
Kairos_lab/
├── assets/                    # USD robot assets and generated models
├── cfg/
│   └── robotcfg.py            # Robot articulation configuration and actuator setup
├── controllers/
│   └── local_base_action.py   # Custom local / mecanum base action controllers
├── logs/                      # Training logs
├── scripts/
│   ├── train.py               # Training entry point for PPO / RL runs
│   ├── play.py                # Policy evaluation and visualization in simulation
│   ├── script.py              # Quick debugging or manual inspection script
│   └── fix_usd.py             # USD correction utilities for wheel and collision cleanup
├── slurms/                    # SLURM job scripts for cluster execution
├── tasks/
│   ├── kairosplus_env_cfg.py  # Main environment configuration
│   ├── ppo_cfg.py             # PPO algorithm configuration
│   └── mdp/                   # Observation, reward, termination, curriculum and event logic 
├── utils/                     # Utility helpers and shared support functions
└── README.md                  # Project overview and usage notes
```

The project is organized around a modular RL pipeline:

- `cfg/` defines robot and articulation setup.
- `controllers/` contains custom action abstractions used to translate policy outputs into valid low-level chassis or wheel commands.
- `tasks/` contains the IsaacLab environment, MDP logic, reward functions, resets, and curricula.
- `scripts/` provides the training and evaluation workflows used to run experiments and inspect behavior.
- `slurms/` stores execution jobs for cluster or server-based runs.

## urdf.xacro -> usd conversion

1. In `rbkairos_plus.urdf.xacro`, modify the `ur_type` argument and set it to `"ur5e"`.

2. Convert to `.urdf` from the ROS package:

``` bash
    xacro rbkairos_plus.urdf.xacro > rbkairos_plus.urdf
```
> Note: It is necessary to delete the build files (`build/robotnik_description`, `install/robotnik_description`) and rebuild the ROS package (`colcon build`).

3. Convert to USD using IsaacLab's conversion script:

``` bash
IsaacLab/scripts/tools/convert_urdf.py /path/to/your/new/rbkairos_plus.urdf /path/to/destination/rbkairos_plus.usd
```
> Note: It may be necessary to adjust paths to find the `ur_description` and `robotnik_sensors` packages.

4. Fix the wheel and arm collision meshes in the USD:

```bash
Kairos_lab/scripts/fix_usd.py --headless
```

## Tasks

- Kairos-PPO-v1

## Scripts

The scripts folder contains the main execution entry points for training, inference, and quick debugging.

### `train.py`
This script starts the training pipeline for the Kairos+ environment. It configures the task, launches the learner, stores logs under the chosen directory, and can optionally enable video capture during training.

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
- Train a new PPO policy from scratch.
- Monitor training metrics and save checkpoints.
- Capture periodic video samples for qualitative inspection.

### `play.py`
This script loads a trained policy and runs it in simulation for evaluation or demonstration. It is useful to verify that the learned behavior transfers to the environment and to export the final policy in a deployable format if required by the training stack.

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
- Visualize the learned behavior in simulation.
- Evaluate a checkpoint with a fixed number of environments.
- Record videos of successful or failing controller behavior.

### `script.py`
This is a lightweight script intended for quick visualization and debugging tests. It is especially useful when you want to verify the environment or a specific controller behavior without starting a full RL training process.

```bash
    Kairos_lab/scripts/script.py --headless --record_video
```

Typical use cases:
- Quick sanity checks of the environment.
- Manual robot motion tests.
- Fast recording of a short simulation sequence for diagnosis.

## MDP

**Rewards**
- `action_rate_penalty`: Penalizes abrupt action changes to prevent the robot from jerking.

- `arm_joint_torques`: Penalizes excessive force usage in the arm motors to save energy and reduce wear.

- `arm_joint_acc`: Penalizes high arm accelerations, forcing smoother movements.

- `dof_pos_limits`: Penalizes the arm when its motors approach their maximum or minimum mechanical limits.

- `joint_vel`: Penalizes high motor velocities. Helps avoid poses where the robot becomes blocked (singularities).

- `track_lin_vel_xy`: Rewards the mobile base for moving exactly at the commanded forward/lateral velocity.

- `track_ang_vel_z`: Rewards the base for turning at the correct speed.

- `ee_position_tracking`: Awards points when the arm places the tray (end effector) at the exact commanded position.

- `ee_flat_orientation`: Severely penalizes any tray tilt. This is the key to learning to keep it always flat/horizontal.

- `ee_stability`: Penalizes high velocities or tray vibrations so the payload transport remains stable.

**Observations**

- `base_ang_vel`: How fast the mobile base is rotating about itself.

- `base_projected_gravity`: Detects whether the base is on a slope or uneven terrain (measured by the IMU).

- `velocity_commands`: The command specifying how fast the base should move.

- `tray_pose_command`: The command specifying where the tray should move.

- `arm_joint_pos`: The current angle of each arm motor.

- `arm_joint_vel`: How fast each arm motor is rotating at this instant.

- `ee_position_b` & `ee_orientation_b`: The current tray coordinates and rotation relative to the robot base.

- `ee_projected_gravity`: The actual tray tilt. Essential for the robot to know if the payload is starting to fall.

- `last_action`: Memory of what the robot decided a moment earlier, helping it compute continuous motions.

- `base_lin_vel`: The true and exact linear velocity of the base in the world.

- `ee_pose_b`: The perfect tray position and inclination, without the error that real sensors would have.

- `tray_payload_mass`: The exact payload mass placed on the tray in that episode.

- `base_mass_offset`: Random variations in robot chassis weight.

- `ee_velocity`: The exact linear and angular velocity at which the tray is moving in 3D space.

**Terminations**

- `timeout`:

- `Arm impact`: Terminates if the UR5e arm hits the floor, pyramids, or itself.

- `Bad orientation`: Terminates if the robot tips over beyond a critical angle even before the base hits the ground.

- `Tray overturned`: Terminates if the tray tilts more than ~35 degrees relative to horizontal.

## Dependencies
- Isaac Sim 4.5.0 
- Isaac Lab v2.3.2
- unitree_rl_lab 0.2.1
- rsl-rl v3.1.2
- Pytorch v2.5.1
- Numpy