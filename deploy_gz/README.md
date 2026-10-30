# Kairos Deploy

This package provides the deployment layer for the Kairos mobile manipulator used to run the trained policies in simulation and evaluate tray stabilization and navigation performance in a ROS 2 environment.

The deployment stack is responsible for:
- launching the robot and simulation world,
- loading and running the trained policy,
- receiving sensor and task observations,
- publishing control commands to the robot,
- evaluating tray orientation and balance metrics.

## Project purpose

The main objective of this package is to validate in simulation that the control policy learned during training can be executed in a real robot-like environment. The package focuses on a practical scenario in which the mobile base must move while the arm and tray remain stable and aligned with the commanded pose.

The most important objectives are:
- mobile base tracking,
- tray end-effector positioning,
- tray horizontal stabilization under motion and terrain disturbances,
- comparison between policy-based control and a fixed-arm baseline.

## Package structure

```text
Kairos_lab/deploy_gz/
├── CMakeLists.txt
├── package.xml
├── README.md
├── config/
│   └── rl_params.yaml
├── launch/
│   ├── full_simulation.launch.py
│   ├── nav2_full.launch.py
│   └── ...
├── models/
│   └── policy.onnx
├── observations/
│   ├── robot_observations.cpp
│   └── robot_observations.hpp
├── scripts/
│   ├── plot_tilt_comparison.py
│   ├── spawn_cube_on_tray.py
│   ├── teleop.py
│   └── twist_relay.py
├── src/
│   ├── rl_inference_node.cpp
│   └── tray_tilt_eval_node.cpp
├── worlds/
│   └── ...
├── Docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── setup.sh
└── ...
```

## Main nodes and components

- `rl_inference_node.cpp`: runs the trained policy in real time and publishes actions.
- `tray_tilt_eval_node.cpp`: measures tray motion, tilt, and stability.
- `full_simulation.launch.py`: starts the robot and simulation scene.
- `nav2_full.launch.py`: launches navigation components and planner stack.
- `config/rl_params.yaml`: model and execution parameters.
- `models/`: contains the ONNX or inference model used at runtime.
- `scripts/`: utilities for teleop, object spawning, analysis, and visualization.

## Requirements

Before building the project, ensure the following are available:

- ROS 2 installed and configured,
- `colcon` available,
- Docker and Docker Compose installed,
- the `robotnik_simulation` repository cloned locally,
- the `Kairos_lab` repository cloned locally.

## Build and setup

Clone the repositories:

```bash
git clone https://github.com/jeblanco05/Kairos_lab.git
git clone https://github.com/RobotnikAutomation/robotnik_simulation.git
```

Start the simulator container:

```bash
cd Kairos_lab/deploy_gz/Docker
docker compose up -d --build --force-recreate simulation
```

Open a shell inside the container:

```bash
docker exec -it -u robot kairos_sim_jazzy bash
```

Build the workspace:

```bash
cd /home/robot/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Launch workflow

Start the complete simulation environment:

```bash
ros2 launch kairos_deploy full_simulation.launch.py launch_nav2:=true spawn_cube:=true
```

### Terminal 1: policy inference

```bash
ros2 run kairos_deploy rl_inference_node --ros-args -p model_name:="ppo"
```

This node loads the inference model and applies the learned control policy to the robot.

### Terminal 2: tray evaluation

Run the evaluator in policy mode:

```bash
ros2 run kairos_deploy tray_tilt_eval_node --ros-args -p run_label:=policy -p test_num:=0
```

Run the evaluator with a fixed arm configuration for comparison:

```bash
ros2 run kairos_deploy tray_tilt_eval_node --ros-args \
  -p run_label:=arm_fixed -p hold_arm_fixed:=true \
  -p fixed_arm_joint_positions:="[0.0, -1.57, 1.57, 0.0, 1.57, 0.0]" \
  -p test_num:=0
```

This makes it possible to compare:
- learned, policy-controlled stabilization,
- a fixed baseline controller.

## Runtime commands

Set a target tray offset:

```bash
ros2 topic pub /kairos_deploy/tray_offset_target geometry_msgs/msg/Vector3 "{x: 0.1, y: 0.0, z: 0.05}" -r 10
```

Publish navigation commands:

```bash
ros2 topic pub -r 10 /robot/nav2_cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.7, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

Manual teleoperation via keyboard:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/robot/nav2_cmd_vel
```

## How the system works

The deployment flow is usually:

1. the simulator starts the robot and world,
2. Nav2 computes navigation commands,
3. the policy node receives observations and runs inference,
4. the actions are converted into robot commands,
5. the evaluation node measures tray tilt, orientation, and stability,
6. the experiment is compared against a baseline for analysis.

This gives a practical sim-to-real validation loop: train in IsaacLab, deploy in ROS 2 simulation, and measure the learned behavior under task-relevant conditions.

## Typical use cases

- validate a trained policy in simulation,
- compare learned tray stabilization against a fixed controller,
- evaluate navigation behavior with payload motion,
- debug tray tilt limits and stability under command changes.
