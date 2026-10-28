# Kairos deploy

## Estructura del paquete

rbkairos_deploy/
    ├── CMakeLists.txt
    ├── package.xml
    ├── models/
    │   └── policy.onnx
    ├── config/
    │   └── rl_params.yaml
    ├── launch/
    │   ├── sim2sim.launch.py
    │   └── sim2real.launch.py
    └── src/
        └── rl_inference_node.cpp

sudo apt install ros-jazzy-ur-simulation-gz

colcon build --symlink-install --packages-select kairos_deploy
source install/setup.bash
export GZ_SIM_RESOURCE_PATH="$HOME/ros2_ws/install/kairos_deploy/share:$GZ_SIM_RESOURCE_PATH"

## Build

git clone https://github.com/RobotnikAutomation/robotnik_simulation.git



## Launchs

ros2 launch kairos_deploy full_simulation.launch.py launch_nav2:=true spawn_cube:=true

ros2 run kairos_deploy rl_inference_node --ros-args -p model_name:="ppo"

ros2 run kairos_deploy tray_tilt_eval_node --ros-args -p run_label:=policy -p test_num:=0
ros2 run kairos_deploy tray_tilt_eval_node --ros-args \
  -p run_label:=arm_fixed -p hold_arm_fixed:=true \
  -p fixed_arm_joint_positions:="[0.0, -1.57, 1.57, 0.0, 1.57, 0.0]" \
  -p test_num:=0

ros2 topic pub /kairos_deploy/tray_offset_target geometry_msgs/msg/Vector3 "{x: 0.1, y: 0.0, z: 0.05}" -r 10

ros2 topic pub -r 10 /robot/nav2_cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.7, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/robot/nav2_cmd_vel