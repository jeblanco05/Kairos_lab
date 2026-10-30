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

## Build

git clone https://github.com/jeblanco05/Kairos_lab.git

git clone https://github.com/RobotnikAutomation/robotnik_simulation.git


cd Kairos_lab/deploy_gz/Docker
docker compose up -d --build --force-recreate simulation
docker exec -it -u robot kairos_sim_jazzy bash

sudo apt update && sudo apt install ros-jazzy-mecanum-drive-controller

cd /home/robot/ros2_ws
colcon build --symlink-install 
source install/setup.bash

## Launchs
1º docker exec -it -u robot kairos_sim_jazzy bash

2º ros2 launch kairos_deploy full_simulation.launch.py launch_nav2:=true spawn_cube:=true

3º ros2 run kairos_deploy rl_inference_node --ros-args -p model_name:="ppo"

ros2 run kairos_deploy tray_tilt_eval_node --ros-args -p run_label:=policy -p test_num:=0
ros2 run kairos_deploy tray_tilt_eval_node --ros-args \
  -p run_label:=arm_fixed -p hold_arm_fixed:=true \
  -p fixed_arm_joint_positions:="[0.0, -1.57, 1.57, 0.0, 1.57, 0.0]" \
  -p test_num:=0

ros2 topic pub /kairos_deploy/tray_offset_target geometry_msgs/msg/Vector3 "{x: 0.1, y: 0.0, z: 0.05}" -r 10

ros2 topic pub -r 10 /robot/nav2_cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.7, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/robot/nav2_cmd_vel