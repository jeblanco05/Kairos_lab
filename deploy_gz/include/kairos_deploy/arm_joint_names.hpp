#pragma once
// Canonical UR5e joint name list, in the order the trained policy expects
// (shoulder_pan -> wrist_3). Shared by every ROS2 node in kairos_deploy that
// needs to publish/read arm joint values, so the order can't drift between
// them (rl_inference_node.cpp, tray_tilt_eval_node.cpp, ...).

#include <array>
#include <string>

namespace kairos_deploy {

inline const std::array<std::string, 6> kArmJointNames = {
    "robot_arm_shoulder_pan_joint", "robot_arm_shoulder_lift_joint", "robot_arm_elbow_joint",
    "robot_arm_wrist_1_joint",      "robot_arm_wrist_2_joint",       "robot_arm_wrist_3_joint"};

}  // namespace kairos_deploy
