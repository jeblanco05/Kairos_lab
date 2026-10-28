#pragma once
// Loads deploy.yaml at runtime instead of hardcoding step_dt, joint offsets,
// action scales, and observation term/scale/history_length values in C++.
// This is what lets the same rl_inference_node.cpp work across different
// trained policies (models/<run_name>/params/deploy.yaml) without a rebuild.

#include <array>
#include <string>
#include <vector>

struct JointPositionActionCfg {
    std::vector<int> joint_ids;     // indices into the resolved "arm_.*_joint" list
    std::vector<double> scale;
    std::vector<double> offset;     // default joint positions, per joint_ids order
    std::vector<double> clip;
    int action_dim() const { return static_cast<int>(joint_ids.size()); }
};

struct ActionsCfg {
    // The base is no longer actuated by the policy -- it's driven directly by
    // an external node publishing cmd_vel. The ONNX action vector is now
    // arm-only, so there's no base action config and no offset into the
    // vector: acciones[:, 0:6] = arm.
    JointPositionActionCfg arm;
    int arm_offset = 0;
};

struct ObsTermCfg {
    std::string name;             // key used to look up the function in the term registry
    std::vector<double> scale;
    int history_length = 1;
};

struct ObsGroupCfg {
    std::string name;             // e.g. "obs_noised", "history"
    std::vector<ObsTermCfg> terms;
};

struct RangeCfg {
    double min = 0.0;
    double max = 0.0;
};
struct BaseVelocityCommandCfg {
    RangeCfg lin_vel_x;
    RangeCfg lin_vel_y;
    RangeCfg ang_vel_z;
};

struct TrayPoseCommandCfg {
    // Offsets from the EE home position (arm at default pose), decoded from
    // deploy.yaml's absolute base_link-frame ranges. Re-applied at runtime
    // on top of the REAL measured home (ee_center_b_), not the simulated one.
    RangeCfg offset_x;
    RangeCfg offset_y;
    RangeCfg offset_z;
    // Simulated TCP home used to decode the above -- kept for a sanity-check
    // log comparing it against the real measured ee_center_b_ at startup.
    std::array<double, 3> training_home_b{};
};

struct CommandsCfg {
    BaseVelocityCommandCfg base_velocity;
    TrayPoseCommandCfg tray_pose;
};

struct DeployConfig {
    double step_dt = 0.0333;
    ActionsCfg actions;
    std::vector<ObsGroupCfg> obs_groups;  // in policy_input_groups order
    CommandsCfg commands;
};

// Throws std::runtime_error on missing/malformed fields -- fail loudly at
// startup rather than silently deploying with wrong parameters.
DeployConfig load_deploy_config(const std::string& yaml_path);