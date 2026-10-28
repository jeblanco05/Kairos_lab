#pragma once
// One function per observation term, mirroring the structure of observations.py
// from the IsaacLab training pipeline, PLUS a name -> function registry so the
// set of terms actually used (and their scale / history_length) is driven by
// deploy.yaml at runtime instead of being hardcoded here.
//
// IMPORTANT: obs:: functions return RAW (unscaled) values. Scaling and history
// stacking are applied by ObservationBuffer using deploy_config.hpp data.

#include <array>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "kairos_deploy/deploy_config.hpp"

// Latest raw state of the robot, filled in by the ROS2 callbacks in
// rl_inference_node.cpp. Plain struct (no ROS types) so obs functions stay
// testable outside of ROS.
struct RobotState {
    // --- Base ---
    std::array<double, 3> base_ang_vel_b{0.0, 0.0, 0.0};
    std::array<double, 3> base_projected_gravity_b{0.0, 0.0, -1.0};

    // Reference base velocity command (vx, vy, wz) the policy should track.
    // Deployment source: teleop / Nav2 / fixed setpoint (NOT literally keyboard
    // input -- "velocity_commands" is just the training-side command term name).
    std::array<double, 3> base_vel_command{0.0, 0.0, 0.0};

    // --- Arm ---
    // Order: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
    // TODO(confirm): still needs verification via
    //   print(env.unwrapped.scene["robot"].joint_names)[4:10]
    std::array<double, 6> arm_joint_pos{};
    std::array<double, 6> arm_joint_vel{};

    // --- End effector (tray), expressed in robot_base_link frame, from TF ---
    std::array<double, 3> ee_pos_b{0.0, 0.0, 0.0};
    std::array<double, 4> ee_quat_b{1.0, 0.0, 0.0, 0.0};  // w, x, y, z
    std::array<double, 3> ee_projected_gravity_b{0.0, -1.0, 0.0};

    // --- Command ---
    std::array<double, 3> tray_pos_command{0.0, 0.0, 0.0};

    // --- Last action applied (6,) ---
    // [d_j0..d_j5] (arm joint order as above). Base is no longer part of the
    // action space -- it's driven directly by an external node via cmd_vel.
    std::array<double, 6> last_action{};
};

namespace obs {

std::vector<double> base_ang_vel(const RobotState& s);
std::vector<double> base_projected_gravity(const RobotState& s);
std::vector<double> velocity_commands(const RobotState& s);   // renamed from keyboard_velocity_commands
std::vector<double> tray_pose_command(const RobotState& s);   // position only
std::vector<double> arm_joint_pos(const RobotState& s);
std::vector<double> arm_joint_vel(const RobotState& s);
std::vector<double> ee_position_b(const RobotState& s);
std::vector<double> ee_orientation_b(const RobotState& s);
std::vector<double> ee_projected_gravity(const RobotState& s);
std::vector<double> last_action(const RobotState& s);

using ObsFn = std::vector<double> (*)(const RobotState&);

// Maps the term name string as it appears in deploy.yaml (e.g. "velocity_commands")
// to the corresponding function above. Extending the observation set in training
// only requires adding one entry here -- rl_inference_node.cpp doesn't change.
const std::unordered_map<std::string, ObsFn>& term_registry();

}  // namespace obs

// One ObservationBuffer per policy_input_groups entry (e.g. "obs_noised", "history").
// Each term keeps its OWN circular buffer at its OWN history_length, since
// deploy.yaml allows this to differ per term even within the same group.
class ObservationBuffer {
public:
    explicit ObservationBuffer(const ObsGroupCfg& group_cfg);

    // Fills every term's history with the same initial sample.
    void reset(const RobotState& s0);

    // Pushes a new sample into every term's history, dropping the oldest.
    void push(const RobotState& s);

    // Flattened, scaled observation vector for this group, ready for the
    // matching ONNX input tensor. Term order = declaration order in deploy.yaml;
    // within a term, history is oldest -> newest.
    std::vector<float> get_obs() const;

    const std::string& name() const { return name_; }
    int size() const { return size_; }

private:
    struct TermRuntime {
        std::string name;
        obs::ObsFn fn;
        std::vector<double> scale;
        int history_length;
        std::deque<std::vector<double>> history;  // raw (unscaled), oldest .. newest
    };

    std::string name_;
    int size_ = 0;
    std::vector<TermRuntime> terms_;
};