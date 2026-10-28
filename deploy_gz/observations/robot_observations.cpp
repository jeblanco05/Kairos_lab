#include "robot_observations.hpp"
#include <stdexcept>

namespace obs {

std::vector<double> base_ang_vel(const RobotState& s) {
    return {s.base_ang_vel_b[0], s.base_ang_vel_b[1], s.base_ang_vel_b[2]};
}

std::vector<double> base_projected_gravity(const RobotState& s) {
    return {s.base_projected_gravity_b[0], s.base_projected_gravity_b[1], s.base_projected_gravity_b[2]};
}

std::vector<double> velocity_commands(const RobotState& s) {
    return {s.base_vel_command[0], s.base_vel_command[1], s.base_vel_command[2]};
}

std::vector<double> tray_pose_command(const RobotState& s) {
    return {s.tray_pos_command[0], s.tray_pos_command[1], s.tray_pos_command[2]};
}

std::vector<double> arm_joint_pos(const RobotState& s) {
    return {s.arm_joint_pos.begin(), s.arm_joint_pos.end()};
}

std::vector<double> arm_joint_vel(const RobotState& s) {
    return {s.arm_joint_vel.begin(), s.arm_joint_vel.end()};
}

std::vector<double> ee_position_b(const RobotState& s) {
    return {s.ee_pos_b[0], s.ee_pos_b[1], s.ee_pos_b[2]};
}

std::vector<double> ee_orientation_b(const RobotState& s) {
    return {s.ee_quat_b[0], s.ee_quat_b[1], s.ee_quat_b[2], s.ee_quat_b[3]};
}

std::vector<double> ee_projected_gravity(const RobotState& s) {
    return {s.ee_projected_gravity_b[0], s.ee_projected_gravity_b[1], s.ee_projected_gravity_b[2]};
}

std::vector<double> last_action(const RobotState& s) {
    return {s.last_action.begin(), s.last_action.end()};
}

const std::unordered_map<std::string, ObsFn>& term_registry() {
    static const std::unordered_map<std::string, ObsFn> registry = {
        {"base_ang_vel", &base_ang_vel},
        {"base_projected_gravity", &base_projected_gravity},
        {"velocity_commands", &velocity_commands},
        {"tray_pose_command", &tray_pose_command},
        {"arm_joint_pos", &arm_joint_pos},
        {"arm_joint_vel", &arm_joint_vel},
        {"ee_position_b", &ee_position_b},
        {"ee_orientation_b", &ee_orientation_b},
        {"ee_projected_gravity", &ee_projected_gravity},
        {"last_action", &last_action},
    };
    return registry;
}

}  // namespace obs

ObservationBuffer::ObservationBuffer(const ObsGroupCfg& group_cfg) : name_(group_cfg.name) {
    const auto& registry = obs::term_registry();
    for (const auto& term_cfg : group_cfg.terms) {
        auto it = registry.find(term_cfg.name);
        if (it == registry.end()) {
            throw std::runtime_error("ObservationBuffer: unknown observation term '" + term_cfg.name +
                                      "' in deploy.yaml -- add it to obs::term_registry() in "
                                      "robot_observations.cpp");
        }
        TermRuntime term;
        term.name = term_cfg.name;
        term.fn = it->second;
        term.scale = term_cfg.scale;
        term.history_length = term_cfg.history_length;
        terms_.push_back(std::move(term));
        size_ += static_cast<int>(term_cfg.scale.size()) * term_cfg.history_length;
    }
}

void ObservationBuffer::reset(const RobotState& s0) {
    for (auto& term : terms_) {
        term.history.clear();
        std::vector<double> raw = term.fn(s0);
        if (raw.size() != term.scale.size()) {
            throw std::runtime_error("ObservationBuffer::reset: term '" + term.name +
                                      "' size does not match its scale array size in deploy.yaml");
        }
        for (int i = 0; i < term.history_length; ++i) term.history.push_back(raw);
    }
}

void ObservationBuffer::push(const RobotState& s) {
    if (terms_.empty()) return;
    if (terms_.front().history.empty()) {
        reset(s);
        return;
    }
    for (auto& term : terms_) {
        term.history.push_back(term.fn(s));
        while (static_cast<int>(term.history.size()) > term.history_length) {
            term.history.pop_front();
        }
    }
}

std::vector<float> ObservationBuffer::get_obs() const {
    std::vector<float> flat;
    flat.reserve(size_);
    for (const auto& term : terms_) {
        if (static_cast<int>(term.history.size()) != term.history_length) {
            throw std::runtime_error("ObservationBuffer::get_obs: term '" + term.name +
                                      "' called before reset()/enough push() calls");
        }
        for (const auto& sample : term.history) {
            for (size_t i = 0; i < sample.size(); ++i) {
                flat.push_back(static_cast<float>(sample[i] * term.scale[i]));
            }
        }
    }
    return flat;
}