#include "deploy_config.hpp"
#include <yaml-cpp/yaml.h>
#include <stdexcept>

namespace {

std::vector<double> as_double_vec(const YAML::Node& n) {
    std::vector<double> out;
    for (const auto& v : n) out.push_back(v.as<double>());
    return out;
}

}  // namespace

DeployConfig load_deploy_config(const std::string& yaml_path) {
    YAML::Node root = YAML::LoadFile(yaml_path);
    DeployConfig cfg;

    cfg.step_dt = root["step_dt"].as<double>();

    // --- actions ---
    // The base is no longer actuated by the policy (it's driven directly by an
    // external node via cmd_vel), so the only action term left is the arm's
    // JointPositionAction -- there is no JointVelocityAction to parse anymore.
    YAML::Node actions_node = root["actions"];
    if (!actions_node["JointPositionAction"]) {
        throw std::runtime_error(
            "deploy_config: expected JointPositionAction under 'actions' -- this loader is "
            "written for that specific schema, update it if the action set changes.");
    }
    {
        YAML::Node n = actions_node["JointPositionAction"];
        auto& a = cfg.actions.arm;
        for (const auto& v : n["joint_ids"]) a.joint_ids.push_back(v.as<int>());
        a.scale = as_double_vec(n["scale"]);
        a.offset = as_double_vec(n["offset"]);
        
        // Nueva lógica para leer el clip dinámicamente
        if (!n["clip"].IsNull()) {
            a.clip = as_double_vec(n["clip"]);
            if (a.clip.size() != a.joint_ids.size()) {
                throw std::runtime_error("deploy_config: JointPositionAction clip size mismatch");
            }
        }

        if (a.scale.size() != a.joint_ids.size() || a.offset.size() != a.joint_ids.size()) {
            throw std::runtime_error("deploy_config: JointPositionAction scale/offset/joint_ids size mismatch");
        }
    }
    cfg.actions.arm_offset = 0;

    // --- commands: defines the TRAINED DISTRIBUTION for the external target
    // topics, as opposed to `actions`, which defines the raw network-output
    // clip. Both matter but for different things -- see rl_inference_node.cpp
    // update_velocity_command()/update_tray_target(). ---
    YAML::Node commands_node = root["commands"];
    if (!commands_node["base_velocity"] || !commands_node["tray_pose"]) {
        throw std::runtime_error(
            "deploy_config: expected both 'base_velocity' and 'tray_pose' under "
            "'commands' -- this loader is written for that specific schema, "
            "update it if the command set changes.");
    }
    {
        YAML::Node r = commands_node["base_velocity"]["ranges"];
        auto& c = cfg.commands.base_velocity;
        c.lin_vel_x = {r["lin_vel_x"][0].as<double>(), r["lin_vel_x"][1].as<double>()};
        c.lin_vel_y = {r["lin_vel_y"][0].as<double>(), r["lin_vel_y"][1].as<double>()};
        c.ang_vel_z = {r["ang_vel_z"][0].as<double>(), r["ang_vel_z"][1].as<double>()};
    }
    {
        // tray_pose ranges in deploy.yaml are ABSOLUTE base_link-frame positions,
        // already shifted by center_command_ranges_on_body using the SIMULATED
        // EE home at the arm's default pose. We decode them back into pure
        // offsets-from-home here, so they can be re-applied to the REAL measured
        // home (ee_center_b_) at deploy time -- see update_tray_target().
        //
        // kTrainingTrayHomeB matches the project's confirmed TCP home position
        // (0.6921, 0.1335, 1.1764 at yaw=0). If you retrain with a different
        // default arm pose or different tray ranges, this constant must be
        // updated to match, or the decoded offsets will be silently wrong.
        constexpr std::array<double, 3> kTrainingTrayHomeB = {0.6921, 0.1335, 1.1764};

        YAML::Node r = commands_node["tray_pose"]["ranges"];
        auto decode = [&](const char* key, double home) {
            RangeCfg rc;
            rc.min = r[key][0].as<double>() - home;
            rc.max = r[key][1].as<double>() - home;
            return rc;
        };
        auto& c = cfg.commands.tray_pose;
        c.offset_x = decode("pos_x", kTrainingTrayHomeB[0]);
        c.offset_y = decode("pos_y", kTrainingTrayHomeB[1]);
        c.offset_z = decode("pos_z", kTrainingTrayHomeB[2]);
        c.training_home_b = kTrainingTrayHomeB;
    }

    // --- observations, in policy_input_groups order ---
    YAML::Node policy_groups = root["policy_input_groups"];
    YAML::Node obs_node = root["observations"];
    for (const auto& g : policy_groups) {
        const std::string group_name = g.as<std::string>();
        YAML::Node group_node = obs_node[group_name];
        if (!group_node) {
            throw std::runtime_error("deploy_config: observation group '" + group_name + "' not found");
        }

        ObsGroupCfg group;
        group.name = group_name;
        for (auto it = group_node.begin(); it != group_node.end(); ++it) {
            ObsTermCfg term;
            term.name = it->first.as<std::string>();
            YAML::Node term_node = it->second;
            term.scale = as_double_vec(term_node["scale"]);
            term.history_length = term_node["history_length"].as<int>();
            group.terms.push_back(std::move(term));
        }
        cfg.obs_groups.push_back(std::move(group));
    }

    return cfg;
}