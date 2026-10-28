#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <onnxruntime_cxx_api.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <algorithm>
#include <chrono>
#include <array>
#include <vector>
#include <memory>

#include "robot_observations.hpp"
#include "kairos_deploy/deploy_config.hpp"
#include "kairos_deploy/quat_math.hpp"
#include "kairos_deploy/arm_joint_names.hpp"

using namespace std::chrono_literals;

namespace {

using kairos_deploy::kArmJointNames;
using kairos_math::kGravityVecW;
using kairos_math::quat_apply;
using kairos_math::quat_apply_inverse;

constexpr double kArmSettleTimeoutSec = 5.0;

}  // namespace

class RLInferenceNode : public rclcpp::Node {
public:
    RLInferenceNode()
        : Node("rl_inference_node"),
          ort_env_(ORT_LOGGING_LEVEL_WARNING, "RL_Inference"),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_) {
        // "model_name" selects models/<model_name>/{exported/policy.onnx, params/deploy.yaml}.
        // Lets you switch trained policies without touching code or rebuilding.
        this->declare_parameter<std::string>("model_name", "ppo");
        const std::string model_name = this->get_parameter("model_name").as_string();

        const std::string package_share_directory =
            ament_index_cpp::get_package_share_directory("kairos_deploy");
        const std::string model_dir = package_share_directory + "/models/" + model_name;
        const std::string onnx_path = model_dir + "/exported/policy.onnx";
        const std::string deploy_yaml_path = model_dir + "/params/deploy.yaml";

        cfg_ = load_deploy_config(deploy_yaml_path);
        RCLCPP_INFO(this->get_logger(), "Loaded deploy config from %s (step_dt=%.4f, %zu obs groups)",
                    deploy_yaml_path.c_str(), cfg_.step_dt, cfg_.obs_groups.size());

        for (const auto& group_cfg : cfg_.obs_groups) {
            obs_buffers_.emplace_back(group_cfg);
            RCLCPP_INFO(this->get_logger(), "  obs group '%s': %d terms, flat size %d",
                        group_cfg.name.c_str(), static_cast<int>(group_cfg.terms.size()),
                        obs_buffers_.back().size());
        }

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_ = std::make_unique<Ort::Session>(ort_env_, onnx_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        size_t n_inputs = session_->GetInputCount();
        for (size_t i = 0; i < n_inputs; ++i) {
            auto name_ptr = session_->GetInputNameAllocated(i, allocator);
            input_names_storage_.emplace_back(name_ptr.get());
        }
        for (const auto& s : input_names_storage_) input_names_.push_back(s.c_str());
        RCLCPP_INFO(this->get_logger(), "ONNX input count: %zu", n_inputs);
        for (const auto& n : input_names_storage_) RCLCPP_INFO(this->get_logger(), "  input: %s", n.c_str());
        if (n_inputs != obs_buffers_.size()) {
            RCLCPP_WARN(this->get_logger(),
                        "ONNX input count (%zu) != number of obs groups in deploy.yaml (%zu) -- "
                        "double check the mapping before trusting inference output.",
                        n_inputs, obs_buffers_.size());
        }

        this->declare_parameter<double>("command_timeout_sec", 0.5);
        command_timeout_sec_ = this->get_parameter("command_timeout_sec").as_double();

        sub_joint_states_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/robot/joint_states", 10,
            std::bind(&RLInferenceNode::joint_state_callback, this, std::placeholders::_1));
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/robot/imu/data", 10, std::bind(&RLInferenceNode::imu_callback, this, std::placeholders::_1));

        // External command inputs 
        sub_cmd_vel_target_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/robot/nav2_cmd_vel", 10,
            std::bind(&RLInferenceNode::cmd_vel_target_callback, this, std::placeholders::_1));
        // tray_offset_target: offset (meters) from the trained resting position of
        // the tray (ee_center_b_, measured at startup with the arm at its default
        // pose) -- NOT a world-anchored point. This matches how tray_pose's
        // command ranges were actually sampled in training
        // (center_command_ranges_on_body), and keeps the policy in-distribution
        // regardless of how far the base has driven.
        sub_tray_offset_target_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/kairos_deploy/tray_offset_target", 10,
            std::bind(&RLInferenceNode::tray_offset_target_callback, this, std::placeholders::_1));
        
        pub_arm_traj_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/robot/joint_trajectory_controller/joint_trajectory", 10);

        last_action_.fill(0.0);
        node_start_time_ = this->get_clock()->now();
        // Move to the trained default pose before enabling the policy.
        publish_arm_target(cfg_.actions.arm.offset);

        const auto period = std::chrono::duration<double>(cfg_.step_dt);
        timer_ = this->create_wall_timer(std::chrono::duration_cast<std::chrono::milliseconds>(period),
                                          std::bind(&RLInferenceNode::control_loop, this));
    }

private:
    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        for (size_t i = 0; i < kArmJointNames.size(); ++i) {
            for (size_t j = 0; j < msg->name.size(); ++j) {
                if (msg->name[j] == kArmJointNames[i]) {
                    if (j < msg->position.size()) {
                        // FIX: Restamos el offset para enviar la posición RELATIVA a la red neuronal
                        state_.arm_joint_pos[i] = msg->position[j] - cfg_.actions.arm.offset[i];
                    }
                    if (j < msg->velocity.size()) state_.arm_joint_vel[i] = msg->velocity[j];
                    break;
                }
            }
        }
        joint_state_ready_ = true;
    }

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        geometry_msgs::msg::TransformStamped tf_imu_to_base;
        try {
            tf_imu_to_base = tf_buffer_.lookupTransform("robot_base_link", msg->header.frame_id, tf2::TimePointZero);
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                "TF robot_base_link->%s not available yet: %s", msg->header.frame_id.c_str(), ex.what());
            return;
        }
        const std::array<double, 4> q_imu_to_base = {
            tf_imu_to_base.transform.rotation.w, tf_imu_to_base.transform.rotation.x,
            tf_imu_to_base.transform.rotation.y, tf_imu_to_base.transform.rotation.z};

        // Both quantities are body-frame vectors -- a pure rotation by the static
        // mounting offset carries them from imu frame into base_link frame.
        const std::array<double, 3> ang_vel_imu = {msg->angular_velocity.x, msg->angular_velocity.y,
                                                    msg->angular_velocity.z};
        state_.base_ang_vel_b = quat_apply(q_imu_to_base, ang_vel_imu);

        const std::array<double, 4> imu_quat = {msg->orientation.w, msg->orientation.x, msg->orientation.y,
                                                msg->orientation.z};
        const std::array<double, 3> grav_in_imu = quat_apply_inverse(imu_quat, kGravityVecW);
        state_.base_projected_gravity_b = quat_apply(q_imu_to_base, grav_in_imu);

        imu_ready_ = true;
    }

    void cmd_vel_target_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        raw_vel_command_ = {msg->linear.x, msg->linear.y, msg->angular.z};
        last_cmd_vel_time_ = this->get_clock()->now();
        cmd_vel_received_ = true;
    }

    void tray_offset_target_callback(const geometry_msgs::msg::Vector3::SharedPtr msg) {
        raw_tray_offset_ = {msg->x, msg->y, msg->z};
        last_tray_offset_time_ = this->get_clock()->now();
        tray_offset_received_ = true;
    }

    bool update_ee_state_from_tf() {
        geometry_msgs::msg::TransformStamped tf_ee;
        try {
            tf_ee = tf_buffer_.lookupTransform("robot_base_link", "rg6_tcp_link", tf2::TimePointZero);
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                  "TF robot_base_link->rg6_tcp_link not available yet: %s", ex.what());
            return false;
        }
        state_.ee_pos_b = {tf_ee.transform.translation.x, tf_ee.transform.translation.y,
                            tf_ee.transform.translation.z};
        state_.ee_quat_b = {tf_ee.transform.rotation.w, tf_ee.transform.rotation.x,
                             tf_ee.transform.rotation.y, tf_ee.transform.rotation.z};
        // Compose gravity through BASE frame (not the world-fixed
        // kGravityVecW directly): base_projected_gravity_b already accounts
        // for the base's real orientation via the IMU, which is what
        // actually captures tilt from ramps/uneven terrain. Rotating
        // kGravityVecW straight through ee_quat_b would silently assume the
        // base itself is level -- wrong on any slope.
        state_.ee_projected_gravity_b = quat_apply_inverse(state_.ee_quat_b, state_.base_projected_gravity_b);
        return true;
    }

    void update_velocity_command() {
        const bool fresh = cmd_vel_received_ &&
                            (this->get_clock()->now() - last_cmd_vel_time_).seconds() < command_timeout_sec_;
        if (!fresh) {
            state_.base_vel_command = {0.0, 0.0, 0.0};
            return;
        }
        const auto& c = cfg_.commands.base_velocity;
        state_.base_vel_command = {
            std::clamp(raw_vel_command_[0], c.lin_vel_x.min, c.lin_vel_x.max),
            std::clamp(raw_vel_command_[1], c.lin_vel_y.min, c.lin_vel_y.max),
            std::clamp(raw_vel_command_[2], c.ang_vel_z.min, c.ang_vel_z.max),
        };
    }

    void update_tray_target() {
        const bool fresh = tray_offset_received_ &&
                            (this->get_clock()->now() - last_tray_offset_time_).seconds() < command_timeout_sec_;
        std::array<double, 3> offset = fresh ? raw_tray_offset_ : std::array<double, 3>{0.0, 0.0, 0.0};

        const auto& c = cfg_.commands.tray_pose;
        offset[0] = std::clamp(offset[0], c.offset_x.min, c.offset_x.max);
        offset[1] = std::clamp(offset[1], c.offset_y.min, c.offset_y.max);
        offset[2] = std::clamp(offset[2], c.offset_z.min, c.offset_z.max);

        for (int i = 0; i < 3; ++i) state_.tray_pos_command[i] = ee_center_b_[i] + offset[i];
    }

    void publish_arm_target(const std::vector<double>& target_pos) {
        trajectory_msgs::msg::JointTrajectory msg;
        msg.joint_names.assign(kArmJointNames.begin(), kArmJointNames.end());
        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions.assign(target_pos.begin(), target_pos.end());
        point.time_from_start = rclcpp::Duration(std::chrono::duration<double>(cfg_.step_dt));
        msg.points.push_back(point);
        pub_arm_traj_->publish(msg);
    }

    void apply_arm_action(const float* action) {
        const auto& a = cfg_.actions.arm;
        std::vector<double> target(a.joint_ids.size());
        for (size_t i = 0; i < a.joint_ids.size(); ++i) {
            double net_action = action[cfg_.actions.arm_offset + i];
            
            // Si el YAML define un clip, limitamos la acción cruda de la red
            if (!a.clip.empty()) {
                net_action = std::clamp(net_action, -a.clip[i], a.clip[i]);
            }
            
            // Calculamos el delta aplicando la escala (0.15)
            const double delta = a.scale[i] * net_action;
            target[i] = a.offset[i] + delta;
        }
        publish_arm_target(target);
    }

    bool arm_at_default_pose() const {
        constexpr double kTolRad = 0.05;
        for (size_t i = 0; i < cfg_.actions.arm.offset.size(); ++i) {
            // Como arm_joint_pos ahora es relativo, debe tender a 0 cuando alcanza la pose por defecto
            if (std::abs(state_.arm_joint_pos[i]) > kTolRad) return false;
        }
        return true;
    }

    void control_loop() {
        if (!joint_state_ready_ || !imu_ready_) return;
        if (!update_ee_state_from_tf()) return;

        if (!center_measured_) {
            // Esperar 1 segundo para que Gazebo estabilice la gravedad, la IMU y el TF
            const double elapsed = (this->get_clock()->now() - node_start_time_).seconds();
            if (elapsed < 1.0) {
                return; 
            }

            ee_center_b_ = state_.ee_pos_b;
            center_measured_ = true;
            RCLCPP_INFO(this->get_logger(), "ee_center_b_ = (%.3f, %.3f, %.3f)", ee_center_b_[0],
                        ee_center_b_[1], ee_center_b_[2]);

            const auto& home = cfg_.commands.tray_pose.training_home_b;
            for (int i = 0; i < 3; ++i) {
                if (std::abs(ee_center_b_[i] - home[i]) > 0.05) {
                    RCLCPP_WARN(this->get_logger(),
                                "ee_center_b_[%d]=%.3f differs from training home %.3f by >5cm -- "
                                "check URDF/calibration, tray_pose clamp bounds may be off in the real frame.",
                                i, ee_center_b_[i], home[i]);
                }
            }
        }

        update_velocity_command();
        update_tray_target();

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> input_tensors;
        std::vector<std::vector<float>> group_obs;  // keep alive until Run() returns
        std::vector<std::vector<int64_t>> group_shapes;

        for (auto& buf : obs_buffers_) {
            buf.push(state_);
            group_obs.push_back(buf.get_obs());
            group_shapes.push_back({1, static_cast<int64_t>(group_obs.back().size())});
        }
        for (size_t i = 0; i < group_obs.size(); ++i) {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, group_obs[i].data(), group_obs[i].size(), group_shapes[i].data(),
                group_shapes[i].size()));
        }

        const char* output_names[] = {"actions"};
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), input_tensors.data(),
                                             input_tensors.size(), output_names, 1);
        const float* action = output_tensors.front().GetTensorMutableData<float>();

        first_action_logged_ = true;
        if (!first_action_logged_) {
            first_action_logged_ = true;
            std::string obs_dump;
            for (float v : group_obs[0]) obs_dump += std::to_string(v) + ", ";
            RCLCPP_INFO(this->get_logger(), "First obs vector (size %zu): [%s]", group_obs[0].size(),
                        obs_dump.c_str());
            RCLCPP_INFO(this->get_logger(), "First policy action (arm only): [%.3f %.3f %.3f %.3f %.3f %.3f]",
                        action[0], action[1], action[2], action[3], action[4], action[5]);
            RCLCPP_INFO(this->get_logger(), "arm_joint_pos at activation: [%.3f %.3f %.3f %.3f %.3f %.3f]",
                        state_.arm_joint_pos[0], state_.arm_joint_pos[1], state_.arm_joint_pos[2],
                        state_.arm_joint_pos[3], state_.arm_joint_pos[4], state_.arm_joint_pos[5]);
        }

        apply_arm_action(action);

        for (int i = 0; i < 6; ++i) last_action_[i] = action[i];
        state_.last_action = last_action_;
    }

    Ort::Env ort_env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_storage_;
    std::vector<const char*> input_names_;

    DeployConfig cfg_;
    std::vector<ObservationBuffer> obs_buffers_;

    RobotState state_;
    std::array<double, 6> last_action_;
    bool joint_state_ready_ = false;
    bool imu_ready_ = false;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_states_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_vel_target_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr sub_tray_offset_target_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr pub_arm_traj_;
    rclcpp::TimerBase::SharedPtr timer_;

    double command_timeout_sec_ = 0.5;
    std::array<double, 3> raw_vel_command_{0.0, 0.0, 0.0};
    bool cmd_vel_received_ = false;
    rclcpp::Time last_cmd_vel_time_;
    std::array<double, 3> raw_tray_offset_{0.0, 0.0, 0.0};
    bool tray_offset_received_ = false;
    rclcpp::Time last_tray_offset_time_;

    // Measured once at startup, arm at default pose -- the deployment equivalent
    // of center_command_ranges_on_body's "center".
    std::array<double, 3> ee_center_b_{0.0, 0.0, 0.0};
    bool center_measured_ = false;
    rclcpp::Time node_start_time_;
    bool first_action_logged_ = false;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RLInferenceNode>());
    rclcpp::shutdown();
    return 0;
}