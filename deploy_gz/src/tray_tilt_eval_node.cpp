// tray_tilt_eval_node.cpp
//
// Standalone evaluation node for comparing tray tilt WITH the RL policy
// running vs. WITH the arm held fixed, over the identical base trajectory.
// It drives the base through a fixed sequence of test maneuvers via
// /robot/nav2_cmd_vel and logs a CSV of tilt (and a few other signals) over
// time, tagged with a `run_label` so two runs can be overlaid later.
//
// Two ways to use it:
//
//   1) WITH rl_inference_node running (measures the policy; it reads the
//      same /robot/nav2_cmd_vel this node publishes as its velocity
//      observation, so no extra wiring is needed):
//        ros2 run kairos_deploy rl_inference_node &
//        ros2 run kairos_deploy tray_tilt_eval_node --ros-args -p run_label:=policy
//
//   2) WITHOUT rl_inference_node (arm-fixed baseline): nothing else publishes
//      to the arm trajectory controller in this case, so the arm's pose is
//      otherwise undefined (wherever it was last left). Set hold_arm_fixed
//      true and pass the same joint target the policy uses as its default
//      pose (cfg_.actions.arm.offset in deploy.yaml) so both runs start from
//      an identical, reproducible arm pose:
//        ros2 run kairos_deploy tray_tilt_eval_node --ros-args \   
//          -p run_label:=arm_fixed -p hold_arm_fixed:=true \   
//          -p fixed_arm_joint_positions:="[0.0, -1.57, 1.57, 0.0, 1.57, 0.0]"
//
// Keep every other parameter (test_lin_vel, phase_duration_sec, ...)
// identical between the two runs so both follow the exact same base
// trajectory -- that's what makes the two CSVs comparable.
//
// IMPORTANT: /robot/nav2_cmd_vel is now BOTH the base actuation topic and the
// topic rl_inference_node reads for its base_vel_command OBSERVATION (there
// is no longer a separate /kairos_deploy/cmd_vel_target). So simply
// publishing the test maneuvers here is enough for the policy to see them
// and compensate with the arm -- no mirroring needed.
//
// Reuses RobotState / obs::ee_projected_gravity from robot_observations.hpp,
// and the same quaternion + joint-name helpers as rl_inference_node.cpp
// (quat_math.hpp, arm_joint_names.hpp), so tilt and joint order match exactly
// what the policy itself sees.

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/vector3.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "robot_observations.hpp"
#include "kairos_deploy/quat_math.hpp"
#include "kairos_deploy/arm_joint_names.hpp"

using namespace std::chrono_literals;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Tray local "up" axis in the rg6_tcp_link frame -- confirmed empirically for
// this robot (NOT (0,0,1)); see project notes. "Down" is therefore (0,-1,0).
constexpr int kUpAxisIndex = 1;

struct TiltStats {
    double sum_deg = 0.0;
    double sum_sq_deg = 0.0;
    double max_deg = 0.0;
    int count = 0;

    void add(double tilt_deg) {
        sum_deg += tilt_deg;
        sum_sq_deg += tilt_deg * tilt_deg;
        max_deg = std::max(max_deg, tilt_deg);
        ++count;
    }

    double mean() const { return count > 0 ? sum_deg / count : 0.0; }
    double stddev() const {
        if (count < 2) return 0.0;
        const double m = mean();
        return std::sqrt(std::max(0.0, sum_sq_deg / count - m * m));
    }
};

struct Phase {
    std::string name;
    geometry_msgs::msg::Twist cmd;
    geometry_msgs::msg::Vector3 tray_offset; // NUEVO: Offset objetivo de la bandeja
    double duration_sec = 0.0;
};

struct TiltSample {
    double tilt_deg = 0.0;
    std::array<double, 3> proj_grav{0.0, 0.0, 0.0};
    std::array<double, 3> ee_pos_b{0.0, 0.0, 0.0};
};

geometry_msgs::msg::Twist make_twist(double vx, double vy, double wz) {
    geometry_msgs::msg::Twist t;
    t.linear.x = vx; t.linear.y = vy; t.angular.z = wz;
    return t;
}

geometry_msgs::msg::Vector3 make_vec3(double x, double y, double z) {
    geometry_msgs::msg::Vector3 v;
    v.x = x; v.y = y; v.z = z;
    return v;
}

}  // namespace

class TrayTiltEvalNode : public rclcpp::Node {
public:
    TrayTiltEvalNode() : Node("tray_tilt_eval_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
        this->declare_parameter<std::string>("base_frame", "robot_base_link");
        this->declare_parameter<std::string>("ee_frame", "rg6_tcp_link");
        this->declare_parameter<std::string>("command_topic", "/robot/nav2_cmd_vel");
        this->declare_parameter<double>("eval_rate_hz", 30.0);
        this->declare_parameter<double>("test_lin_vel", 0.7);   // m/s
        this->declare_parameter<double>("test_ang_vel", 0.3);   // rad/s
        this->declare_parameter<double>("phase_duration_sec", 30.0);
        this->declare_parameter<double>("settle_duration_sec", 2.0);
        this->declare_parameter<int>("test_num", 0);

        // -- Run tagging / logging --
        this->declare_parameter<std::string>("run_label", "run");
        this->declare_parameter<std::string>("log_dir", "src/kairos_deploy/tilt_eval_logs");

        // -- Arm-fixed baseline mode (use when rl_inference_node is NOT running) --
        this->declare_parameter<bool>("hold_arm_fixed", false);
        this->declare_parameter<std::vector<double>>("fixed_arm_joint_positions", std::vector<double>{});
        this->declare_parameter<std::string>("arm_traj_topic",
                                              "/robot/joint_trajectory_controller/joint_trajectory");

        base_frame_ = this->get_parameter("base_frame").as_string();
        ee_frame_ = this->get_parameter("ee_frame").as_string();
        const std::string command_topic = this->get_parameter("command_topic").as_string();
        const double eval_rate_hz = this->get_parameter("eval_rate_hz").as_double();
        const double test_lin_vel = this->get_parameter("test_lin_vel").as_double();
        const double test_ang_vel = this->get_parameter("test_ang_vel").as_double();
        const double phase_duration_sec = this->get_parameter("phase_duration_sec").as_double();
        const double settle_duration_sec = this->get_parameter("settle_duration_sec").as_double();
        const int test_num = this->get_parameter("test_num").as_int();
        run_label_ = this->get_parameter("run_label").as_string();
        const std::string log_dir = this->get_parameter("log_dir").as_string();
        hold_arm_fixed_ = this->get_parameter("hold_arm_fixed").as_bool();
        fixed_arm_joint_positions_ = this->get_parameter("fixed_arm_joint_positions").as_double_array();
        const std::string arm_traj_topic = this->get_parameter("arm_traj_topic").as_string();

        if (eval_rate_hz <= 0.0) throw std::runtime_error("eval_rate_hz must be > 0");
        if (phase_duration_sec <= 0.0) throw std::runtime_error("phase_duration_sec must be > 0");
        if (settle_duration_sec < 0.0) throw std::runtime_error("settle_duration_sec must be >= 0");
        if (test_num > 2) throw std::runtime_error("test_num must be 0, 1, 2");
        if (run_label_.empty()) throw std::runtime_error("run_label must not be empty (used in the CSV filename)");
        if (hold_arm_fixed_ && fixed_arm_joint_positions_.size() != kairos_deploy::kArmJointNames.size()) {
            throw std::runtime_error(
                "hold_arm_fixed=true requires fixed_arm_joint_positions with exactly " +
                std::to_string(kairos_deploy::kArmJointNames.size()) +
                " values (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3) -- "
                "use the same values as JointPositionAction.offset in deploy.yaml so the "
                "baseline matches the policy's own default pose.");
        }

        build_phases(test_lin_vel, test_ang_vel, phase_duration_sec, settle_duration_sec, test_num);
        open_log_file(log_dir);

        pub_cmd_vel_ = this->create_publisher<geometry_msgs::msg::Twist>(command_topic, 10);
        pub_tray_target_ = this->create_publisher<geometry_msgs::msg::Vector3>("/kairos_deploy/tray_offset_target", 10);
        if (hold_arm_fixed_) {
            pub_arm_traj_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(arm_traj_topic, 10);
        }

        // Needed to correctly account for the BASE's own real-world tilt
        // (ramps, uneven terrain) -- see measure_tilt() / imu_callback().
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/robot/imu/data", 10, std::bind(&TrayTiltEvalNode::imu_callback, this, std::placeholders::_1));


        RCLCPP_INFO(this->get_logger(),
                    "tray_tilt_eval_node: run_label='%s', publishing base commands to '%s' (also read by "
                    "rl_inference_node as its velocity observation, if running). %zu phases queued.",
                    run_label_.c_str(), command_topic.c_str(), phases_.size());
        if (hold_arm_fixed_) {
            RCLCPP_INFO(this->get_logger(),
                         "hold_arm_fixed=true -- will continuously hold the arm at the given fixed pose "
                         "on '%s'. Make sure rl_inference_node is NOT also commanding the arm.",
                         arm_traj_topic.c_str());
        } else {
            RCLCPP_WARN_ONCE(this->get_logger(),
                              "hold_arm_fixed=false -- this node will not command the arm at all. If "
                              "rl_inference_node isn't running, the arm's pose is whatever it was last "
                              "left at (not a controlled baseline).");
        }

        start_time_ = this->get_clock()->now();
        phase_start_time_ = start_time_;
        RCLCPP_INFO(this->get_logger(), "Phase [1/%zu] '%s' (%.1fs, vx=%.2f vy=%.2f wz=%.2f)", phases_.size(),
                    phases_[0].name.c_str(), phases_[0].duration_sec, phases_[0].cmd.linear.x,
                    phases_[0].cmd.linear.y, phases_[0].cmd.angular.z);

        const auto period = std::chrono::duration<double>(1.0 / eval_rate_hz);
        timer_ = this->create_wall_timer(std::chrono::duration_cast<std::chrono::milliseconds>(period),
                                          std::bind(&TrayTiltEvalNode::tick, this));
    }

    ~TrayTiltEvalNode() override {
        if (log_file_.is_open()) log_file_.close();
    }

private:
    void build_phases(double v, double w, double phase_dur, double settle_dur, int test_num) {
        (void)w;
        phases_.push_back({"settle", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.0, 0.0), settle_dur});

        if (test_num == 0) {
            phases_.push_back({"forward_x", make_twist(v, 0.0, 0.0), make_vec3(0.0, 0.0, 0.0), phase_dur});

        }

        else if (test_num == 1) {
            phases_.push_back({"forward_x", make_twist(v, 0.0, 0.0), make_vec3(0.0, 0.0, 0.0), phase_dur});
            phases_.push_back({"lift_z", make_twist(-v, 0.0, 0.0), make_vec3(0.25, 0.15, 0.15), phase_dur/2});
        
            phases_.push_back({"settle", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.0, 0.0), settle_dur});
            phases_.push_back({"lateral_y", make_twist(0.0, v, 0.0), make_vec3(0.0, 0.0, 0.0), phase_dur/2});
            phases_.push_back({"lateral_y", make_twist(0.0, -v, 0.0), make_vec3(-0.25, -0.15, -0.15), phase_dur/2});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.0, 0.0), settle_dur});
            phases_.push_back({"curve", make_twist(v, 0.0, -w), make_vec3(0.0, 0.0, 0.0), phase_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.0, 0.0), settle_dur});

        }

        else if (test_num == 2) {
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.0, 0.0), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.05, 0.0, 0.0), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.1, 0.0, 0.0), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.15, 0.0, 0.0), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.2, 0.0, 0.0), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.25, 0.0, 0.0), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.3, 0.0, 0.0), settle_dur});
            
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.0, 0.0), settle_dur});

            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.0, 0.0), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.0, 0.05), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.0, 0.1), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.0, 0.15), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.0, 0.2), settle_dur});

            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.0, 0.0), settle_dur});

            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.0, 0.0), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.05, 0.0), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.1, 0.0), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.15, 0.0), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.2, 0.0), settle_dur});

            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.0, 0.0), settle_dur});

            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.0, 0.0, 0.0), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.05, 0.05, 0.05), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.1, 0.1, 0.1), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.15, 0.15, 0.15), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.2, 0.2, 0.2), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.25, 0.2, 0.2), settle_dur});
            phases_.push_back({"zero", make_twist(0.0, 0.0, 0.0), make_vec3(0.3, 0.2, 0.2), settle_dur});


        }
    }

    void open_log_file(const std::string& log_dir) {
        std::error_code ec;
        std::filesystem::create_directories(log_dir, ec);
        if (ec) {
            throw std::runtime_error("tray_tilt_eval_node: could not create log_dir '" + log_dir +
                                      "': " + ec.message());
        }
        log_path_ = (std::filesystem::path(log_dir) / ("tilt_eval_" + run_label_ + ".csv")).string();
        log_file_.open(log_path_, std::ios::out | std::ios::trunc);
        if (!log_file_.is_open()) {
            throw std::runtime_error("tray_tilt_eval_node: could not open log file '" + log_path_ + "'");
        }
        log_file_ << "run_label,t_sec,phase_idx,phase_name,cmd_vx,cmd_vy,cmd_wz,tilt_deg,"
                     "proj_grav_x,proj_grav_y,proj_grav_z,ee_pos_x,ee_pos_y,ee_pos_z,ef_pos_error\n";
        log_file_.flush();
        RCLCPP_INFO(this->get_logger(), "Logging to '%s'", log_path_.c_str());
    }

    void log_row(double t_sec, const geometry_msgs::msg::Twist& cmd, const TiltSample& sample, double ef_pos_error) {
        log_file_ << run_label_ << ',' << t_sec << ',' << phase_idx_ << ',' << phases_[phase_idx_].name << ','
                   << cmd.linear.x << ',' << cmd.linear.y << ',' << cmd.angular.z << ',' << sample.tilt_deg << ','
                   << sample.proj_grav[0] << ',' << sample.proj_grav[1] << ',' << sample.proj_grav[2] << ','
                   << sample.ee_pos_b[0] << ',' << sample.ee_pos_b[1] << ',' << sample.ee_pos_b[2] << ','
                   << ef_pos_error << '\n';
        log_file_.flush();
    }

    // Mirrors rl_inference_node::imu_callback() -- keep them in sync if this
    // logic ever changes. Computes the world gravity vector expressed in
    // base_link frame, using the IMU's real orientation (this is what
    // actually captures the base tilting on a ramp/slope; joint FK alone
    // cannot see that). This is the DEPLOYED reconstruction path (tilt_deg).
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        geometry_msgs::msg::TransformStamped tf_imu_to_base;
        try {
            tf_imu_to_base = tf_buffer_.lookupTransform(base_frame_, msg->header.frame_id, tf2::TimePointZero);
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "TF %s->%s not available yet: %s",
                                  base_frame_.c_str(), msg->header.frame_id.c_str(), ex.what());
            return;
        }
        const std::array<double, 4> q_imu_to_base = {
            tf_imu_to_base.transform.rotation.w, tf_imu_to_base.transform.rotation.x,
            tf_imu_to_base.transform.rotation.y, tf_imu_to_base.transform.rotation.z};

        const std::array<double, 4> imu_quat = {msg->orientation.w, msg->orientation.x, msg->orientation.y,
                                                  msg->orientation.z};
        const std::array<double, 3> grav_in_imu = kairos_math::quat_apply_inverse(imu_quat, kairos_math::kGravityVecW);
        base_projected_gravity_b_ = kairos_math::quat_apply(q_imu_to_base, grav_in_imu);
        imu_ready_ = true;
    }

    // Returns true if it could measure the current (deployed-pipeline) tray tilt.
    bool measure_tilt(TiltSample& sample_out) {
        if (!imu_ready_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                  "No IMU data yet on '/robot/imu/data' -- can't correct for base tilt, "
                                  "skipping this sample.");
            return false;
        }

        geometry_msgs::msg::TransformStamped tf_ee;
        try {
            tf_ee = tf_buffer_.lookupTransform(base_frame_, ee_frame_, tf2::TimePointZero);
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "TF %s->%s not available yet: %s",
                                  base_frame_.c_str(), ee_frame_.c_str(), ex.what());
            return false;
        }

        RobotState s;
        s.ee_quat_b = {tf_ee.transform.rotation.w, tf_ee.transform.rotation.x, tf_ee.transform.rotation.y,
                        tf_ee.transform.rotation.z};
        // Compose through base frame (not world-fixed kGravityVecW directly)
        // so tilt from the base itself (ramps, uneven terrain) is captured --
        // same fix as rl_inference_node::update_ee_state_from_tf().
        s.ee_projected_gravity_b = kairos_math::quat_apply_inverse(s.ee_quat_b, base_projected_gravity_b_);

        // Reuse the actual observation accessor instead of reading the struct
        // field directly, so this stays in sync with robot_observations.cpp.
        const std::vector<double> proj_grav = obs::ee_projected_gravity(s);

        // When level, gravity expressed in the ee frame equals -up_axis (the
        // tray's local "down"). Tilt = angle between measured and ideal down.
        const double dot_with_ideal_down = -proj_grav[kUpAxisIndex];
        const double clamped = std::clamp(dot_with_ideal_down, -1.0, 1.0);

        sample_out.tilt_deg = std::acos(clamped) * 180.0 / kPi;
        sample_out.proj_grav = {proj_grav[0], proj_grav[1], proj_grav[2]};
        sample_out.ee_pos_b = {tf_ee.transform.translation.x, tf_ee.transform.translation.y,
                                tf_ee.transform.translation.z};
        return true;
    }

    void publish_current_command() {
        const auto& phase = phases_[phase_idx_];
        pub_cmd_vel_->publish(phase.cmd);
        pub_tray_target_->publish(phase.tray_offset);
    }

    void publish_fixed_arm_target() {
        trajectory_msgs::msg::JointTrajectory msg;
        msg.joint_names.assign(kairos_deploy::kArmJointNames.begin(), kairos_deploy::kArmJointNames.end());
        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = fixed_arm_joint_positions_;
        point.time_from_start = rclcpp::Duration(std::chrono::duration<double>(0.1));
        msg.points.push_back(point);
        pub_arm_traj_->publish(msg);
    }

    void tick() {
        if (done_) return;

        TiltSample sample;
        if (!measure_tilt(sample)) return; 

        if (!center_measured_) {
            ee_center_b_ = sample.ee_pos_b;
            center_measured_ = true;
            
            start_time_ = this->get_clock()->now();
            phase_start_time_ = start_time_;
            RCLCPP_INFO(this->get_logger(), "Centro medido: (%.3f, %.3f, %.3f). Empezando evaluación.", 
                        ee_center_b_[0], ee_center_b_[1], ee_center_b_[2]);
            return;
        }

        publish_current_command();
        if (hold_arm_fixed_) publish_fixed_arm_target();

        const auto& offset = phases_[phase_idx_].tray_offset;
        double err_x = sample.ee_pos_b[0] - (ee_center_b_[0] + offset.x);
        double err_y = sample.ee_pos_b[1] - (ee_center_b_[1] + offset.y);
        double err_z = sample.ee_pos_b[2] - (ee_center_b_[2] + offset.z);
        double ef_pos_error = std::sqrt(err_x*err_x + err_y*err_y + err_z*err_z);

        phase_stats_.add(sample.tilt_deg);
        overall_stats_.add(sample.tilt_deg);

        const double t_sec = (this->get_clock()->now() - start_time_).seconds();
        
        // Llamada actualizada sin argumentos GT
        log_row(t_sec, phases_[phase_idx_].cmd, sample, ef_pos_error);

        const double elapsed = (this->get_clock()->now() - phase_start_time_).seconds();
        if (elapsed >= phases_[phase_idx_].duration_sec) {
            report_phase();
            advance_phase();
        }
    }

    void report_phase() {
        RCLCPP_INFO(this->get_logger(), "  -> '%s' tilt: mean=%.2f deg, std=%.2f deg, max=%.2f deg (n=%d)",
                    phases_[phase_idx_].name.c_str(), phase_stats_.mean(), phase_stats_.stddev(),
                    phase_stats_.max_deg, phase_stats_.count);
    }

    void advance_phase() {
        ++phase_idx_;
        phase_stats_ = TiltStats{};
        if (phase_idx_ >= phases_.size()) {
            finish();
            return;
        }
        phase_start_time_ = this->get_clock()->now();
        const auto& p = phases_[phase_idx_];
        RCLCPP_INFO(this->get_logger(), "Phase [%zu/%zu] '%s' (%.1fs, vx=%.2f vy=%.2f wz=%.2f)", phase_idx_ + 1,
                    phases_.size(), p.name.c_str(), p.duration_sec, p.cmd.linear.x, p.cmd.linear.y,
                    p.cmd.angular.z);
    }

    void finish() {
        done_ = true;
        // Leave the base stopped.
        pub_cmd_vel_->publish(make_twist(0.0, 0.0, 0.0));
        RCLCPP_INFO(this->get_logger(),
                    "Evaluation complete (run_label='%s'). Overall tilt: mean=%.2f deg, std=%.2f deg, "
                    "max=%.2f deg (n=%d). Log: %s",
                    run_label_.c_str(), overall_stats_.mean(), overall_stats_.stddev(), overall_stats_.max_deg,
                    overall_stats_.count, log_path_.c_str());
        log_file_.close();
        timer_->cancel();
    }

    std::string base_frame_;
    std::string ee_frame_;
    std::string run_label_;
    bool hold_arm_fixed_ = false;
    std::vector<double> fixed_arm_joint_positions_;
    std::array<double, 3> base_projected_gravity_b_{0.0, 0.0, -1.0};
    bool imu_ready_ = false;

    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr pub_tray_target_;
    std::array<double, 3> ee_center_b_{0.0, 0.0, 0.0};
    bool center_measured_ = false;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_cmd_vel_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr pub_arm_traj_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::vector<Phase> phases_;
    size_t phase_idx_ = 0;
    rclcpp::Time start_time_;
    rclcpp::Time phase_start_time_;
    TiltStats phase_stats_;
    TiltStats overall_stats_;
    bool done_ = false;

    std::ofstream log_file_;
    std::string log_path_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrayTiltEvalNode>());
    rclcpp::shutdown();
    return 0;
}