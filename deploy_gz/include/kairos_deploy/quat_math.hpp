#pragma once
// Small quaternion utilities shared by the ROS2 deployment nodes
// (rl_inference_node.cpp, tray_tilt_eval_node.cpp). Kept dependency-free
// (no tf2::Quaternion / Eigen) so it works directly on plain
// array<double,3>/array<double,4> data coming from TF/IMU messages.
//
// Quaternion convention: (w, x, y, z).

#include <array>

namespace kairos_math {

// World-frame gravity direction (unit vector, pointing down).
constexpr std::array<double, 3> kGravityVecW = {0.0, 0.0, -1.0};

inline std::array<double, 3> cross3(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// Rotates vector v by quaternion q.
inline std::array<double, 3> quat_apply(const std::array<double, 4>& q, const std::array<double, 3>& v) {
    const double w = q[0];
    const std::array<double, 3> u = {q[1], q[2], q[3]};
    const std::array<double, 3> uxv = cross3(u, v);
    const std::array<double, 3> uxuxv = cross3(u, uxv);
    return {v[0] + 2.0 * w * uxv[0] + 2.0 * uxuxv[0], v[1] + 2.0 * w * uxv[1] + 2.0 * uxuxv[1],
            v[2] + 2.0 * w * uxv[2] + 2.0 * uxuxv[2]};
}

// Rotates vector v by the inverse (conjugate) of quaternion q.
inline std::array<double, 3> quat_apply_inverse(const std::array<double, 4>& q, const std::array<double, 3>& v) {
    const double w = q[0];
    const std::array<double, 3> u = {-q[1], -q[2], -q[3]};
    const std::array<double, 3> uxv = cross3(u, v);
    const std::array<double, 3> uxuxv = cross3(u, uxv);
    return {v[0] + 2.0 * w * uxv[0] + 2.0 * uxuxv[0], v[1] + 2.0 * w * uxv[1] + 2.0 * uxuxv[1],
            v[2] + 2.0 * w * uxv[2] + 2.0 * uxuxv[2]};
}

}  // namespace kairos_math
