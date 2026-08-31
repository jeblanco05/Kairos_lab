"""Custom observation functions for the Kairos+ whole-body control task.

They are added here because IsaacLab only calculates `projected_gravity_b`
and relative poses for the Articulation root by default. For this project, we need
the same information but referenced to an arbitrary body (the end effector / tray,
`rg6_tcp_link`), as well as privileged observations (payload mass) that only make
sense in simulation.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def command_position(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Only the position part (3,) of a pose command (e.g., `tray_pose`)."""
    command = env.command_manager.get_command(command_name)
    return command[:, :3]

def ee_pose_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """End-effector pose (tray) expressed in the robot base frame.

    Returns a tensor (num_envs, 7) = [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z].
    This is a quantity that can be computed from the robot's kinematics, so it is not
    considered privileged.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ee_body_id = asset_cfg.body_ids[0]

    ee_pos_w = asset.data.body_pos_w[:, ee_body_id]
    ee_quat_w = asset.data.body_quat_w[:, ee_body_id]

    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, ee_pos_w, ee_quat_w
    )
    return torch.cat((ee_pos_b, ee_quat_b), dim=-1)


def ee_position_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Only the position (3,) of `ee_pose_b`. It is separated from the orientation because the noise
    of a position sensor (m) and that of an orientation sensor (quaternion, dimensionless) have
    very different scales, and `ObsTerm.noise` only accepts a single range per term."""
    asset: Articulation = env.scene[asset_cfg.name]
    ee_body_id = asset_cfg.body_ids[0]
    ee_pos_b, _ = subtract_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, asset.data.body_pos_w[:, ee_body_id]
    )
    return ee_pos_b


def ee_orientation_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Only the orientation (quaternion, 4) of `ee_pose_b`. See `ee_position_b`."""
    asset: Articulation = env.scene[asset_cfg.name]
    ee_body_id = asset_cfg.body_ids[0]
    _, ee_quat_b = subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_pos_w[:, ee_body_id],
        asset.data.body_quat_w[:, ee_body_id],
    )
    return ee_quat_b


def ee_position_command_error_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Position error (command - actual) between the tray target and the end effector,
    both expressed in the robot base frame. This gives the agent a direct signal of the
    correction error to minimize.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ee_body_id = asset_cfg.body_ids[0]

    command = env.command_manager.get_command(command_name)  # (num_envs, 7): pos(3) + quat(4)
    des_pos_b = command[:, :3]

    ee_pos_b, _ = subtract_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, asset.data.body_pos_w[:, ee_body_id]
    )
    return des_pos_b - ee_pos_b


def body_projected_gravity_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Gravity vector projected into the local frame of an arbitrary Articulation body."""

    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]

    body_quat_w = asset.data.body_quat_w[:, body_id]
    return quat_apply_inverse(body_quat_w, asset.data.GRAVITY_VEC_W)


def ee_tilt_angle(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Angle (rad) between the local "up" axis of the body and the world vertical.

    0 when perfectly level, pi/2 when tilted sideways.
    """
    projected_gravity_b = body_projected_gravity_b(env, asset_cfg)
    up = torch.tensor(up_axis, device=projected_gravity_b.device, dtype=projected_gravity_b.dtype)
    up = up / torch.linalg.norm(up)
    # Level = proj_grav_b == -up => cos(angle) = -proj_grav_b . up
    cos_angle = -torch.sum(projected_gravity_b * up, dim=-1)
    return torch.acos(cos_angle.clamp(-1.0, 1.0))


def body_mass(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Privileged observation: current mass of a rigid body (e.g., payload mass on the end effector).
    It only makes sense in simulation (critic group), since the real robot does not have a direct
    measurement for it.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses().to(env.device)  # (num_envs, num_bodies)
    return masses[:, asset_cfg.body_ids]


def body_mass_relative(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Same as `body_mass`, but as a deviation from the nominal (default) body mass.
    This is usually more informative/easier to normalize for the network than the absolute mass.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses().to(env.device)[:, asset_cfg.body_ids]
    default_masses = asset.data.default_mass.to(env.device)[:, asset_cfg.body_ids]
    return masses - default_masses


def ee_velocity_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Privileged observation: linear and angular velocity (world frame) of the end effector.

    The real robot does not have a direct sensor on the wrist/tray at this precision, so this is
    treated as privileged information for the critic.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    lin_vel = asset.data.body_lin_vel_w[:, body_id]
    ang_vel = asset.data.body_ang_vel_w[:, body_id]
    return torch.cat((lin_vel, ang_vel), dim=-1)