"""Custom reward functions for the Kairos+ whole-body control task.

IsaacLab provides generic velocity tracking and `flat_orientation_l2` rewards for the robot root,
but there is nothing ready-made for:
  1) tracking a target pose for a specific body other than the root (the tray), and
  2) penalizing the tilt of that body with respect to gravity (tray balance).

Both functions reuse the kinematics already computed in `tasks.mdp.observations`.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from .observations import body_projected_gravity_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_position_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Exponential reward for tracking the tray target position (command `tray_pose`).

    It follows the same pattern as `track_lin_vel_xy_yaw_frame_exp`: exp(-squared_error / std^2).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ee_body_id = asset_cfg.body_ids[0]

    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]

    ee_pos_b, _ = subtract_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, asset.data.body_pos_w[:, ee_body_id]
    )

    error = torch.sum(torch.square(des_pos_b - ee_pos_b), dim=1)
    return torch.exp(-error / std**2)

def ee_position_tracking_exp_bubble(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    tolerance: float = 0.05,  # Burbuja de tolerancia (ej. 8 cm)
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Exponential reward for tracking the tray target position with a deadzone bubble.
    If the tray is within 'tolerance' meters of the target, it receives maximum reward.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ee_body_id = asset_cfg.body_ids[0]

    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]

    ee_pos_b, _ = subtract_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, asset.data.body_pos_w[:, ee_body_id]
    )

    # 1. Calcular la distancia euclidiana lineal (L2 norm)
    distance = torch.norm(des_pos_b - ee_pos_b, dim=1)
    
    # 2. Aplicar la burbuja: restamos la tolerancia y limitamos a 0.
    # Si la distancia es menor que la tolerancia, el resultado es 0 (error nulo).
    adjusted_distance = torch.clamp(distance - tolerance, min=0.0)
    
    # 3. Elevar al cuadrado el error ajustado para la función exponencial
    error_sq = torch.square(adjusted_distance)
    
    return torch.exp(-error_sq / std**2)


def ee_flat_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Penalizes (L2) the tilt of the tray: squared distance between the gravity vector projected
    into the end-effector frame and `-up_axis` (the vector that should point out if the tray is
    perfectly level). It is 0 when horizontal. Designed for use with a NEGATIVE weight,
    similar to `flat_orientation_l2` for the base.
    """
    projected_gravity_b = body_projected_gravity_b(env, asset_cfg)
    up = torch.tensor(up_axis, device=projected_gravity_b.device, dtype=projected_gravity_b.dtype)
    up = up / torch.linalg.norm(up)
    deviation = projected_gravity_b + up  # == 0 when perfectly level
    return torch.sum(torch.square(deviation), dim=-1)


def ee_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Penalizes the linear + angular velocity of the end effector to encourage a stable tray
    (without oscillations), not just a good average orientation. Small NEGATIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    lin_vel = asset.data.body_lin_vel_w[:, body_id]
    ang_vel = asset.data.body_ang_vel_w[:, body_id]
    return torch.sum(torch.square(lin_vel), dim=1) + torch.sum(torch.square(ang_vel), dim=1)