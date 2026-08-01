"""Funciones de recompensa personalizadas para la tarea de Whole-Body Control del Kairos+.

IsaacLab trae recompensas genéricas de tracking de velocidad y de `flat_orientation_l2`
para el root del robot, pero no hay nada preparado de fábrica para:
  1) seguir una pose objetivo de un *cuerpo* concreto que no sea el root (la bandeja), y
  2) penalizar la inclinación de ESE cuerpo respecto a la gravedad (equilibrio de bandeja).

Ambas funciones reutilizan la cinemática ya calculada en `tasks.mdp.observations`.
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
    """Recompensa exponencial por seguir la posición objetivo de la bandeja (comando `tray_pose`).

    Sigue el mismo patrón que `track_lin_vel_xy_yaw_frame_exp`: exp(-error_cuadratico / std^2).
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


def ee_flat_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Penaliza (L2) la inclinación de la bandeja: distancia al cuadrado entre la gravedad
    proyectada en el frame del efector final y `-up_axis` (el vector que debería salir si la
    bandeja estuviera perfectamente nivelada). Vale 0 cuando está horizontal. Pensada para
    usarse con peso NEGATIVO, igual que `flat_orientation_l2` para la base.

    Ver la nota sobre `up_axis` en `observations.ee_bad_orientation`: es un eje del frame
    LOCAL del cuerpo, fijo según el URDF/USD, no necesariamente Z.
    """
    projected_gravity_b = body_projected_gravity_b(env, asset_cfg)
    up = torch.tensor(up_axis, device=projected_gravity_b.device, dtype=projected_gravity_b.dtype)
    up = up / torch.linalg.norm(up)
    deviation = projected_gravity_b + up  # == 0 cuando está nivelado
    return torch.sum(torch.square(deviation), dim=-1)


def ee_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Penaliza la velocidad lineal + angular del efector final, para fomentar una bandeja
    estable (sin oscilaciones), no sólo bien orientada en promedio. Peso NEGATIVO pequeño.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    lin_vel = asset.data.body_lin_vel_w[:, body_id]
    ang_vel = asset.data.body_ang_vel_w[:, body_id]
    return torch.sum(torch.square(lin_vel), dim=1) + torch.sum(torch.square(ang_vel), dim=1)