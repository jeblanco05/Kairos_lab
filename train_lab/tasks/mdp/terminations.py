from __future__ import annotations
import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import ee_tilt_angle, ee_position_command_error_b

def ee_bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Termination condition: the tray has tilted beyond `limit_angle` (rad)
    with respect to the horizontal. Same logic as `mdp.bad_orientation`, but applied to the
    end effector instead of the robot base.
    """
    tilt_angle = ee_tilt_angle(env, asset_cfg, up_axis)
    return tilt_angle > limit_angle

def ee_position_too_far(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Termination condition: the distance between the current tray position and the target
    position of the command (`command_name`) exceeds `threshold` meters.
    """
    error = ee_position_command_error_b(env, command_name, asset_cfg)
    distance = torch.norm(error, dim=-1)
    return distance > threshold