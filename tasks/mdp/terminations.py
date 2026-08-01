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
    """Condición de terminación: la bandeja se ha inclinado más allá de `limit_angle` (rad)
    respecto a la horizontal. Misma lógica que `mdp.bad_orientation`, pero referida al
    efector final en lugar de a la base del robot.

    `up_axis` es el eje del frame LOCAL de `asset_cfg` que apunta "hacia arriba" cuando la
    bandeja está nivelada. Es una propiedad fija de cómo está definido ese frame en el
    URDF/USD (no depende de la pose actual del brazo) — verifícalo mirando los ejes del
    link en Isaac Sim o en el URDF antes de asumir que es (0,0,1); en gripper/TCP frames
    lo habitual es que el eje "arriba" NO sea Z (Z suele ser la dirección de aproximación).
    """
    tilt_angle = ee_tilt_angle(env, asset_cfg, up_axis)
    return tilt_angle > limit_angle

def ee_position_too_far(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Condición de terminación: la distancia entre la posición actual de la bandeja y la
    posición objetivo del comando (`command_name`) supera `threshold` metros.
 
    OJO con el umbral: el comando `tray_pose` se re-muestrea cada `resampling_time_range`
    segundos con un salto BRUSCO (no una rampa), así que justo después de cada resample el
    error puede dar un pico grande de golpe aunque la política esté funcionando bien. El
    `threshold` debe ser holgado respecto al tamaño máximo de ese salto (la diagonal de la
    caja `limit_ranges` del comando), o terminarás episodios "buenos" solo por mala suerte
    en el momento del resample.
    """
    error = ee_position_command_error_b(env, command_name, asset_cfg)
    distance = torch.norm(error, dim=-1)
    return distance > threshold