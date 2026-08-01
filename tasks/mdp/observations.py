"""Funciones de observación personalizadas para la tarea de Whole-Body Control del Kairos+.

Se añaden aquí porque IsaacLab, de forma nativa, sólo calcula `projected_gravity_b`
y las poses relativas para el *root* del Articulation. Para este proyecto necesitamos
lo mismo pero referido a un cuerpo arbitrario (el efector final / bandeja, `rg6_tcp_link`),
así como observaciones privilegiadas (masa de la carga) que sólo tienen sentido en simulación.
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
    """Sólo la parte de posición (3,) de un comando de pose (p. ej. `tray_pose`).
 
    Usar en vez de `mdpIL.generated_commands` cuando el resto del comando (orientación)
    es constante -- por ejemplo si `roll=pitch=yaw=(0,0)`, la parte de orientación del
    comando es siempre el mismo cuaternión identidad y no aporta información a la política.
    """
    command = env.command_manager.get_command(command_name)
    return command[:, :3]

def ee_pose_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Pose del efector final (bandeja) expresada en el frame de la base del robot.

    Devuelve un tensor (num_envs, 7) = [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z].
    Es una magnitud calculable con la cinemática directa del robot, por lo que no se
    considera privilegiada (también estaría disponible en el robot real).
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
    """Sólo la posición (3,) de `ee_pose_b`. Se separa de la orientación porque el ruido
    de un sensor de posición (m) y el de una orientación (cuaternión, adimensional) tienen
    escalas muy distintas, y `ObsTerm.noise` sólo admite un único rango por término."""
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
    """Sólo la orientación (cuaternión, 4) de `ee_pose_b`. Ver `ee_position_b`."""
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
    """Error de posición (comando - actual) entre el objetivo de la bandeja y el efector final,
    ambos expresados en el frame de la base. Da al agente una señal directa del error a corregir.
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
    """Vector de gravedad proyectado en el frame de un cuerpo arbitrario del Articulation.

    Es el equivalente de `mdp.projected_gravity` (que sólo funciona para el root) pero
    aplicado a la bandeja. Con orientación perfectamente horizontal (roll = pitch = 0
    respecto a gravedad) el resultado es aprox. (0, 0, -1); cualquier inclinación de la
    bandeja aparece como componentes x/y distintas de cero, que es justo la señal que la
    política necesita para mantener el equilibrio.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]

    body_quat_w = asset.data.body_quat_w[:, body_id]
    return quat_apply_inverse(body_quat_w, asset.data.GRAVITY_VEC_W)

def ee_tilt_angle(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Ángulo (rad) entre el eje "arriba" local del cuerpo y la vertical del mundo.

    0 cuando está perfectamente nivelado, pi/2 cuando está de canto. Se apoya en
    `body_projected_gravity_b`; ver la nota sobre `up_axis` en `ee_bad_orientation`.
    """
    projected_gravity_b = body_projected_gravity_b(env, asset_cfg)
    up = torch.tensor(up_axis, device=projected_gravity_b.device, dtype=projected_gravity_b.dtype)
    up = up / torch.linalg.norm(up)
    # Nivelado <=> proj_grav_b == -up  =>  cos(angulo) = -proj_grav_b . up
    cos_angle = -torch.sum(projected_gravity_b * up, dim=-1)
    return torch.acos(cos_angle.clamp(-1.0, 1.0))


def body_mass(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Observación privilegiada: masa actual de un cuerpo rígido (p. ej. masa de la carga
    acoplada al efector final). Sólo tiene sentido en simulación (grupo `critic`), ya que
    el robot real no dispone de esta medida directa.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses().to(env.device)  # (num_envs, num_bodies)
    return masses[:, asset_cfg.body_ids]


def body_mass_relative(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Igual que `body_mass`, pero como desviación respecto a la masa nominal (por defecto)
    del cuerpo. Suele ser más informativa/fácil de normalizar para la red que la masa absoluta.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses().to(env.device)[:, asset_cfg.body_ids]
    default_masses = asset.data.default_mass.to(env.device)[:, asset_cfg.body_ids]
    return masses - default_masses


def ee_velocity_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
) -> torch.Tensor:
    """Observación privilegiada: velocidad lineal y angular (mundo) del efector final.

    En el robot real no se dispone de un sensor directo en la muñeca/bandeja con esta
    precisión, por lo que se trata como información privilegiada para el crítico.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    lin_vel = asset.data.body_lin_vel_w[:, body_id]
    ang_vel = asset.data.body_ang_vel_w[:, body_id]
    return torch.cat((lin_vel, ang_vel), dim=-1)