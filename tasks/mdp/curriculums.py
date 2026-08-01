from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def terrain_distance_curriculum(
    env, # Asumiendo ManagerBasedRLEnv
    env_ids: torch.Tensor,
    distance_threshold: float,
    asset_cfg = None, # Puedes mantener tus tipos (SceneEntityCfg)
) -> torch.Tensor:
    """
    Evaluates the distance traveled by the robot from its initial origin.
    Updates the terrain level internally and returns the mean update for logging.
    """
    if asset_cfg is None:
        robot = env.scene["robot"]
    else:
        robot = env.scene[asset_cfg.name]
    
    # Calculate Euclidean distance on the XY plane
    current_pos = robot.data.root_pos_w[env_ids, :2]
    origin_pos = env.scene.env_origins[env_ids, :2]
    distance_traveled = torch.norm(current_pos - origin_pos, dim=1)
    
    # 1. Creamos las MÁSCARAS BOOLEANAS (True/False)
    move_up = distance_traveled >= distance_threshold
    move_down = distance_traveled < 0.5
    
    # 2. Las pasamos directamente al actualizador de terrenos (¡EL ARREGLO ESTÁ AQUÍ!)
    if hasattr(env.scene, "terrain"):
        env.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        
    # 3. Mantenemos el cálculo de level_update solo para los logs/gráficas
    level_update = torch.zeros_like(env_ids, dtype=torch.long)
    level_update[move_up] = 1
    level_update[move_down] = -1
    
    # Reducimos a escalar para que el Logger no colapse
    mean_level_update = torch.mean(level_update.float())
    
    return mean_level_update

def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)

def ee_target_range_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "tray_pose",
    reward_term_name: str = "ee_position_tracking",
    axis_deltas: dict[str, tuple[float, float]] | None = None,
) -> torch.Tensor:
    """Expande progresivamente el rango de posiciones objetivo de la bandeja (pos_x, pos_y,
    pos_z del comando `tray_pose`) a medida que el agente sigue bien el objetivo actual.
    """
    if axis_deltas is None:
        axis_deltas = {"pos_x": (-0.05, 0.05), "pos_y": (-0.05, 0.05), "pos_z": (-0.05, 0.05)}
 
    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges
 
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
 
    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            for axis, (delta_lower, delta_upper) in axis_deltas.items():
                lower, upper = getattr(ranges, axis)
                limit_lower, limit_upper = getattr(limit_ranges, axis)
                new_lower = min(max(lower + delta_lower, limit_lower), upper)
                new_upper = max(min(upper + delta_upper, limit_upper), lower)
                setattr(ranges, axis, (new_lower, new_upper))
 
    return torch.tensor(ranges.pos_z[1], device=env.device)