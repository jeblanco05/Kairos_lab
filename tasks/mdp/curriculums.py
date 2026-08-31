from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def terrain_distance_curriculum(
    env, # Assuming ManagerBasedRLEnv
    env_ids: torch.Tensor,
    distance_threshold: float,
    asset_cfg = None,
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
    
    # 1. We create the BOOLEAN MASKS (True/False)
    move_up = distance_traveled >= distance_threshold
    move_down = distance_traveled < 0.5
    
    # 2. We pass them directly to the terrain updater (THE FIX IS HERE!)
    if hasattr(env.scene, "terrain"):
        env.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        
    # 3. We keep the level_update calculation only for logs/plots
    level_update = torch.zeros_like(env_ids, dtype=torch.long)
    level_update[move_up] = 1
    level_update[move_down] = -1
    
    # Reduce to a scalar so the Logger does not collapse
    mean_level_update = torch.mean(level_update.float())
    
    return mean_level_update

def terrain_curriculum(
    env, 
    env_ids: torch.Tensor,
    vel_reward_name: str = "track_lin_vel_xy",
    ee_flat_penalty_name: str = "ee_flat_orientation",
    # Umbrales
    vel_threshold_ratio: float = 0.8, # 80% de éxito en velocidad
    flat_threshold: float = -0.2,     # Tolerancia a la inclinación
) -> torch.Tensor:
    """
    Curriculum simplificado: Solo evalúa si corre bien y la bandeja está recta.
    Ignora torques, velocidades del brazo, etc.
    """
    episode_length = env.max_episode_length_s
    
    # Extraer SOLO las dos métricas críticas
    vel_rewards = env.reward_manager._episode_sums[vel_reward_name][env_ids] / episode_length
    ee_flat_penalties = env.reward_manager._episode_sums[ee_flat_penalty_name][env_ids] / episode_length
    
    vel_weight = env.reward_manager.get_term_cfg(vel_reward_name).weight
    
    # 1. ¿Logró el objetivo general? (Corre bien Y bandeja recta)
    task_success = (vel_rewards > vel_weight * vel_threshold_ratio) & (ee_flat_penalties > flat_threshold)
    
    # 2. ¿Fracasó catastróficamente? (No se mueve O volcó la bandeja)
    task_failure = (vel_rewards < vel_weight * 0.4) | (ee_flat_penalties < flat_threshold * 2.0)
    
    # Aplicar currículo
    if hasattr(env.scene, "terrain"):
        env.scene.terrain.update_env_origins(env_ids, move_up=task_success, move_down=task_failure)
        
    level_update = torch.zeros_like(env_ids, dtype=torch.long)
    level_update[task_success] = 1
    level_update[task_failure] = -1
    
    return torch.mean(level_update.float())

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
    """Progressively expands the target position range of the tray (`pos_x`, `pos_y`,
    `pos_z` of the `tray_pose` command) as the agent follows the current target correctly.
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