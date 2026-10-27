from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def terrain_curriculum_arm(
    env, 
    env_ids: torch.Tensor,
    ee_tracking_reward_name: str = "ee_position_tracking",
    ee_flat_penalty_name: str = "ee_flat_orientation",
    tracking_threshold_ratio: float = 0.7, # 70% de éxito en el tracking del brazo
    flat_threshold: float = -0.15,          # 10º Tolerancia a la inclinación promedio por paso
) -> torch.Tensor:
    """
    Curriculum evaluado exclusivamente en el desempeño del brazo y la supervivencia.
    Avanza de nivel si el robot sobrevive el episodio estabilizando bien la bandeja.
    """
    # 1. Supervivencia: ¿Llegó al final del episodio sin activar terminaciones fatales?
    max_steps = env.max_episode_length
    steps_lived = env.episode_length_buf[env_ids].float()
    
    # Consideramos que sobrevivió si llegó a menos de 5 pasos del límite máximo
    survived_long_enough = steps_lived >= (max_steps - 20)
    
    # Evitar división por cero si mueren en el paso 0
    steps_lived = torch.clamp(steps_lived, min=1.0)
    
    # 2. Extraer métricas normalizadas por los pasos que realmente vivió el agente
    tracking_rewards = env.reward_manager._episode_sums[ee_tracking_reward_name][env_ids] / env.max_episode_length_s

    flat_penalties = (env.reward_manager._episode_sums[ee_flat_penalty_name][env_ids]/ env.max_episode_length_s)
    
    tracking_weight = env.reward_manager.get_term_cfg(ee_tracking_reward_name).weight
    
    # 3. Éxito: Sobrevivió + buen tracking del brazo + bandeja mayormente plana
    task_success = (
        survived_long_enough & 
        (tracking_rewards > tracking_weight * tracking_threshold_ratio) & 
        (flat_penalties > flat_threshold)
    )
    
    # 4. Fracaso: Murió muy rápido (ej. volcó en la primera rampa) O inclinación catastrófica
    task_failure = (steps_lived < max_steps // 2) | (flat_penalties < flat_threshold * 3.0)
    
    if hasattr(env.scene, "terrain"):
        env.scene.terrain.update_env_origins(env_ids, move_up=task_success, move_down=task_failure)
        
    level_update = torch.zeros_like(env_ids, dtype=torch.long)
    level_update[task_success] = 1
    level_update[task_failure] = -1
    
    return torch.mean(level_update.float())
    

def terrain_curriculum(
    env, 
    env_ids: torch.Tensor,
    ee_flat_penalty_name: str = "ee_flat_orientation",
    # Umbrales basados en el ERROR PURO (valores positivos). Más cerca de 0 es mejor.
    max_error_success: float = 0.05,  # Error máximo permitido para subir de nivel (muy horizontal)
    min_error_failure: float = 0.3,   # Si el error supera esto, baja de nivel (muy inclinado)
) -> torch.Tensor:
    """
    Curriculum basado puramente en el error físico de inclinación de la bandeja.
    Es independiente del peso (weight) configurado en la recompensa.
    """
    episode_length = env.max_episode_length_s
    
    # 1. Obtener la penalización acumulada (valor ponderado negativo)
    weighted_penalties = env.reward_manager._episode_sums[ee_flat_penalty_name][env_ids]
    
    # 2. Obtener el peso configurado (ej. -1.0, -5.0, etc.)
    weight = env.reward_manager.get_term_cfg(ee_flat_penalty_name).weight
    
    # 3. Calcular el error RAW promedio por paso (positivo).
    # Al dividir entre el peso negativo, volvemos a la escala original de la función (0 = perfecto)
    raw_avg_error = (weighted_penalties / episode_length) / weight
    
    # 4. ¿Logró mantenerla estable? (El error es menor que nuestra tolerancia)
    task_success = raw_avg_error < max_error_success
    
    # 5. ¿Fracasó? (El error promedio es demasiado alto)
    task_failure = raw_avg_error > min_error_failure
    
    # Aplicar currículo modificando los orígenes del terreno
    if hasattr(env.scene, "terrain"):
        env.scene.terrain.update_env_origins(env_ids, move_up=task_success, move_down=task_failure)
        
    # Calcular estadística para el registro (log)
    level_update = torch.zeros_like(env_ids, dtype=torch.long)
    level_update[task_success] = 1
    level_update[task_failure] = -1
    
    return torch.mean(level_update.float())

# def lin_vel_cmd_levels(
#     env: ManagerBasedRLEnv,
#     env_ids: Sequence[int],
#     reward_term_name: str = "track_lin_vel_xy",
# ) -> torch.Tensor:
#     command_term = env.command_manager.get_term("base_velocity")
#     ranges = command_term.cfg.ranges
#     limit_ranges = command_term.cfg.limit_ranges

#     reward_term = env.reward_manager.get_term_cfg(reward_term_name)
#     reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

#     if env.common_step_counter % env.max_episode_length == 0:
#         if reward > reward_term.weight * 0.8:
#             delta_command = torch.tensor([-0.1, 0.1], device=env.device)
#             ranges.lin_vel_x = torch.clamp(
#                 torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
#                 limit_ranges.lin_vel_x[0],
#                 limit_ranges.lin_vel_x[1],
#             ).tolist()
#             ranges.lin_vel_y = torch.clamp(
#                 torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
#                 limit_ranges.lin_vel_y[0],
#                 limit_ranges.lin_vel_y[1],
#             ).tolist()

#     return torch.tensor(ranges.lin_vel_x[1], device=env.device)

def lin_vel_cmd_levels(
    env,
    env_ids,
    orientation_threshold=0.01,
):
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limits = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg("ee_flat_orientation")

    # Reward acumulado ponderado -> error geométrico medio
    orientation_error = (
        torch.mean(env.reward_manager._episode_sums["ee_flat_orientation"][env_ids])/ env.max_episode_length_s/ abs(reward_term.weight))

    if env.common_step_counter % env.max_episode_length == 0:
        if orientation_error < orientation_threshold:
            delta = 0.05

            ranges.lin_vel_x = [
                max(limits.lin_vel_x[0],
                    ranges.lin_vel_x[0] - delta),
                min(limits.lin_vel_x[1],
                    ranges.lin_vel_x[1] + delta),
            ]

            ranges.lin_vel_y = [
                max(limits.lin_vel_y[0],
                    ranges.lin_vel_y[0] - delta),
                min(limits.lin_vel_y[1],
                    ranges.lin_vel_y[1] + delta),
            ]

    return torch.tensor(
        ranges.lin_vel_x[1],
        device=env.device,
    )

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