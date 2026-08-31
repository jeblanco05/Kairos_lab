import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.utils.math import subtract_frame_transforms

def center_command_ranges_on_body(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    range_offsets: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    limit_offsets: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rg6_tcp_link"),
):
    """Startup event (use with `mode="startup"`): sets the `ranges` and `limit_ranges` of a pose
    command (e.g., `tray_pose`) as offsets relative to the actual position of `asset_cfg`
    with respect to the base, measured in the initial robot pose (the one defined by
    `ArticulationCfg.init_state`).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
 
    asset.write_joint_state_to_sim(asset.data.default_joint_pos, asset.data.default_joint_vel)
    env.sim.forward()
 
    ee_pos_b, _ = subtract_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, asset.data.body_pos_w[:, body_id]
    )

    center = ee_pos_b[0].tolist()
 
    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges
 
    for axis, c, (r_min, r_max), (l_min, l_max) in zip(
        ("pos_x", "pos_y", "pos_z"), center, range_offsets, limit_offsets
    ):
        setattr(ranges, axis, (c + r_min, c + r_max))
        setattr(limit_ranges, axis, (c + l_min, c + l_max))
 
    print(
        f"[center_command_ranges_on_body] '{command_name}' centered at "
        f"{tuple(round(v, 3) for v in center)} (measured in the initial robot pose)"
    )

def custom_reset_kairos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    z_clearance: float = 0.5,
    yaw_range: tuple[float, float] = (-3.14159, 3.14159),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]

    env_origins = env.scene.env_origins[env_ids]
        
    # SPATIAL POSITION CALCULATION
    # Final height = terrain height + robot structural base height + extra clearance
    terrain_height = env_origins[:, 2]
    dynamic_clearance = torch.where(terrain_height < 0.0, z_clearance, z_clearance)
    
    # Start from the default state corrected in the world
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, 0] = env_origins[:, 0]
    root_state[:, 1] = env_origins[:, 1]
    
    # Modify the Z by adding the clearance to the USD default Z (avoids clipping)
    root_state[:, 2] += dynamic_clearance + terrain_height

    # Orientation: we compute a random yaw
    yaw = torch.rand(len(env_ids), device=asset.device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
    half_yaw = yaw * 0.5
    zeros = torch.zeros_like(yaw)
    orientations = torch.stack([torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)], dim=-1)
    
    # If the robot has a base orientation in the USD, the ideal approach would be to multiply quaternions,
    # but if you want a flat alignment with random yaw, this is the simplest option:
    root_state[:, 3:7] = orientations
 
    # Force absolute zero velocities
    root_state[:, 7:13] = 0.0
    
    # Write the complete clean structure to the simulation
    asset.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
 
    # Force joint reset to remove residual tensions
    default_joint_pos = asset.data.default_joint_pos[env_ids].clone()
    zero_joint_vel = torch.zeros_like(asset.data.default_joint_vel[env_ids])
    asset.write_joint_state_to_sim(default_joint_pos, zero_joint_vel, env_ids=env_ids)
    asset.set_joint_position_target(default_joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(zero_joint_vel, env_ids=env_ids)

def custom_reset_kairos_(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    z_clearance: float = 0.5,
    yaw_range: tuple[float, float] = (-3.14159, 3.14159),
    arm_joint_pos_range: tuple[float, float] = (-0.05, 0.05),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arm_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="arm_.*_joint"),
):
    asset: Articulation = env.scene[asset_cfg.name]
 
    env_origins = env.scene.env_origins[env_ids]
        
    # 3. DYNAMIC SPATIAL POSITION CALCULATION
    # Final height = terrain height + robot structural base height + extra clearance
    terrain_height = env_origins[:, 2]
    dynamic_clearance = torch.where(terrain_height < 0.0, z_clearance, z_clearance)
    
    # Start from the default state corrected in the world
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, 0] = env_origins[:, 0]
    root_state[:, 1] = env_origins[:, 1]
    
    # Modify the Z by adding the clearance to the USD default Z (avoids clipping)
    root_state[:, 2] += dynamic_clearance + terrain_height
 
    # Orientation: we compute a random yaw
    yaw = torch.rand(len(env_ids), device=asset.device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
    half_yaw = yaw * 0.5
    zeros = torch.zeros_like(yaw)
    orientations = torch.stack([torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)], dim=-1)
    
    # If the robot has a base orientation in the USD, the ideal approach would be to multiply quaternions,
    # but if you want a flat alignment with random yaw, this is the simplest option:
    root_state[:, 3:7] = orientations
 
    # Force absolute zero velocities
    root_state[:, 7:13] = 0.0
    
    # Write the complete clean structure to the simulation
    asset.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)
 
    # Force joint reset to remove residual tensions
    default_joint_pos = asset.data.default_joint_pos[env_ids].clone()
    zero_joint_vel = torch.zeros_like(asset.data.default_joint_vel[env_ids])
 
    # Small noise in the initial ARM pose (not on wheels or gripper) so that the policy does not
    # memorize the same starting posture every time; this helps sim-to-real transfer, where the
    # real initial pose is never exactly the nominal one.
    arm_joint_ids = arm_asset_cfg.joint_ids
    arm_noise = torch.empty(len(env_ids), len(arm_joint_ids), device=asset.device).uniform_(
        *arm_joint_pos_range
    )
    default_joint_pos[:, arm_joint_ids] += arm_noise
 
    asset.write_joint_state_to_sim(default_joint_pos, zero_joint_vel, env_ids=env_ids)
    asset.set_joint_position_target(default_joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(zero_joint_vel, env_ids=env_ids)
