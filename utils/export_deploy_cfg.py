import numpy as np
import os
import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict
from isaaclab.utils.string import resolve_matching_names


def format_value(x):
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, list):
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    else:
        return x


def export_deploy_cfg(env: ManagerBasedRLEnv, obs_groups, log_dir):
    asset: Articulation = env.scene["robot"]
    joint_sdk_names = asset.data.joint_names
    joint_ids_map = list(range(len(joint_sdk_names)))

    cfg = {}  # noqa: SIM904
    cfg["joint_ids_map"] = joint_ids_map
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation
    stiffness = np.zeros(len(joint_sdk_names))
    stiffness[joint_ids_map] = asset.data.default_joint_stiffness[0].detach().cpu().numpy().tolist()
    cfg["stiffness"] = stiffness.tolist()
    damping = np.zeros(len(joint_sdk_names))
    damping[joint_ids_map] = asset.data.default_joint_damping[0].detach().cpu().numpy().tolist()
    cfg["damping"] = damping.tolist()
    cfg["default_joint_pos"] = asset.data.default_joint_pos[0].detach().cpu().numpy().tolist()

    # --- commands ---
    cfg["commands"] = {}
    if hasattr(env.cfg.commands, "base_velocity"):  # some environments do not have base_velocity command
        cfg["commands"]["base_velocity"] = {}
        if hasattr(env.cfg.commands.base_velocity, "limit_ranges"):
            ranges = env.cfg.commands.base_velocity.limit_ranges.to_dict()
        else:
            ranges = env.cfg.commands.base_velocity.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            ranges[item_name] = list(ranges[item_name])
        cfg["commands"]["base_velocity"]["ranges"] = ranges

    # --- actions ---
    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    for action_name, action_term in action_terms:
        term_cfg = action_term.cfg.copy()
        
        # 1. Comprobar si la acción tiene 'scale' antes de modificarlo
        if hasattr(term_cfg, "scale") and term_cfg.scale is not None:
            if isinstance(term_cfg.scale, float):
                term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
            else:  # dict
                term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()

        # 2. Comprobar si la acción tiene 'clip' antes de modificarlo
        if hasattr(term_cfg, "clip") and term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()

        if action_name in ["JointPositionAction", "JointVelocityAction", "base_velocity", "arm_position"]:
            if hasattr(term_cfg, "use_default_offset") and term_cfg.use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        # clean cfg
        term_cfg = term_cfg.to_dict()

        keys_to_remove = ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]
        for key in keys_to_remove:
            term_cfg.pop(key, None)
        cfg["actions"][action_name] = term_cfg

        if hasattr(action_term, "_joint_ids"):
            if action_term._joint_ids == slice(None):
                cfg["actions"][action_name]["joint_ids"] = None
            else:
                cfg["actions"][action_name]["joint_ids"] = action_term._joint_ids

    # --- observations ---
    cfg["observations"] = {}
    
    # Extract dynamically which groups the policy uses
    if hasattr(obs_groups, "get"):
        policy_groups = obs_groups.get("policy", ["policy"])
    else:
        # Si no está definido en el agente, asumimos el grupo estándar
        policy_groups = ["policy"]
    
    # Save the order of the groups so the C++ deployment knows how to concatenate them
    cfg["policy_input_groups"] = policy_groups
    
    for group_name in policy_groups:
        if group_name not in env.observation_manager.active_terms:
            continue
            
        # Initialize a nested dictionary for this specific group
        cfg["observations"][group_name] = {}
            
        obs_names = env.observation_manager.active_terms[group_name]
        obs_cfgs = env.observation_manager._group_obs_term_cfgs[group_name]
        obs_terms = zip(obs_names, obs_cfgs)
        
        for obs_name, obs_cfg in obs_terms:
            #target_name = "keyboard_velocity_commands" if obs_name == "velocity_commands" else obs_name
            obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
            term_cfg = obs_cfg.copy()
            
            if term_cfg.scale is not None:
                scale = term_cfg.scale.detach().cpu().numpy().tolist()
                if isinstance(scale, float):
                    term_cfg.scale = [scale for _ in range(obs_dims[1])]
                else:
                    term_cfg.scale = scale
            else:
                term_cfg.scale = [1.0 for _ in range(obs_dims[1])]
                
            if term_cfg.clip is not None:
                term_cfg.clip = list(term_cfg.clip)
            if term_cfg.history_length == 0:
                term_cfg.history_length = 1

            # Clean cfg
            term_cfg = term_cfg.to_dict()
            for _ in ["func", "modifiers", "noise", "flatten_history_dim"]:
                if _ in term_cfg:
                    del term_cfg[_]
            
            # Save the observation configuration INSIDE its corresponding group
            cfg["observations"][group_name][obs_name] = term_cfg

    # --- save config file ---
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    cfg = format_value(cfg)
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
