"""
eval_policy.py — Simple evaluation script for a trained Kairos+ WBC policy.

A stripped-down mix of script.py (manual env stepping, headless-friendly)
and play.py (checkpoint loading via RSL-RL). It does NOT export the policy
or record video — it just runs the trained policy for N steps and reports
two numbers you asked about:

  1. Tray tilt (deg) around the world X axis and Y axis, plus the total
     tilt magnitude reported by ee_tilt_angle (for cross-checking).
  2. End-effector position tracking error vs the commanded target, reusing
     ee_position_command_error_b from observations.py directly.

Usage
-----
python eval_policy.py --task <TASK_ID> --load_run <RUN_DIR> --checkpoint -1 \
    --num_envs 64 --steps 500
"""

# ──────────────────────────────────────────────────────────────────────────
# 1. EARLY IMPORTS & PATH SETUP
# ──────────────────────────────────────────────────────────────────────────
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import my_projects.Kairos_lab.train_lab.tasks as tasks  # noqa: F401 - registers custom Gym tasks

import argparse
from isaaclab.app import AppLauncher
from my_projects.Kairos_lab.train_lab.utils import cli_args

# ──────────────────────────────────────────────────────────────────────────
# 2. ARGUMENT PARSING
# ──────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Simple headless evaluation of a Kairos+ WBC policy.")
parser.add_argument("--task", type=str, required=True, help="Gym task id, e.g. Kairos-WBC-PPO-v0")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=500, help="Number of evaluation steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Use USD I/O instead of Fabric.")
parser.add_argument(
    "--command_name", type=str, default="tray_pose",
    help="Name of the pose command term (as in CommandsCfg) used for ee_position_command_error_b.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Force headless: this script is pure evaluation, no viewer/video needed.
args_cli.headless = True

# Pass only Hydra args so its ConfigStore stays clean
sys.argv = [sys.argv[0]] + hydra_args

# ──────────────────────────────────────────────────────────────────────────
# 3. LAUNCH ISAAC SIM
# ──────────────────────────────────────────────────────────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ──────────────────────────────────────────────────────────────────────────
# 4. REST OF IMPORTS (after Isaac Sim is up)
# ──────────────────────────────────────────────────────────────────────────
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from my_projects.Kairos_lab.train_lab.utils.parser_cfg import parse_env_cfg

# Reuse the project's own observation functions instead of re-deriving them,
# so the evaluation stays consistent with what the policy is actually trained on.
from my_projects.Kairos_lab.train_lab.tasks.mdp.observations import ee_position_command_error_b, ee_tilt_angle

# ──────────────────────────────────────────────────────────────────────────
# 5. CONFIG YOU MAY NEED TO ADJUST TO MATCH YOUR ENV CFG
# ──────────────────────────────────────────────────────────────────────────
# rg6_tcp_link's local "up" axis is +Y, not +Z (confirmed empirically via the
# ee_flat_orientation reward fix). Used both for the world-frame X/Y tilt
# breakdown below and when calling ee_tilt_angle.
UP_AXIS_LOCAL = (0.0, 0.0, 1.0)

TRAY_CFG = SceneEntityCfg("robot", body_names="Link7_2")


def resolve_checkpoint(agent_cfg, args_cli) -> str:
    """Resolve the full path to the model checkpoint (mirrors play.py's _resolve_checkpoint).

    Kept as its own branch: when --load_run is an absolute, already-existing
    directory, log_root IS the run directory itself — so we must NOT pass it
    again to get_checkpoint_path as a run-dir pattern to search *inside*
    log_root (that's what caused the "No runs present... match: '<same path>'"
    error). In that case we just glob model_*.pt directly.
    """
    log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    override_run_dir = False

    if args_cli.load_run:
        if os.path.isabs(args_cli.load_run) and os.path.isdir(args_cli.load_run):
            log_root = args_cli.load_run
            override_run_dir = True
        else:
            candidate = os.path.abspath(os.path.join("logs", args_cli.load_run))
            if os.path.isdir(candidate):
                log_root = candidate
                override_run_dir = True

    log_root = os.path.abspath(log_root)
    print(f"[INFO] Loading experiment from: {log_root}")

    if args_cli.checkpoint and args_cli.checkpoint.strip() != "-1":
        return retrieve_file_path(args_cli.checkpoint)

    if override_run_dir:
        pts = sorted(
            [f for f in os.listdir(log_root) if f.startswith("model_") and f.endswith(".pt")],
            key=lambda m: int(m.split("_")[1].split(".")[0]),
        )
        if not pts:
            raise FileNotFoundError(f"[ERROR] No model_*.pt checkpoints in: {log_root}")
        return os.path.join(log_root, pts[-1])

    pattern = agent_cfg.load_checkpoint if agent_cfg.load_checkpoint != "-1" else ".*"
    return get_checkpoint_path(log_root, agent_cfg.load_run or ".*", pattern)


# ──────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────────────────────
def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    resume_path = resolve_checkpoint(agent_cfg, args_cli)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped

    # Resolve the tray body id once (required before body_ids can be indexed).
    TRAY_CFG.resolve(base_env.scene)
    tray_body_id = TRAY_CFG.body_ids[0]

    up_axis_local = torch.tensor(UP_AXIS_LOCAL, device=base_env.device)

    # Sanity check the command term exists before running the whole loop.
    has_command = args_cli.command_name in base_env.command_manager.active_terms
    if not has_command:
        print(
            f"[AVISO] El término de comando '{args_cli.command_name}' no está en "
            f"command_manager.active_terms ({list(base_env.command_manager.active_terms)}). "
            f"Pasa --command_name <nombre_correcto>."
        )

    tilt_x_samples = []
    tilt_y_samples = []
    tilt_total_samples = []
    ee_err_samples = []

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    print(f"[INFO] Running evaluation for {args_cli.steps} steps over {args_cli.num_envs} envs...")

    with torch.inference_mode():
        for step in range(args_cli.steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            # ---- Tray tilt: world-frame X/Y breakdown -------------------------
            # (base-frame projected gravity doesn't separate roll/pitch cleanly
            # once the base has yawed, so this is done in world frame instead.)
            tray_quat_w = base_env.scene["robot"].data.body_quat_w[:, tray_body_id]
            up_world = quat_apply(tray_quat_w, up_axis_local.expand(tray_quat_w.shape[0], -1))

            tilt_about_y = torch.atan2(-up_world[:, 0], up_world[:, 2])  # shows up as X drift
            tilt_about_x = torch.atan2(up_world[:, 1], up_world[:, 2])   # shows up as Y drift
            tilt_x_samples.append(torch.rad2deg(tilt_about_y).abs())
            tilt_y_samples.append(torch.rad2deg(tilt_about_x).abs())

            # ---- Tray tilt: total magnitude, reusing your own function --------
            tilt_total = ee_tilt_angle(base_env, TRAY_CFG, up_axis=UP_AXIS_LOCAL)
            tilt_total_samples.append(torch.rad2deg(tilt_total))

            # ---- End-effector position tracking error -------------------------
            if has_command:
                err_b = ee_position_command_error_b(base_env, args_cli.command_name, TRAY_CFG)
                ee_err_samples.append(torch.norm(err_b, dim=-1))

    tilt_x_all = torch.cat(tilt_x_samples)
    tilt_y_all = torch.cat(tilt_y_samples)
    tilt_total_all = torch.cat(tilt_total_samples)

    print("\n===== RESULTADOS DE LA EVALUACIÓN =====")
    print(
        f"Inclinación bandeja eje X: media={tilt_x_all.mean().item():.3f}°  "
        f"std={tilt_x_all.std().item():.3f}°  max={tilt_x_all.max().item():.3f}°"
    )
    print(
        f"Inclinación bandeja eje Y: media={tilt_y_all.mean().item():.3f}°  "
        f"std={tilt_y_all.std().item():.3f}°  max={tilt_y_all.max().item():.3f}°"
    )
    print(
        f"Inclinación bandeja total (ee_tilt_angle): media={tilt_total_all.mean().item():.3f}°  "
        f"std={tilt_total_all.std().item():.3f}°  max={tilt_total_all.max().item():.3f}°"
    )

    if ee_err_samples:
        ee_err_all = torch.cat(ee_err_samples)
        print(
            f"Error de seguimiento del efector final: media={ee_err_all.mean().item() * 100:.2f} cm  "
            f"std={ee_err_all.std().item() * 100:.2f} cm  max={ee_err_all.max().item() * 100:.2f} cm"
        )
    else:
        print("[AVISO] No se pudo calcular el error de seguimiento (comando no encontrado).")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()