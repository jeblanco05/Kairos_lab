"""
Unified training script for RSL-RL algorithms in Isaac Lab.

Supports PPO, Distillation and D-PPO, selected automatically from the task name:
  - task contains 'DPPO'       → DPPORunner      (RslRlDPPORunnerCfg)
  - task contains 'Distill'    → DistillationRunner (RslRlOnPolicyRunnerCfg)
  - otherwise                  → OnPolicyRunner   (RslRlOnPolicyRunnerCfg)

Example usage
-------------
# Standard PPO
python train.py --task G1-RlSl-PPO-v1 --num_envs 4096

# Distillation (teacher checkpoint resolved automatically)
python train.py --task G1-RlSl-Distill-v1 --load_run <run_dir> --checkpoint -1

# D-PPO
python train.py --task G1-RlSl-DPPO-v1 --load_run <run_dir> --checkpoint -1
"""

# ──────────────────────────────────────────────────────────────────────────────
# 1. EARLY IMPORTS & PATH SETUP  (must happen before Isaac Sim launches)
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import pathlib

# Make the project root importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import tasks  # noqa: F401 – registers custom Gym tasks 

import argparse
import argcomplete
from isaaclab.app import AppLauncher
from utils import cli_args

# ──────────────────────────────────────────────────────────────────────────────
# 2. ARGUMENT PARSING
# ──────────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified RSL-RL training launcher (PPO | Distillation | D-PPO).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", type=str, required=True, help="Gym task id, e.g. G1-RlSl-DPPO-v1")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Training iterations.")
    parser.add_argument("--log_dir", type=str, default="G1_rlsl_lab/logs",help="Root directory for experiment logs.",)
    parser.add_argument("--video", action="store_true", default=False, help="Record training videos.")
    parser.add_argument("--video_length", type=int, default=200, help="Video length in steps.")
    parser.add_argument("--video_interval", type=int, default=2000, help="Steps between video recordings.")
    parser.add_argument("--distributed", action="store_true", default=False,help="Enable multi-GPU / multi-node training.",)
    parser.add_argument("--use_pretrained_checkpoint", action="store_true",
        help="Use a published pre-trained checkpoint from Nucleus (Distillation / D-PPO only).",)
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


_parser = _build_parser()
argcomplete.autocomplete(_parser)
args_cli, hydra_args = _parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# Pass only Hydra args through sys.argv so Hydra's ConfigStore is clean
sys.argv = [sys.argv[0]] + hydra_args

# ──────────────────────────────────────────────────────────────────────────────
# 3. DETECT TRAINING MODE FROM TASK NAME
# ──────────────────────────────────────────────────────────────────────────────
def _detect_mode(task: str) -> str:
    """Return 'dppo', 'distillation', or 'ppo' based on the task name."""
    task_upper = task.upper()
    if "DPPO" in task_upper:
        return "dppo"
    if "DISTILL" in task_upper:
        return "distillation"
    return "ppo"


_MODE = _detect_mode(args_cli.task)
print(f"[INFO] Training mode detected: {_MODE.upper()}  (task={args_cli.task})")

# ──────────────────────────────────────────────────────────────────────────────
# 4. LAUNCH ISAAC SIM
# ──────────────────────────────────────────────────────────────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ──────────────────────────────────────────────────────────────────────────────
# 5. VERSION CHECK (distributed only)
# ──────────────────────────────────────────────────────────────────────────────
import importlib.metadata as _meta
import platform
from packaging import version as _version

_RSL_RL_MIN = "2.3.1"
_installed = _meta.version("rsl-rl-lib")
if args_cli.distributed and _version.parse(_installed) < _version.parse(_RSL_RL_MIN):
    _install_cmd = (
        [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={_RSL_RL_MIN}"]
        if platform.system() == "Windows"
        else ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={_RSL_RL_MIN}"]
    )
    print(
        f"[ERROR] Installed rsl-rl-lib=={_installed}, but distributed training requires >={_RSL_RL_MIN}.\n"
        f"Fix with:\n\n\t{' '.join(_install_cmd)}\n"
    )
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# 6. REST OF IMPORTS (after Isaac Sim is up)
# ──────────────────────────────────────────────────────────────────────────────
import inspect
import shutil
import torch
from datetime import datetime

import gymnasium as gym
import isaaclab_tasks  # noqa: F401

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from utils.export_deploy_cfg import export_deploy_cfg

# D-PPO cfg only imported when needed to avoid ConfigStore conflicts on PPO/Distillation runs
if _MODE == "dppo":
    from isaaclab_rl.rsl_rl import RslRlDPPORunnerCfg

try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    def get_published_pretrained_checkpoint(*args, **kwargs):  # type: ignore[misc]
        return None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# 7. HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_teacher_checkpoint(agent_cfg, args_cli) -> str | None:
    """
    Resolve the teacher model checkpoint path for Distillation and D-PPO modes.

    Resolution priority (mirrors the original dppo_train.py logic):
      1. --use_pretrained_checkpoint  → Nucleus asset
      2. --checkpoint <path>          → direct file path via retrieve_file_path
      3. --checkpoint -1              → latest .pt file inside load_run directory
      4. No --checkpoint              → get_checkpoint_path with load_run / load_checkpoint
    """
    # --- build teacher root ---
    teacher_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    override_run_dir = False

    if args_cli.load_run:
        if os.path.isdir(args_cli.load_run):
            teacher_root = args_cli.load_run
            override_run_dir = True
        else:
            candidate = os.path.abspath(os.path.join("logs", args_cli.load_run))
            if os.path.isdir(candidate):
                teacher_root = candidate
                override_run_dir = True

    teacher_root = os.path.abspath(teacher_root)
    print(f"[INFO] Searching teacher checkpoint in: {teacher_root}")

    # --- resolve checkpoint ---
    if args_cli.use_pretrained_checkpoint:
        path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not path:
            raise ValueError("[ERROR] Pre-trained checkpoint unavailable for this task.")
        return path

    if args_cli.checkpoint:
        if args_cli.checkpoint.strip() == "-1":
            if override_run_dir:
                pts = sorted(
                    [f for f in os.listdir(teacher_root) if f.startswith("model_") and f.endswith(".pt")],
                    key=lambda m: int(m.split("_")[1].split(".")[0]),
                )
                if not pts:
                    raise ValueError(f"[ERROR] No model_*.pt checkpoints found in: {teacher_root}")
                return os.path.join(teacher_root, pts[-1])
            elif agent_cfg.load_run and os.path.isabs(agent_cfg.load_run):
                pattern = agent_cfg.load_checkpoint if agent_cfg.load_checkpoint != "-1" else ".*"
                return get_checkpoint_path(agent_cfg.load_run, ".*", pattern)
            else:
                pattern = agent_cfg.load_checkpoint if agent_cfg.load_checkpoint != "-1" else ".*"
                return get_checkpoint_path(teacher_root, agent_cfg.load_run or ".*", pattern)
        else:
            return retrieve_file_path(args_cli.checkpoint)

    return get_checkpoint_path(teacher_root, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _build_log_dir(args_cli, agent_cfg, mode: str) -> str:
    """Return the full path to the run-specific log directory."""
    log_root = os.path.abspath(os.path.join(args_cli.log_dir, agent_cfg.experiment_name))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{agent_cfg.run_name}" if agent_cfg.run_name else ""
    run_label = f"_{mode}" if mode != "ppo" else ""
    return os.path.join(log_root, f"{timestamp}{suffix}{run_label}")


def _apply_common_overrides(env_cfg, agent_cfg, args_cli, app_launcher):
    """Apply CLI overrides to env and agent configs."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    return env_cfg, agent_cfg


def _wrap_env(raw_env, log_dir: str, args_cli, agent_cfg):
    """Apply MARL→single-agent conversion, optional video wrapper, and RSL-RL wrapper."""
    env = raw_env
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    return RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


def _dump_artifacts(log_dir: str, env_cfg, agent_cfg, env):
    """Save YAML configs and environment class file to the log directory."""
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, agent_cfg.obs_groups, log_dir)
    src = inspect.getfile(env_cfg.__class__)
    shutil.copy(src, os.path.join(log_dir, "params", os.path.basename(src)))


# ──────────────────────────────────────────────────────────────────────────────
# 8. MODE-SPECIFIC ENTRY POINTS
# ──────────────────────────────────────────────────────────────────────────────

def _run_ppo(env_cfg, agent_cfg):
    """Standard OnPolicy PPO training."""
    from rsl_rl.runners import OnPolicyRunner

    log_dir = _build_log_dir(args_cli, agent_cfg, mode="ppo")
    print(f"[INFO] PPO log directory: {log_dir}")

    # Resolve resume checkpoint (if resuming or running Distillation via train.py compat)
    log_root = os.path.abspath(os.path.join("G1_rlsl_lab/logs", agent_cfg.experiment_name))
    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = _wrap_env(env, log_dir, args_cli, agent_cfg)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if resume_path:
        print(f"[INFO] Resuming from checkpoint: {resume_path}")
        runner.load(resume_path)

    _dump_artifacts(log_dir, env_cfg, agent_cfg, env) # TODO da error con exportar de unitree arreglar

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    finally:
        env.close()
        simulation_app.close()


def _run_distillation(env_cfg, agent_cfg):
    """Teacher→Student distillation using DistillationRunner."""
    from rsl_rl.runners import DistillationRunner

    # Enforce correct algorithm class
    if agent_cfg.algorithm.class_name != "Distillation":
        print("[WARNING] Forcing algorithm class_name → 'Distillation'.")
        agent_cfg.algorithm.class_name = "Distillation"

    resume_path = _resolve_teacher_checkpoint(agent_cfg, args_cli)
    print(f"[INFO] Teacher checkpoint: {resume_path}")

    log_dir = _build_log_dir(args_cli, agent_cfg, mode="distillation")
    print(f"[INFO] Distillation log directory: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = _wrap_env(env, log_dir, args_cli, agent_cfg)

    runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    runner.load(resume_path)

    _dump_artifacts(log_dir, env_cfg, agent_cfg, env)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()
        simulation_app.close()


def _run_dppo(env_cfg, agent_cfg):
    """D-PPO (Distillation-PPO) training using DPPORunner."""
    from rsl_rl.runners import DPPORunner

    # Enforce correct algorithm class
    if agent_cfg.algorithm.class_name != "DPPO":
        print("[WARNING] Forcing algorithm class_name → 'DPPO'.")
        agent_cfg.algorithm.class_name = "DPPO"

    resume_path = _resolve_teacher_checkpoint(agent_cfg, args_cli)
    print(f"[INFO] Teacher checkpoint: {resume_path}")

    log_dir = _build_log_dir(args_cli, agent_cfg, mode="dppo")
    print(f"[INFO] D-PPO log directory: {log_dir}")

    print(f"[INFO] Creating gym environment (seed={agent_cfg.seed})")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print("[INFO] Gym environment created.")
    env = _wrap_env(env, log_dir, args_cli, agent_cfg)

    runner = DPPORunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    runner.load(resume_path)

    _dump_artifacts(log_dir, env_cfg, agent_cfg, env)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()
        simulation_app.close()


# ──────────────────────────────────────────────────────────────────────────────
# 9. MAIN ENTRY POINT  (Hydra-decorated, so cfg injection happens here)
# ──────────────────────────────────────────────────────────────────────────────

def _cfg_entry_point_for_mode(mode: str) -> str:
    """Return the Hydra cfg entry point key for the detected mode."""
    return "rsl_rl_cfg_entry_point"


# The agent_cfg type annotation drives Hydra's config class selection.
# We need the correct type at decoration time → conditional decoration.

if _MODE == "dppo":
    @hydra_task_config(args_cli.task, _cfg_entry_point_for_mode(_MODE))
    def main(
        env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        agent_cfg: "RslRlDPPORunnerCfg",
    ):
        env_cfg, agent_cfg = _apply_common_overrides(env_cfg, agent_cfg, args_cli, app_launcher)
        _run_dppo(env_cfg, agent_cfg)

else:
    @hydra_task_config(args_cli.task, _cfg_entry_point_for_mode(_MODE))
    def main(
        env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        agent_cfg: RslRlOnPolicyRunnerCfg,
    ):
        env_cfg, agent_cfg = _apply_common_overrides(env_cfg, agent_cfg, args_cli, app_launcher)
        if _MODE == "distillation":
            _run_distillation(env_cfg, agent_cfg)
        else:
            _run_ppo(env_cfg, agent_cfg)


if __name__ == "__main__":
    main()
    simulation_app.close()