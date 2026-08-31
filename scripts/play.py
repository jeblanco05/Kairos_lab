"""
Interactive play script for RSL-RL policies in Isaac Lab.

Loads a trained checkpoint (PPO, Distillation or D-PPO), exports it to
JIT and ONNX, and runs the policy in the simulator indefinitely (or for
--video_length steps when --video is active).

Runner type is resolved automatically from agent_cfg.class_name:
  OnPolicyRunner | DistillationRunner | DPPORunner

Example usage
-------------
python play.py --task G1-RlSl-PPO-v1 --load_run <run_dir> --checkpoint -1
python play.py --task G1-RlSl-DPPO-v1 --load_run /abs/path/to/run --checkpoint -1 --video
"""

# ──────────────────────────────────────────────────────────────────────────────
# 1. EARLY IMPORTS & PATH SETUP
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys

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
parser = argparse.ArgumentParser(
    description="Play a trained RSL-RL policy interactively.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--task", type=str, required=True, help="Gym task id, e.g. G1-RlSl-PPO-v1")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--video", action="store_true", default=False, help="Record a single video and exit.")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Use USD I/O instead of Fabric.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Load a published Nucleus checkpoint.")
parser.add_argument("--real_time", action="store_true", default=False, help="Throttle simulation to real-time.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)

args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# Pass only Hydra args so its ConfigStore stays clean
sys.argv = [sys.argv[0]] + hydra_args

# ──────────────────────────────────────────────────────────────────────────────
# 3. LAUNCH ISAAC SIM
# ──────────────────────────────────────────────────────────────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ──────────────────────────────────────────────────────────────────────────────
# 4. REST OF IMPORTS (after Isaac Sim is up)
# ──────────────────────────────────────────────────────────────────────────────
import time
import torch
import gymnasium as gym
from importlib.metadata import version as lib_version

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from rsl_rl.runners import OnPolicyRunner, DistillationRunner, DPPORunner

try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    def get_published_pretrained_checkpoint(*args, **kwargs):
        return None

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path
from utils.parser_cfg import parse_env_cfg
from utils.export_deploy_cfg import export_deploy_cfg

# ──────────────────────────────────────────────────────────────────────────────
# 5. HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_checkpoint(agent_cfg, args_cli) -> str:
    """Resolve the full path to the model checkpoint."""
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

    if args_cli.use_pretrained_checkpoint:
        path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not path:
            raise RuntimeError("[ERROR] No pre-trained checkpoint available for this task.")
        return path

    if args_cli.checkpoint:
        if args_cli.checkpoint.strip() == "-1":
            if override_run_dir:
                pts = sorted(
                    [f for f in os.listdir(log_root) if f.startswith("model_") and f.endswith(".pt")],
                    key=lambda m: int(m.split("_")[1].split(".")[0]),
                )
                if not pts:
                    raise FileNotFoundError(f"[ERROR] No model_*.pt checkpoints in: {log_root}")
                return os.path.join(log_root, pts[-1])
            elif agent_cfg.load_run and os.path.isabs(agent_cfg.load_run):
                pattern = agent_cfg.load_checkpoint if agent_cfg.load_checkpoint != "-1" else ".*"
                return get_checkpoint_path(agent_cfg.load_run, ".*", pattern)
            else:
                pattern = agent_cfg.load_checkpoint if agent_cfg.load_checkpoint != "-1" else ".*"
                return get_checkpoint_path(log_root, agent_cfg.load_run or ".*", pattern)
        else:
            return retrieve_file_path(args_cli.checkpoint)

    return get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _build_runner(agent_cfg, env):
    """Instantiate the correct RSL-RL runner from agent_cfg.class_name."""
    class_name = agent_cfg.class_name
    kwargs = dict(env=env, train_cfg=agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runners = {
        "OnPolicyRunner":    OnPolicyRunner,
        "DistillationRunner": DistillationRunner,
        "DPPORunner":        DPPORunner,
    }
    if class_name not in runners:
        raise ValueError(f"[ERROR] Unsupported runner class: '{class_name}'. "
                         f"Valid options: {list(runners)}")
    print(f"[INFO] Instantiating {class_name}.")
    return runners[class_name](**kwargs)


def _export_policy(runner, resume_path: str) -> None:
    """Export the policy network to JIT and ONNX formats."""
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    normalizer = (
        getattr(policy_nn, "actor_obs_normalizer", None)
        or getattr(policy_nn, "student_obs_normalizer", None)
    )
    export_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
    print(f"[INFO] Policy exported to: {export_dir}")


def _get_initial_obs(env):
    """Get the initial observation, handling rsl-rl-lib 2.3.x API."""
    obs = env.get_observations()
    if lib_version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    return obs


# ──────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Config ────────────────────────────────────────────────────────────────
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    resume_path = _resolve_checkpoint(agent_cfg, args_cli)
    print(f"[INFO] Checkpoint resolved: {resume_path}")
    log_dir = os.path.dirname(resume_path)

    # ── Environment ───────────────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos", "play")
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video → {video_dir}")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Policy ────────────────────────────────────────────────────────────────
    runner = _build_runner(agent_cfg, env)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    export_deploy_cfg(env.unwrapped, agent_cfg.obs_groups, log_dir)
    _export_policy(runner, resume_path)

    # ── Simulation loop ───────────────────────────────────────────────────────
    obs = _get_initial_obs(env)
    dt = env.unwrapped.step_dt
    timestep = 0

    try:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    print("[INFO] Video length reached — exiting.")
                    break

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()