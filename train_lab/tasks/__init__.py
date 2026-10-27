import gymnasium as gym

# Default Walk
gym.register(
    id="Kairos-PPO-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.kairosplus_env_cfg:KairosEnvCfg",
        "play_env_cfg_entry_point": "tasks.kairosplus_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "tasks.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Kairos-PPO-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.kairosplus_env_cfg_latent:KairosEnvCfg",
        "play_env_cfg_entry_point": "tasks.kairosplus_env_cfg_latent:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "tasks.ppo_cfg:LatentPPORunnerCfg",
    },
)

gym.register(
    id="Kairos-DPPO-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.kairosplus_env_cfg_latent:KairosEnvCfg",
        "play_env_cfg_entry_point": "tasks.kairosplus_env_cfg_latent:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "tasks.dppo_cfg:LatentDPPORunnerCfg",
    },
)
