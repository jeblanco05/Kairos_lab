from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (RslRlDPPORunnerCfg, RslRlDPPOAlgorithmCfg, 
RslRlDPPOStudentTeacherCfg, RslRlDPPOStudentTeacherLatentCfg)

@configclass
class DPPORunnerCfg(RslRlDPPORunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 500
    experiment_name = "Kairos-DPPO-v1"
    empirical_normalization = False
    obs_groups = {
        "policy": ["obs_noised", "history"],
        "teacher": ["obs_noised", "obs_priv"],
        "critic": ["obs", "obs_priv"],
    }
    policy = RslRlDPPOStudentTeacherCfg(
        init_noise_std=0.05,
        teacher_hidden_dims=[256, 128, 64],
        student_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256],
        activation="elu",
    )
    algorithm = RslRlDPPOAlgorithmCfg(
        # PPO
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1817208507534304,
        entropy_coef= 0.009052066070652268,
        num_learning_epochs=6,
        num_mini_batches=4,
        learning_rate=0.0005063169358053299,
        schedule="adaptive",
        gamma=0.9819023007941059,
        lam=0.9068786393155833,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Distillation (alpha=0.5, beta=0.5 as in the paper)
        distillation_loss_coef=0.5,
        ppo_loss_coef=0.5,
        distillation_loss_type="mse",
    )

@configclass
class LatentDPPORunnerCfg(RslRlDPPORunnerCfg):
    """Runner para el Paso 2 del pipeline latente (D-PPO con StudentTeacherLatentDPPO)."""
 
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 500
    experiment_name = "Kairos-DPPO-v2"
    empirical_normalization = False
 
    obs_groups = {
        "policy": ["obs_noised", "history"],
        "teacher": ["obs_noised", "obs_priv"],
        "critic": ["obs", "obs_priv"],
    }
 
    policy = RslRlDPPOStudentTeacherLatentCfg(
        init_noise_std=0.05,
        activation="elu",
        latent_dim=8,
        lateral_teacher_hidden_dims=[128, 64],
        lateral_student_hidden_dims=[512,256, 128],
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256],
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        critic_obs_normalization=False,
        freeze_actor_base_steps=500,  # congela actor_base las primeras x iters
        freeze_lateral_net_steps=-500,  # congela lateral_net a partir de x iters
        freeze_critic_steps = -1,
        lateral_net_type="tcn",
        num_history_frames=80,       # T — obs_history_dim / obs_dim_per_frame
        tcn_mlp_hidden_dim=256,
        tcn_conv_layers=[             # defaults del paper; puedes omitir este campo
            [32, 32, 8, 4],
            [32, 32, 5, 1],
            [32, 32, 5, 1],
        ],
    )
 
    algorithm = RslRlDPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1817208507534304,
        entropy_coef=0.009052066070652268,
        num_learning_epochs=6,
        num_mini_batches=4,
        learning_rate=0.0005063169358053299,
        schedule="adaptive",
        gamma=0.9819023007941059,
        lam=0.9068786393155833,
        desired_kl=0.01,
        max_grad_norm=1.0,
        distillation_loss_coef=0.5,
        ppo_loss_coef=0.5,
        latent_distillation_coef = 0.2,
        distillation_loss_type="mse",
    )