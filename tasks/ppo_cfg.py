from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticLatentCfg

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 1000
    experiment_name = "Kairos-PPO-v1"  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class G1LatentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner para entrenar el teacher con arquitectura latente (Paso 1)."""
 
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 1000
    experiment_name = "G1-RlSl-PPO-v2"
    empirical_normalization = False
    obs_groups = {
        "policy": ["obs_noised", "obs_priv"],
        "critic": ["obs", "obs_priv"],
    }
 
    policy = RslRlPpoActorCriticLatentCfg(
        init_noise_std=1.0,
        activation="elu",
        latent_dim=8,
        lateral_hidden_dims=[128, 64],
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256],
    )

    algorithm = RslRlPpoAlgorithmCfg(
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
    )