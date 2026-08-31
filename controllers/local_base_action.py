import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.assets.articulation import Articulation
from isaaclab.utils import configclass

# Mecanum Controller Class
class MecanumChassisVelocityAction(ActionTerm):
    cfg: "MecanumChassisVelocityActionCfg"
    _asset: Articulation
 
    def __init__(self, cfg: "MecanumChassisVelocityActionCfg", env):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
 
        self._base_body_index, _ = self._asset.find_bodies(cfg.body_name)
 
        self._action_dim = 3
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
 
        self._vel_limits = torch.tensor(
            [self.cfg.max_lin_vel, self.cfg.max_lin_vel, self.cfg.max_ang_vel],
            device=self.device,
        )
 
        # --- Parameters for sim2real ---
        self._accel_limits = torch.tensor(
            [self.cfg.max_lin_accel, self.cfg.max_lin_accel, self.cfg.max_ang_accel],
            device=self.device,
        )
 
        # Local velocity that is actually applied each physics step. It mimics the fact that the
        # lower-level controller / real motors cannot jump instantaneously to the policy's requested
        # velocity.
        self._current_vel_local = torch.zeros(self.num_envs, self._action_dim, device=self.device)
 
        # Physics dt, used to convert the maximum acceleration (m/s^2) into a maximum
        # speed delta per physics step. Adjust the attribute name if your Isaac Lab version exposes
        # dt differently.
        try:
            self._physics_dt = env.sim.get_physics_dt()
        except AttributeError:
            self._physics_dt = env.physics_dt
 
        # Circular command buffer to simulate communication/control latency of the real robot.
        # With action_latency_steps=0 (default), no extra delay is introduced; increase it when you
        # want to stress the policy against that effect.
        self._latency_steps = max(int(self.cfg.action_latency_steps), 0)
        self._cmd_buffer = torch.zeros(
            self._latency_steps + 1, self.num_envs, self._action_dim, device=self.device
        )
 
    @property
    def action_dim(self) -> int:
        return self._action_dim
 
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
 
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
 
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # Saturate to reasonable physical limits (prevents absurd commands from the policy during
        # early training phases).
        target = torch.clamp(actions, min=-self._vel_limits, max=self._vel_limits)
 
        # FIFO latency: we push the new command and pop the oldest one as the one that actually
        # gets applied in this control step.
        self._cmd_buffer = torch.roll(self._cmd_buffer, shifts=-1, dims=0)
        self._cmd_buffer[-1] = target
        self._processed_actions[:] = self._cmd_buffer[0]
 
    def apply_actions(self):
        # --- Rate limiting by maximum acceleration ---
        # This is the element that most closes the sim2real gap: it replaces the instantaneous
        # velocity jump with a ramp bounded by self._accel_limits (evaluated at physics dt), so the
        # policy cannot learn to exploit acceleration that the real Kairos+ could not execute.
        max_delta = self._accel_limits * self._physics_dt
        delta = torch.clamp(
            self._processed_actions - self._current_vel_local, min=-max_delta, max=max_delta
        )
        self._current_vel_local += delta
 
        v_x_local = self._current_vel_local[:, 0]
        v_y_local = self._current_vel_local[:, 1]
        w_z_local = self._current_vel_local[:, 2]
 
        # Current chassis yaw (local -> global)
        quat = self._asset.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
 
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
 
        v_x_global = v_x_local * cos_yaw - v_y_local * sin_yaw
        v_y_global = v_x_local * sin_yaw + v_y_local * cos_yaw
 
        # We start from the current root body velocity state to preserve vz, wx, and wy as computed
        # by physics in the last substep (gravity + contact with the ground -> allows climbing/descending ramps).
        lin_vel = self._asset.data.root_lin_vel_w.clone()
        ang_vel = self._asset.data.root_ang_vel_w.clone()
 
        lin_vel[:, 0] = v_x_global
        lin_vel[:, 1] = v_y_global
        # lin_vel[:, 2] (vz) -> left as computed by physics
 
        ang_vel[:, 2] = w_z_local
        # ang_vel[:, 0:2] (wx, wy / roll, pitch) -> left as computed by physics
 
        root_velocity = torch.cat([lin_vel, ang_vel], dim=-1)
 
        self._asset.write_root_velocity_to_sim(root_velocity)
 
    def reset(self, env_ids: torch.Tensor | None = None):
        # IMPORTANT: without this reset, when an episode restarts the robot would start with the
        # speed ramp inherited from the previous episode (residual state between episodes), biasing
        # training.
        if env_ids is None:
            env_ids = slice(None)
        self._current_vel_local[env_ids] = 0.0
        self._cmd_buffer[:, env_ids] = 0.0
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
 
 
@configclass
class MecanumChassisVelocityActionCfg(ActionTermCfg):
    class_type: type = MecanumChassisVelocityAction
    asset_name: str = "robot"
    body_name: str = "base_link"
    max_lin_vel: float = 2.0   # m/s, Vx/Vy saturation
    max_ang_vel: float = 2.0   # rad/s, Yaw-rate saturation
 
    # --- Key parameters for sim2real ---
    # WARNING: these values are a reasonable estimate for a mecanum AGV of this size,
    # NOT the official Kairos+ specification. Replace them with the actual limits
    # (Robotnik datasheet or measured on the real robot while accelerating at full power)
    # before considering the policy good for transfer.
    max_lin_accel: float = 1.0    # m/s^2, maximum linear acceleration
    max_ang_accel: float = 3.0    # rad/s^2, maximum angular acceleration
    action_latency_steps: int = 0  # number of control steps of delay (0 = no extra latency)