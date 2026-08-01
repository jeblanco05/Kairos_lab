import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.assets.articulation import Articulation
from isaaclab.utils import configclass

# Logical Controller Class
class LocalBaseVelocityAction(ActionTerm):
    def __init__(self, cfg: "LocalBaseVelocityActionCfg", env):
        super().__init__(cfg, env)
        
        # Extract robot and joint indices
        self.robot = env.scene[cfg.asset_name]
        self.joint_indices, _ = self.robot.find_joints(cfg.joint_names)
        
        # NUEVO: Buscamos específicamente el índice del motor de giro (yaw)
        self.yaw_joint_idx, _ = self.robot.find_joints("virtual_joint_yaw")
        
        # Initialize action tensors
        self._action_dim = 3
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)

    @property
    def action_dim(self) -> int:
        """Returns the dimension of the action space."""
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """Returns the raw actions received from the policy."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Returns the processed actions applied to the simulator."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Processes the local velocity actions into global velocity targets."""
        self._raw_actions = actions
        
        yaw = self.robot.data.joint_pos[:, self.yaw_joint_idx[0]]
        
        # Extract local velocity commands
        v_x_local = actions[:, 0]
        v_y_local = actions[:, 1]
        w_z_local = actions[:, 2]

        # Compute trigonometric values
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        # 2D Rotation Matrix from Local to Global
        v_x_global = v_x_local * cos_yaw - v_y_local * sin_yaw
        v_y_global = v_x_local * sin_yaw + v_y_local * cos_yaw

        # Store the processed global actions
        self._processed_actions = torch.stack([v_x_global, v_y_global, w_z_local], dim=-1)

    def apply_actions(self):
        """Applies the processed global velocities to the physics engine."""
        self.robot.set_joint_velocity_target(self._processed_actions, joint_ids=self.joint_indices)

# Configuration Class
@configclass
class LocalBaseVelocityActionCfg(ActionTermCfg):
    class_type: type = LocalBaseVelocityAction 
    asset_name: str = "robot"
    joint_names: list[str] = ["virtual_joint_x", "virtual_joint_y", "virtual_joint_yaw"]




# Mecanum Controller Class
class MecanumChassisVelocityAction(ActionTerm):
    cfg: "MecanumChassisVelocityActionCfg"
    _asset: Articulation
 
    def __init__(self, cfg: "MecanumChassisVelocityActionCfg", env):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
 
        # Solo necesitamos el índice del chasis para leer su orientación actual
        self._base_body_index, _ = self._asset.find_bodies(cfg.body_name)
 
        self._action_dim = 3
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
 
        self._vel_limits = torch.tensor(
            [self.cfg.max_lin_vel, self.cfg.max_lin_vel, self.cfg.max_ang_vel],
            device=self.device,
        )
 
        # --- Parámetros para sim2real ---
        self._accel_limits = torch.tensor(
            [self.cfg.max_lin_accel, self.cfg.max_lin_accel, self.cfg.max_ang_accel],
            device=self.device,
        )
 
        # Velocidad LOCAL que efectivamente se aplica cada physics step, rampada
        # en aceleración hacia el comando objetivo. Imita el hecho de que el
        # controlador de bajo nivel / motores reales no pueden saltar
        # instantáneamente a la velocidad pedida por la política.
        self._current_vel_local = torch.zeros(self.num_envs, self._action_dim, device=self.device)
 
        # dt de física, usado para convertir aceleración máxima (m/s^2) en un
        # delta de velocidad máximo por physics step. Ajusta el nombre de
        # atributo si tu versión de Isaac Lab expone el dt de otra forma.
        try:
            self._physics_dt = env.sim.get_physics_dt()
        except AttributeError:
            self._physics_dt = env.physics_dt
 
        # Buffer circular de comandos para simular la latencia de comunicación
        # / control del robot real. Con action_latency_steps=0 (por defecto)
        # no introduce retardo; súbelo cuando quieras curtir la política contra
        # ese efecto.
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
        # Saturamos a límites físicos razonables (evita comandos absurdos de la
        # política durante las primeras fases del entrenamiento).
        target = torch.clamp(actions, min=-self._vel_limits, max=self._vel_limits)
 
        # FIFO de latencia: metemos el comando nuevo y sacamos el más antiguo
        # como el que realmente toca aplicar este step de control.
        self._cmd_buffer = torch.roll(self._cmd_buffer, shifts=-1, dims=0)
        self._cmd_buffer[-1] = target
        self._processed_actions[:] = self._cmd_buffer[0]
 
    def apply_actions(self):
        # --- Rate limiting por aceleración máxima ---
        # Esto es lo que más cierra el gap sim2real: sustituye el salto
        # instantáneo de velocidad por una rampa acotada por
        # self._accel_limits (evaluada al dt de física), de forma que la
        # política no puede aprender a explotar una aceleración que el
        # Kairos+ real no podría ejecutar.
        max_delta = self._accel_limits * self._physics_dt
        delta = torch.clamp(
            self._processed_actions - self._current_vel_local, min=-max_delta, max=max_delta
        )
        self._current_vel_local += delta
 
        v_x_local = self._current_vel_local[:, 0]
        v_y_local = self._current_vel_local[:, 1]
        w_z_local = self._current_vel_local[:, 2]
 
        # Yaw actual del chasis (local -> global)
        quat = self._asset.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
 
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
 
        v_x_global = v_x_local * cos_yaw - v_y_local * sin_yaw
        v_y_global = v_x_local * sin_yaw + v_y_local * cos_yaw
 
        # Partimos del estado de velocidad actual del cuerpo raíz para conservar
        # vz, wx, wy tal como los ha calculado la física en el último substep
        # (gravedad + contacto con el terreno -> permite subir/bajar cuestas).
        lin_vel = self._asset.data.root_lin_vel_w.clone()
        ang_vel = self._asset.data.root_ang_vel_w.clone()
 
        lin_vel[:, 0] = v_x_global
        lin_vel[:, 1] = v_y_global
        # lin_vel[:, 2] (vz) -> se deja como la calcula la física
 
        ang_vel[:, 2] = w_z_local
        # ang_vel[:, 0:2] (wx, wy / roll, pitch) -> se deja como la calcula la física
 
        root_velocity = torch.cat([lin_vel, ang_vel], dim=-1)
 
        # NOTA DE VERSIÓN: el nombre de este método ha cambiado entre versiones
        # de Isaac Lab (write_root_velocity_to_sim en versiones antiguas;
        # write_root_com_velocity_to_sim / write_root_link_velocity_to_sim en
        # versiones más recientes que separan frame de CoM y de link). Ajusta
        # la siguiente línea al método disponible en tu instalación
        # (`dir(self._asset)` o la documentación de tu versión de Isaac Lab).
        self._asset.write_root_velocity_to_sim(root_velocity)
 
    def reset(self, env_ids: torch.Tensor | None = None):
        # IMPORTANTE: sin este reset, al reiniciar un episodio el robot
        # arrancaría con la rampa de velocidad heredada del episodio anterior
        # (estado residual entre episodios), sesgando el entrenamiento.
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
    max_lin_vel: float = 2.0   # m/s, saturación de Vx/Vy
    max_ang_vel: float = 2.0   # rad/s, saturación de Yaw-rate
 
    # --- Parámetros clave para sim2real ---
    # OJO: estos valores son una estimación razonable para un AGV mecanum de
    # este porte, NO son la especificación oficial del Kairos+. Sustitúyelos
    # por los límites reales (datasheet de Robotnik o medidos con el robot
    # físico acelerando a fondo) antes de dar por buena la política para
    # transferir.
    max_lin_accel: float = 1.0    # m/s^2, aceleración lineal máxima
    max_ang_accel: float = 3.0    # rad/s^2, aceleración angular máxima
    action_latency_steps: int = 0  # nº de steps de control de retardo (0 = sin latencia extra)