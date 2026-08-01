import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs.common import ViewerCfg

from cfg.robotcfg import KAIROS_CFG, KAIROS_CFG_RG6
from controllers.local_base_action import LocalBaseVelocityActionCfg, MecanumChassisVelocityActionCfg
from tasks import mdp 

# 1. Configuración del Terreno
TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=9,
    curriculum=True,
    sub_terrains={
        "pyramid_sloped": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.0, 0.26), # ~15º
        ),
        "inverted_pyramid_sloped": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.0, 0.26), # ~15º
        ),
    },
)

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(12.0, 12.0),
    num_rows=9,
    num_cols=21,
    curriculum=True, # Asegúrate de que esto esté en True
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2)
    },
)


@configclass
class KairosSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TERRAIN_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # robots
    robot: ArticulationCfg = KAIROS_CFG_RG6.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    base_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        history_length=3,
        track_air_time=False,
    )

    wheel_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_wheel_link",
        history_length=3,
        track_air_time=True
    )

    arm_contact_forces = ContactSensorCfg(
        # This regex assumes standard UR5e naming. Adjust it if your USD uses a different namespace or prefix.
        prim_path="{ENV_REGEX_NS}/Robot/arm_(shoulder|upper_arm|forearm|wrist_.*)_link",
        history_length=3,
        track_air_time=False
    )

# 2. Configuración de Acciones (Action Space)
@configclass
class KairosActionsCfg:
    # Acción para la base (Control de velocidad de las articulaciones virtuales)
    base_velocity = MecanumChassisVelocityActionCfg()

    # Acción para el brazo (Control de posición del UR5e)
    arm_position = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["arm_.*_joint"], scale=1.0
    )

# 3. Configuración de Observaciones (Observation Space)
@configclass
class KairosObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observaciones disponibles también en el robot real (sin información privilegiada)."""
 
        # --- Estado de la base ---
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # Inclinación de la base respecto a la gravedad (IMU)
        base_projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
 
        # --- Comandos (tarea 1: velocidad de la base; tarea 2: pose objetivo de la bandeja) ---
        # OJO: el comando de velocidad NO lleva ruido. Es la consigna que el propio
        # simulador/tarea le da a la política (no una medida de un sensor), así que
        # añadirle ruido no modela ninguna incertidumbre física real; sólo haría más
        # difícil aprender a seguirlo. El resto de observaciones sí llevan ruido porque
        # en el robot real vendrían de sensores (odometría, IMU, encoders, FK).
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        tray_pose_command = ObsTerm(
            func=mdp.command_position,
            params={"command_name": "tray_pose"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
 
        # --- Articulaciones del BRAZO únicamente (se excluyen ruedas y dedos de la pinza,
        #     que no forman parte del espacio de acciones ni son relevantes para la política) ---
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*_joint")},
            noise=Unoise(n_min=-0.01, n_max=0.01),  # ruido de encoder, rad
        )
        arm_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*_joint")},
            scale=0.05,
            noise=Unoise(n_min=-0.1, n_max=0.1),  # rad/s
        )
 
        # --- Efector final (bandeja): pose y "equilibrio" (tarea 2 y 3) ---
        # Posición y orientación por separado para poder darles un ruido de escala distinta.
        ee_position_b = ObsTerm(
            func=mdp.ee_position_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link")},
            noise=Unoise(n_min=-0.005, n_max=0.005),  # m
        )
        ee_orientation_b = ObsTerm(
            func=mdp.ee_orientation_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # Gravedad proyectada en el frame de la bandeja: señal directa de cuánto se ha
        # desviado de la horizontal (roll/pitch), clave para la tarea de equilibrio.
        ee_projected_gravity = ObsTerm(
            func=mdp.body_projected_gravity_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
 
        # --- Última acción ---
        # Sin ruido: no es una medida, es el valor exacto que la propia política emitió
        # en el paso anterior (la política y el simulador lo conocen con precisión exacta).
        last_action = ObsTerm(func=mdp.last_action)
 
        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True
 
    policy: PolicyCfg = PolicyCfg()
 
    @configclass
    class CriticCfg(ObsGroup):
        """Observaciones privilegiadas (sólo disponibles en simulación) para el crítico
        en un esquema de entrenamiento actor-crítico asimétrico."""
 
        # Estado base, igual que la política (el crítico necesita al menos la misma
        # información que el actor, sin ruido de sensor)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        tray_pose_command = ObsTerm(
            func=mdp.command_position,
            params={"command_name": "tray_pose"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # --- Articulaciones del BRAZO únicamente (se excluyen ruedas y dedos de la pinza,
        #     que no forman parte del espacio de acciones ni son relevantes para la política) ---
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*_joint")},
            noise=Unoise(n_min=-0.01, n_max=0.01),  # ruido de encoder, rad
        )
        arm_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*_joint")},
            scale=0.05,
            noise=Unoise(n_min=-0.1, n_max=0.1),  # rad/s
        )

        ee_pose_b = ObsTerm(
            func=mdp.ee_pose_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link")},
        )
        ee_projected_gravity = ObsTerm(
            func=mdp.body_projected_gravity_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link")},
        )
 
        # --- Información privilegiada propiamente dicha ---
        # Masa de la carga acoplada al efector final (variable entre episodios, ver eventos).
        # Escalado por el rango de randomización (0-1 kg) para que quede ~[0, 1], en línea
        # con el resto de observaciones.
        tray_payload_mass = ObsTerm(
            func=mdp.body_mass_relative,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link")},
            scale=1.0,
        )
        # Desviación de masa de la base (domain randomization en el chasis, rango -5..15 kg).
        # Sin escalar, esta observación tomaría valores ~15x más grandes que el resto del
        # vector de observación; la escalamos por 1/15 para llevarla a un rango similar.
        base_mass_offset = ObsTerm(
            func=mdp.body_mass_relative,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
            scale=1.0 / 15.0,
        )
        # Velocidad lineal/angular real de la bandeja (no medible con precisión en el robot real)
        ee_velocity = ObsTerm(
            func=mdp.ee_velocity_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link")},
        )
 
        def __post_init__(self):
            self.history_length = 5
            self.concatenate_terms = True
 
    critic: CriticCfg = CriticCfg()

# 4. Configuración de Recompensas (Reward Function)
@configclass
class KairosRewardsCfg:
    # Penalizar movimientos bruscos del brazo
    action_rate_penalty = RewTerm(
        func=mdp.action_rate_l2, weight=-0.01
    )
    arm_joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*_joint")},
    )
    arm_joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*_joint")},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, 
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*_joint")},
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*_joint")},
    )

    # Seguimiento de velocidad de la base
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # Posicionamiento relativo del efector final (bandeja)
    ee_position_tracking = RewTerm(
        func=mdp.ee_position_tracking_exp,
        weight=1.5,
        params={
            "command_name": "tray_pose",
            "std": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link"),
        },
    )

    # Estabilización global de la orientación (equilibrio de bandeja)
    ee_flat_orientation = RewTerm(
        func=mdp.ee_flat_orientation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link")},
    )
    ee_stability = RewTerm(
        func=mdp.ee_velocity_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link")},
    )
    
# 5. Configuración de Terminaciones (Cuándo acaba el episodio)
@configclass
class KairosTerminationsCfg:
    # Terminar por tiempo (timeout)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 2. Arm impact: Terminates if the UR5e arm hits the floor, pyramids, or itself.
    arm_illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("arm_contact_forces"),
            "threshold": 1.0, 
        },
    )

    # 3. Bad orientation: Terminates if the robot tips over beyond a critical angle
    #    even before the base hits the ground (prevents physical instability in PhysX).
    #    Checks the angle between the robot's Z-axis and the world gravity vector.
    vehicle_overturned = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": math.pi / 3, # ~60 degrees tilt limit
        },
    )

    # 4. Tray overturned: termina si la bandeja se inclina más de ~35 grados respecto
    #    a la horizontal (equivalente a `vehicle_overturned` pero para el efector final).
    tray_overturned = DoneTerm(
        func=mdp.ee_bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link"),
            "limit_angle": math.pi / 5,  # ~36 grados
            "up_axis": (0.0, 1.0, 0.0),  # ver comentario en ee_flat_orientation
        },
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-0.75, 0.75)
        ),
    )

    tray_pose = mdp.UniformLevelPoseCommandCfg(
        asset_name="robot",
        body_name="rg6_tcp_link",  # El eslabón que sostiene la bandeja
        resampling_time_range=(5.0, 5.0),  # Cambiar la posición objetivo cada 5s
        debug_vis=False,
        ranges=mdp.UniformLevelPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),   # placeholder, se recalcula en el evento de arranque
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),     # Forzamos que el objetivo siempre sea plano (bandeja recta)
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformLevelPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),   # placeholder, se recalcula en el evento de arranque
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

# 6. Configuración de Eventos (Domain Randomization y Resets)
@configclass
class KairosEventsCfg:
    # reset
    reset_robot_pos = EventTerm(
        func=mdp.custom_reset_kairos,
        mode="reset",
        params={
            "z_clearance": 0.01,
            "asset_cfg": SceneEntityCfg("robot"),
            "yaw_range": (0.0, 0.0)
        },
    )  

    # startup
    randomize_chassis_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-5.0, 15.0), 
            "operation": "add",
        },
    )

    center_tray_pose_command = EventTerm(
        func=mdp.center_command_ranges_on_body,
        mode="startup",
        params={
            "command_name": "tray_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link"),
            "range_offsets": ((0.0, 0.10), (-0.10, 0.10), (-0.10, 0.10)),
            "limit_offsets": ((0.0, 0.60), (-0.35, 0.35), (-0.30, 0.30)),
        },
    )

    randomize_arm_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="arm_(shoulder|upper_arm|forearm|wrist_.*)_link"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range": (0.0, 0.01),
            "num_buckets": 64,
        },
    )

    randomize_tray_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link"),
            "mass_distribution_params": (0.0, 1.0), 
            "operation": "add",
        },
    )

    # interval
    push_robot_base = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}, 
        },
    )

@configclass
class KairosCurriculumCfg:
    """Curriculum terms for terrain advancement."""
    
    terrain_advancement = CurrTerm(
        func=mdp.terrain_distance_curriculum,
        params={
            "distance_threshold": 2.0, 
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Expande el rango de velocidad lineal de la base según el desempeño de seguimiento
    lin_vel_expansion = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={"reward_term_name": "track_lin_vel_xy"},
    )
    
    # Expande el rango de velocidad angular de la base según el desempeño de seguimiento
    ang_vel_expansion = CurrTerm(
        func=mdp.ang_vel_cmd_levels,
        params={"reward_term_name": "track_ang_vel_z"},
    )

    # Expande el rango de posiciones objetivo de la bandeja según el desempeño de tracking
    ee_target_expansion = CurrTerm(
        func=mdp.ee_target_range_curriculum,
        params={
            "command_name": "tray_pose",
            "reward_term_name": "ee_position_tracking",
            "axis_deltas": {"pos_x": (0.0, 0.05), "pos_y": (-0.05, 0.05), "pos_z": (-0.05, 0.05)},
        },
    ) 


# ==============================================================================
# CONFIGURACIÓN PRINCIPAL DEL ENTORNO
# ==============================================================================
@configclass
class KairosEnvCfg(ManagerBasedRLEnvCfg):

    scene = KairosSceneCfg(num_envs=4096, env_spacing=2.5)
    actions = KairosActionsCfg()
    observations = KairosObservationsCfg()
    rewards = KairosRewardsCfg()
    terminations = KairosTerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    events = KairosEventsCfg()
    curriculum: KairosCurriculumCfg = KairosCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # Parámetros del episodio
        self.episode_length_s = 20.0 # Duración en segundos
        
        # Configuración del motor de físicas
        self.sim.dt = 1.0 / 60.0 # Frecuencia de física
        self.decimation = 2 # Frecuencia de control de la red neuronal

        # CONFIGURACIÓN DE LA CÁMARA
        self.viewer.eye = (-35.5, -35.5, 5.0)   # Posición de la cámara (X, Y, Z)
        self.viewer.lookat = (-20.0, -20.0, 1.0) # A dónde mira la cámara (hacia el robot)


@configclass
class RobotPlayEnvCfg(KairosEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges