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

from robot_cfg.robotcfg import KAIROS_CFG, KAIROS_CFG_RG6
from controllers.local_base_action import LocalBaseVelocityActionCfg, MecanumChassisVelocityActionCfg
from tasks import mdp 

# 1. Terrain configuration
TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
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
    curriculum=True, # Make sure this is True
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
        max_init_terrain_level=0, #TODO
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
            texture_file=f"{ISAACLAB_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
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

# 2. Action configuration (Action Space)
@configclass
class KairosActionsCfg:
    # Action for the base (virtual wheel joint velocity control)
    base_velocity = MecanumChassisVelocityActionCfg()

    # Action for the arm (UR5e joint position control)
    arm_position = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["arm_.*_joint"], scale=0.05, use_default_offset=True
    )

# 3. Observation configuration (Observation Space)
@configclass
class KairosObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations also available on the real robot (without privileged information)."""
 
        # --- Base state ---
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # Base tilt with respect to gravity (IMU)
        base_projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
 
        # --- Commands (task 1: base velocity; task 2: target tray pose) ---
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        tray_pose_command = ObsTerm(func=mdp.command_position,params={"command_name": "tray_pose"},noise=Unoise(n_min=-0.01, n_max=0.01),)
 
        # --- ARM joints only (excluding wheels and gripper fingers,
        #     which are not part of the action space and are not relevant to policy learning) ---
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*_joint")},
            noise=Unoise(n_min=-0.01, n_max=0.01),  # encoder noise, rad
        )
        arm_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*_joint")},
            scale=0.05,
            noise=Unoise(n_min=-0.1, n_max=0.1),  # rad/s
        )
 
        # --- End effector (tray): pose and "balance" (tasks 2 and 3) ---
        # Position and orientation separately so they can receive noise at different scales.
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
        # Gravity projected into the tray frame: direct signal of how far it has deviated from
        # horizontal (roll/pitch), key for the balance task.
        ee_projected_gravity = ObsTerm(
            func=mdp.body_projected_gravity_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
 
        # --- Last action ---
        last_action = ObsTerm(func=mdp.last_action)
 
        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True
 
    policy: PolicyCfg = PolicyCfg()
 
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for the critic in an asymmetric actor-critic training setup."""
 
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        tray_pose_command = ObsTerm(func=mdp.command_position,params={"command_name": "tray_pose"},)

        arm_joint_pos = ObsTerm(func=mdp.joint_pos_rel,params={"asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*_joint")},)
        arm_joint_vel = ObsTerm(func=mdp.joint_vel_rel,params={"asset_cfg": SceneEntityCfg("robot", joint_names="arm_.*_joint")},scale=0.05,)

        ee_projected_gravity = ObsTerm(func=mdp.body_projected_gravity_b,params={"asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link")},)

        last_action = ObsTerm(func=mdp.last_action)
 
        # --- Privileged information ---
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        ee_pose_b = ObsTerm(func=mdp.ee_pose_b,params={"asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link")},)
        
        # Mass of the payload attached to the end effector (varies between episodes, see events).
        tray_payload_mass = ObsTerm(func=mdp.body_mass_relative,params={"asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link")},)
        # Base mass deviation (domain randomization on the chassis, range -5..15 kg).
        base_mass_offset = ObsTerm(
            func=mdp.body_mass_relative,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
            scale=1.0 / 15.0,
        )
        # Real linear/angular tray velocity (not measurable with precision on the real robot)
        ee_velocity = ObsTerm(
            func=mdp.ee_velocity_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link")},
        )
 
        def __post_init__(self):
            self.history_length = 5
            self.concatenate_terms = True
 
    critic: CriticCfg = CriticCfg()

# 4. Reward configuration (Reward Function)
@configclass
class KairosRewardsCfg:
    # Penalize abrupt arm movements
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

    # Base velocity tracking
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # Relative positioning of the end effector (tray)
    # ee_position_tracking = RewTerm(
    #     func=mdp.ee_position_tracking_exp,
    #     weight=1.5,
    #     params={
    #         "command_name": "tray_pose",
    #         "std": 0.2,
    #         "asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link"),
    #     },
    # )

    ee_position_tracking = RewTerm(
        func=mdp.ee_position_tracking_exp_bubble,
        weight=1.5,
        params={
            "command_name": "tray_pose",
            "std": 0.2,
            "tolerance": 0.05, # Radio de la burbuja en metros
            "asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link"),
        },
    )

    # Global orientation stabilization (tray balance)
    ee_flat_orientation = RewTerm(
        func=mdp.ee_flat_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link"),"up_axis": (0.0, 1.0, 0.0)},
    )
    ee_velocity = RewTerm(
        func=mdp.ee_velocity_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link")},
    )
    
# 5. Termination configuration (When the episode ends)
@configclass
class KairosTerminationsCfg:
    # End by time (timeout)
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
            "limit_angle": 60*math.pi / 180, # ~60 degrees tilt limit
        },
    )

    # 4. Tray overturned: terminates if the tray tilts more than ~30 degrees relative
    #    to horizontal (equivalent to `vehicle_overturned` but for the end effector).
    tray_overturned = DoneTerm(
        func=mdp.ee_bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="rg6_tcp_link"),
            "limit_angle": 30*math.pi / 180,  # ~30 degrees
            "up_axis": (0.0, 1.0, 0.0),  # see comment in ee_flat_orientation
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
        body_name="rg6_tcp_link",  # The link that holds the tray
        resampling_time_range=(5.0, 5.0),  # Change the target position every 5 s
        debug_vis=False,
        ranges=mdp.UniformLevelPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),   # placeholder, recalculated in the startup event
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),     # Force the target to remain flat (tray aligned)
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformLevelPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),   # placeholder, recalculated in the startup event
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

# 6. Event configuration (Domain Randomization and Resets)
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
            "range_offsets": ((0.0, 0.15), (-0.10, 0.10), (-0.10, 0.10)),#TODO
            "limit_offsets": ((0.0, 0.30), (-0.25, 0.25), (-0.20, 0.20)),#TODO
            #"range_offsets": ((0.0, 0.0), (-0.00, 0.00), (-0.00, 0.00)),
            #"limit_offsets": ((0.0, 0.0), (-0.0, 0.0), (-0.00, 0.0)),
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
    )#TODO

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
    
    # terrain_advancement = CurrTerm(
    #     func=mdp.terrain_distance_curriculum,
    #     params={
    #         "distance_threshold": 2.0, 
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )
    terrain_advancement = CurrTerm(func=mdp.terrain_curriculum,)

    # Expand the base linear velocity command range according to tracking performance
    lin_vel_expansion = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={"reward_term_name": "track_lin_vel_xy"},
    )
    
    # Expand the base angular velocity command range according to tracking performance
    ang_vel_expansion = CurrTerm(
        func=mdp.ang_vel_cmd_levels,
        params={"reward_term_name": "track_ang_vel_z"},
    )

    # Expand the tray target position range according to tracking performance
    ee_target_expansion = CurrTerm(
        func=mdp.ee_target_range_curriculum,
        params={
            "command_name": "tray_pose",
            "reward_term_name": "ee_position_tracking",
            "axis_deltas": {"pos_x": (0.0, 0.05), "pos_y": (-0.05, 0.05), "pos_z": (-0.05, 0.05)},
        },
    )# TODO


# ==============================================================================
# MAIN ENVIRONMENT CONFIGURATION
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
        # Episode parameters
        self.episode_length_s = 20.0 # Duration in seconds
        
        # Physics engine configuration
        self.sim.dt = 1.0 / 60.0 # Physics frequency
        self.decimation = 2 # Neural control frequency

        # CAMERA CONFIGURATION
        #self.viewer.eye = (-35.5, -35.5, 5.0)   # Camera position (X, Y, Z)
        #self.viewer.lookat = (-20.0, -20.0, 1.0) # Where the camera looks (towards the robot)
        self.viewer.eye = (48.0, 0.0, 6.0)
        self.viewer.lookat = (-10.0, -5.0, -5.0)

@configclass
class RobotPlayEnvCfg(KairosEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 30
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

        self.scene.terrain.terrain_generator.curriculum = False
        
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_sloped"].slope_range = (0.24, 0.24)
        self.scene.terrain.terrain_generator.sub_terrains["inverted_pyramid_sloped"].slope_range = (0.24, 0.24)
        
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 6 
        self.scene.terrain.max_init_terrain_level = None
        
        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None
        self.viewer.eye = (10.0, 0.0, 5.0)
        self.viewer.lookat = (-4.0, 1.0, 0.0)