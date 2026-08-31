import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


KAIROS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Path to your USD generated on the cluster
        usd_path="/mnt/beegfs/home/jesuseliseo.blanco/my_projects/Kairos_lab/assets/rbkairos_plus/rbkairos_plus.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={         
            # UR5e arm (safe initial "L" pose so it does not collide with the base on spawn)
            "arm_shoulder_pan_joint": 0.0,
            "arm_shoulder_lift_joint": -1.57,  # -90 degrees (raised)
            "arm_elbow_joint": 1.57,           # 90 degrees
            "arm_wrist_1_joint": 0.00,
            "arm_wrist_2_joint": 1.57,
            "arm_wrist_3_joint": 0.0,
            ".*_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Omnidirectional base actuators
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=100.0,
            velocity_limit=20.0,
            stiffness=0.0,
            damping=0.0,
        ),
        # Main arm actuators (shoulder and elbow)
        "ur5e_shoulder": ImplicitActuatorCfg(
            # Exact names from your previous output
            joint_names_expr=["arm_shoulder_pan_joint", "arm_shoulder_lift_joint", "arm_elbow_joint"],
            effort_limit=150.0,  # The UR5e has max 150 Nm in the large joints
            velocity_limit=3.14, # ~180 degrees per second
            stiffness=800.0,     # PD control stiffness (adjustable for RL)
            damping=40.0,        # PD control damping
        ),
        # Wrist actuators (weaker)
        "ur5e_wrist": ImplicitActuatorCfg(
            # Regular expression to match wrist_1, wrist_2, and wrist_3
            joint_names_expr=["arm_wrist_.*_joint"],
            effort_limit=28.0,   # Wrists support max 28 Nm
            velocity_limit=3.14,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)

KAIROS_CFG_RG6 = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Path to USD
        usd_path="/mnt/beegfs/home/jesuseliseo.blanco/my_projects/Kairos_lab/assets/rbkairos_plus_rg6/rbkairos_plus_rg6.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={         
            # UR5e arm (safe initial "L" pose so it does not collide with the base on spawn)
            "arm_shoulder_pan_joint": 0.0,     # 0.0
            "arm_shoulder_lift_joint": -1.57,  # -1.57 degrees (raised)
            "arm_elbow_joint": 1.57,           # 1.57 degrees
            "arm_wrist_1_joint": 0.00,         # 0.0
            "arm_wrist_2_joint": 1.57,         # 1.57
            "arm_wrist_3_joint": 0.0,          # 0.0
            ".*_wheel_joint": 0.0,             # 0.0
            "rg6_.*_finger_joint": 0.04,       # limits [0.000, 0.080]
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Omnidirectional base actuators
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=100.0,
            velocity_limit=20.0,
            stiffness=0.0,
            damping=0.0,
        ),
        # Main arm actuators (shoulder and elbow)
        "ur5e_shoulder": ImplicitActuatorCfg(
            # Exact names from your previous output
            joint_names_expr=["arm_shoulder_pan_joint", "arm_shoulder_lift_joint", "arm_elbow_joint"],
            effort_limit=150.0,  # The UR5e has max 150 Nm in the large joints
            velocity_limit=3.14, # ~180 degrees per second
            stiffness=800.0,     # PD control stiffness (adjustable for RL)
            damping=40.0,        # PD control damping
        ),
        # Wrist actuators (weaker)
        "ur5e_wrist": ImplicitActuatorCfg(
            # Regular expression to match wrist_1, wrist_2, and wrist_3
            joint_names_expr=["arm_wrist_.*_joint"],
            effort_limit=28.0,   # Wrists support max 28 Nm
            velocity_limit=3.14,
            stiffness=200.0,
            damping=20.0,
        ),
        "rg6_gripper": ImplicitActuatorCfg(
            joint_names_expr=["rg6_.*_finger_joint"],
            stiffness=1000.0, # Very high stiffness
            damping=100.0,    # High damping to avoid oscillations
        ),
    },
)

