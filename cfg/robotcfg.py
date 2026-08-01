import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


KAIROS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # 1. Ruta a tu USD generado en el clúster
        usd_path="/mnt/beegfs/home/jesuseliseo.blanco/my_projects/Kairos_lab/assets/rbkairos_plus/rbkairos_plus.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={         
            # 3. Brazo UR5e (Posición inicial segura tipo "L" para que no choque con la base al spawnear)
            "arm_shoulder_pan_joint": 0.0,
            "arm_shoulder_lift_joint": -1.57,  # -90 grados (Levantado)
            "arm_elbow_joint": 1.57,           # 90 grados
            "arm_wrist_1_joint": 0.00,
            "arm_wrist_2_joint": 1.57,
            "arm_wrist_3_joint": 0.0,
            ".*_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 4. Actuadores de la Base Omnidireccional
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=100.0,
            velocity_limit=20.0,
            stiffness=0.0,
            damping=0.0,
        ),
        # 5. Actuadores Principales del Brazo (Hombro y Codo)
        "ur5e_shoulder": ImplicitActuatorCfg(
            # Nombres exactos de tu output anterior
            joint_names_expr=["arm_shoulder_pan_joint", "arm_shoulder_lift_joint", "arm_elbow_joint"],
            effort_limit=150.0,  # El UR5e tiene máx 150 Nm en las juntas grandes
            velocity_limit=3.14, # ~180 grados por segundo
            stiffness=800.0,     # Rigidez del control PD (ajustable para RL)
            damping=40.0,        # Amortiguación del control PD
        ),
        # 6. Actuadores de la Muñeca (Más débiles)
        "ur5e_wrist": ImplicitActuatorCfg(
            # Expresión regular para pillar wrist_1, wrist_2 y wrist_3
            joint_names_expr=["arm_wrist_.*_joint"],
            effort_limit=28.0,   # Las muñecas del UR5e soportan máx 28 Nm
            velocity_limit=3.14,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)

KAIROS_CFG_RG6 = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # 1. Ruta a tu USD generado en el clúster
        usd_path="/mnt/beegfs/home/jesuseliseo.blanco/my_projects/Kairos_lab/assets/rbkairos_plus_rg6/rbkairos_plus_rg6.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={         
            # 3. Brazo UR5e (Posición inicial segura tipo "L" para que no choque con la base al spawnear)
            "arm_shoulder_pan_joint": 0.0,
            "arm_shoulder_lift_joint": -1.57,  # -90 grados (Levantado)
            "arm_elbow_joint": 1.57,           # 90 grados
            "arm_wrist_1_joint": 0.00,
            "arm_wrist_2_joint": 1.57,
            "arm_wrist_3_joint": 0.0,
            ".*_wheel_joint": 0.0,
            "rg6_.*_finger_joint": 0.04, # limits [0.000, 0.080]
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 4. Actuadores de la Base Omnidireccional
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=100.0,
            velocity_limit=20.0,
            stiffness=0.0,
            damping=0.0,
        ),
        # 5. Actuadores Principales del Brazo (Hombro y Codo)
        "ur5e_shoulder": ImplicitActuatorCfg(
            # Nombres exactos de tu output anterior
            joint_names_expr=["arm_shoulder_pan_joint", "arm_shoulder_lift_joint", "arm_elbow_joint"],
            effort_limit=150.0,  # El UR5e tiene máx 150 Nm en las juntas grandes
            velocity_limit=3.14, # ~180 grados por segundo
            stiffness=5000.0,     # Rigidez del control PD (ajustable para RL)
            damping=200.0,        # Amortiguación del control PD
        ),
        # 6. Actuadores de la Muñeca (Más débiles)
        "ur5e_wrist": ImplicitActuatorCfg(
            # Expresión regular para pillar wrist_1, wrist_2 y wrist_3
            joint_names_expr=["arm_wrist_.*_joint"],
            effort_limit=28.0,   # Las muñecas del UR5e soportan máx 28 Nm
            velocity_limit=3.14,
            stiffness=3000.0,
            damping=100.0,
        ),
        "rg6_gripper": ImplicitActuatorCfg(
            joint_names_expr=["rg6_.*_finger_joint"],
            stiffness=1000.0, # Rigidez muy alta
            damping=100.0,    # Amortiguamiento alto para evitar oscilaciones
        ),
    },
)

