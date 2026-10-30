#!/usr/bin/env python3

"""
Launch completo de la simulación KAIROS+.

Arranca, en este orden lógico:
  1. Gazebo Sim + mundo
  2. KAIROS+ + UR5 + OnRobot RG6 + controladores
  3. Cubo como modelo físico independiente, colocado sobre la bandeja
  4. Nav2

El robot sigue siendo el modelo URDF/Xacro y el cubo es un modelo SDF
independiente de Gazebo. Esto permite que el cubo tenga gravedad, colisión
y fricción y pueda deslizarse o caer de la bandeja.
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ---------------------------------------------------------------------
    # Paquetes
    # ---------------------------------------------------------------------
    kairos_deploy_share = get_package_share_directory("kairos_deploy")
    robotnik_gazebo_share = get_package_share_directory("robotnik_gazebo_ignition")

    spawn_world_launch = os.path.join(
        robotnik_gazebo_share,
        "launch",
        "spawn_world.launch.py",
    )

    spawn_robot_launch = os.path.join(
        robotnik_gazebo_share,
        "launch",
        "spawn_robot.launch.py",
    )

    nav2_launch = os.path.join(
        kairos_deploy_share,
        "launch",
        "nav2_full.launch.py",
    )

    cube_sdf = os.path.join(
        kairos_deploy_share,
        "worlds",
        "tray_cube",
        "model.sdf",
    )

    # ---------------------------------------------------------------------
    # Argumentos
    # ---------------------------------------------------------------------
    world_path = LaunchConfiguration("world_path")
    robot_x = LaunchConfiguration("x")
    robot_y = LaunchConfiguration("y")
    robot_z = LaunchConfiguration("z")
    robot_yaw = LaunchConfiguration("yaw")

    launch_nav2 = LaunchConfiguration("launch_nav2")
    spawn_cube = LaunchConfiguration("spawn_cube")
    run_rviz = LaunchConfiguration("run_rviz")

    cube_height = LaunchConfiguration("cube_height")
    cube_x_offset = LaunchConfiguration("cube_x_offset")
    cube_y_offset = LaunchConfiguration("cube_y_offset")

    base_frame = LaunchConfiguration("base_frame")
    tray_frame = LaunchConfiguration("tray_frame")

    # ---------------------------------------------------------------------
    # Resource paths de Gazebo
    #
    # spawn_robot.launch.py ya configura varios paquetes, pero aquí también
    # añadimos explícitamente los paquetes propios del proyecto para que el
    # modelo SDF del cubo y sus recursos se encuentren de forma determinista.
    # ---------------------------------------------------------------------

    kairos_deploy_parent_share = str(Path(kairos_deploy_share).parent)

    current_gz_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    resource_entries = [kairos_deploy_parent_share]

    if current_gz_path:
        resource_value = ":".join(resource_entries + [current_gz_path])
    else:
        resource_value = ":".join(resource_entries)

    set_gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=resource_value,
    )

    # ---------------------------------------------------------------------
    # Gazebo + mundo
    # ---------------------------------------------------------------------
    world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(spawn_world_launch),
        launch_arguments={
            "world_path": world_path,
        }.items(),
    )

    # ---------------------------------------------------------------------
    # Robot KAIROS+
    #
    # El launch oficial de Robotnik ya:
    #   - procesa el Xacro,
    #   - publica robot_description,
    #   - hace spawn con ros_gz_sim,
    #   - crea el bridge,
    #   - lanza los controladores,
    #   - opcionalmente lanza RViz.
    # ---------------------------------------------------------------------
    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(spawn_robot_launch),
        launch_arguments={
            "robot_id": "robot",
            "robot": "rbkairos",
            "robot_model": "rbkairos_plus",
            "robot_xacro_path": os.path.join(
                kairos_deploy_share,
                "worlds",
                "robots",
                "rbkairos_ur5_rg6.urdf.xacro",
            ),
            "initial_positions_file": kairos_deploy_share + "/config/initial_positions.yaml",
            "x": robot_x,
            "y": robot_y,
            "z": robot_z,
            "arm_type": "ur5",
            "gripper_type": "onrobot_rg6",
            "run_rviz": run_rviz,
            "use_sim_time": "true",
            "low_performance_simulation": "false",
        }.items(),
    )

    # ---------------------------------------------------------------------
    # Nav2
    #
    # Se arranca después de la simulación/robot para evitar condiciones de
    # carrera durante la inicialización de TF, odometría y controladores.
    # ---------------------------------------------------------------------
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch),
    )

    delayed_nav2 = TimerAction(
        period=8.0,
        actions=[nav2],
        condition=IfCondition(launch_nav2),
    )

    # ---------------------------------------------------------------------
    # Spawn del cubo sobre la bandeja
    #
    # Se espera a que Gazebo y el robot estén completamente iniciados.
    # El propio nodo espera además a que exista TF base -> bandeja.
    # ---------------------------------------------------------------------
    cube_node = Node(
        package="kairos_deploy",
        executable="spawn_cube_on_tray.py",
        name="spawn_cube_on_tray",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "robot_x": robot_x,
                "robot_y": robot_y,
                "robot_z": robot_z,
                "robot_yaw": robot_yaw,
                "base_frame": base_frame,
                "tray_frame": tray_frame,
                "cube_height": cube_height,
                "cube_x_offset": cube_x_offset,
                "cube_y_offset": cube_y_offset,
                "cube_sdf": cube_sdf,
            }
        ],
        condition=IfCondition(spawn_cube),
    )

    delayed_cube = TimerAction(
        period=10.0,
        actions=[cube_node],
    )

    return LaunchDescription(
        [
            # ---------------------------
            # Argumentos públicos
            # ---------------------------
            DeclareLaunchArgument(
                "world_path",
                default_value=os.path.join(
                    kairos_deploy_share, "worlds", "piramids.sdf"
                ),
                description="Ruta absoluta al mundo SDF de Gazebo.",
            ),
            DeclareLaunchArgument(
                "x",
                default_value="-7.5",
                description="Posición X inicial del robot.",
            ),
            DeclareLaunchArgument(
                "y",
                default_value="0.0",
                description="Posición Y inicial del robot.",
            ),
            DeclareLaunchArgument(
                "z",
                default_value="0.01",
                description="Posición Z inicial del robot.",
            ),
            DeclareLaunchArgument(
                "yaw",
                default_value="0.0",
                description="Yaw inicial del robot en radianes.",
            ),
            DeclareLaunchArgument(
                "launch_nav2",
                default_value="true",
                description="Lanzar Nav2 automáticamente.",
            ),
            DeclareLaunchArgument(
                "spawn_cube",
                default_value="true",
                description="Crear el cubo físico sobre la bandeja.",
            ),
            DeclareLaunchArgument(
                "run_rviz",
                default_value="true",
                description="Lanzar RViz desde el launch del robot.",
            ),
            DeclareLaunchArgument(
                "base_frame",
                default_value="robot_base_link",
                description="Frame base del robot con frame_prefix aplicado.",
            ),
            DeclareLaunchArgument(
                "tray_frame",
                default_value="tray_link",
                description="Frame TF de la bandeja con frame_prefix aplicado.",
            ),
            DeclareLaunchArgument(
                "cube_height",
                default_value="0.055",
                description=(
                    "Altura del centro del cubo sobre el frame de la bandeja "
                    "en metros."
                ),
            ),
            DeclareLaunchArgument(
                "cube_x_offset",
                default_value="0.0",
                description="Offset X del centro del cubo en el frame de bandeja.",
            ),
            DeclareLaunchArgument(
                "cube_y_offset",
                default_value="0.0",
                description="Offset Y del centro del cubo en el frame de bandeja.",
            ),

            # ---------------------------
            # Inicialización
            # ---------------------------
            set_gz_resource_path,
            world,

            # El robot comienza después de que el mundo haya empezado.
            TimerAction(
                period=4.0,
                actions=[robot],
            ),

            # Objetos/Nav2
            delayed_nav2,
            delayed_cube,
            
        ]
    )