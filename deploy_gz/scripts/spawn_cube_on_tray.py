#!/usr/bin/env python3

"""
Coloca y hace spawn de un cubo SDF sobre la bandeja del robot.

El nodo:
  1. espera a TF entre base y bandeja,
  2. calcula la pose de la bandeja en coordenadas de mundo,
  3. aplica el offset del cubo,
  4. ejecuta ros_gz_sim create con la pose calculada.

El cubo NO forma parte del URDF del robot.
"""

import math
import os
import subprocess
import time

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import tf2_ros
from ament_index_python.packages import get_package_share_directory

def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def rotate_2d(x, y, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * x - s * y, s * x + c * y


class SpawnCubeOnTray(Node):

    def __init__(self):
        super().__init__("spawn_cube_on_tray")

        self.declare_parameter("robot_x", -7.5)
        self.declare_parameter("robot_y", 0.0)
        self.declare_parameter("robot_z", 0.01)
        self.declare_parameter("robot_yaw", 0.0)

        self.declare_parameter("base_frame", "robot_base_link")
        self.declare_parameter("tray_frame", "tray_link")

        self.declare_parameter("cube_height", 0.055)
        self.declare_parameter("cube_x_offset", 0.0)
        self.declare_parameter("cube_y_offset", 0.0)

        self.declare_parameter("cube_sdf", "")

        self.robot_x = float(self.get_parameter("robot_x").value)
        self.robot_y = float(self.get_parameter("robot_y").value)
        self.robot_z = float(self.get_parameter("robot_z").value)
        self.robot_yaw = float(self.get_parameter("robot_yaw").value)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tray_frame = str(self.get_parameter("tray_frame").value)

        self.cube_height = float(self.get_parameter("cube_height").value)
        self.cube_x_offset = float(self.get_parameter("cube_x_offset").value)
        self.cube_y_offset = float(self.get_parameter("cube_y_offset").value)

        pkg_share_dir = get_package_share_directory('kairos_deploy')
        self.cube_sdf = os.path.join(pkg_share_dir, 'worlds', 'tray_cube', 'model.sdf')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.spawned = False
        self.started_at = time.monotonic()

        self.timer = self.create_timer(0.25, self.try_spawn)


    def try_spawn(self):
        if self.spawned:
            return

        if not self.cube_sdf or not os.path.isfile(self.cube_sdf):
            self.get_logger().error(
                f"No existe el SDF del cubo: {self.cube_sdf}")
            self.timer.cancel()
            return

        # Transformación bandeja respecto a la base.
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,     # target_frame: robot_base_link
                self.tray_frame,     # source_frame: robot_tray_link
                rclpy.time.Time()
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"Esperando transform de {self.base_frame} a {self.tray_frame}: {e}")
            return  # Salir de este ciclo del timer y volver a intentar

        t = transform.transform.translation
        q = transform.transform.rotation

        tray_x_local = float(t.x)
        tray_y_local = float(t.y)
        tray_z_local = float(t.z)

        tray_yaw_local = quat_to_yaw(q.x, q.y, q.z, q.w)

        # La base del robot se coloca en (robot_x, robot_y, robot_z)
        # con yaw robot_yaw. Convertimos la posición de la bandeja al mundo.
        tray_x_rot, tray_y_rot = rotate_2d(
            tray_x_local,
            tray_y_local,
            self.robot_yaw,
        )

        world_x = self.robot_x + tray_x_rot
        world_y = self.robot_y + tray_y_rot
        world_z = self.robot_z + tray_z_local + self.cube_height + 0.1

        world_yaw = self.robot_yaw + tray_yaw_local

        self.get_logger().info(
            f"Pose calculada del cubo: x={world_x} y={world_y} z={world_z} yaw={world_yaw}")

        # Convertimos offsets X/Y definidos en el frame de bandeja.
        offset_x_rot, offset_y_rot = rotate_2d(
            self.cube_x_offset,
            self.cube_y_offset,
            world_yaw,
        )

        world_x += offset_x_rot + 0.14
        world_y += offset_y_rot

        cmd = [
            "ros2",
            "run",
            "ros_gz_sim",
            "create",
            "-name",
            "tray_cube",
            "-file",
            self.cube_sdf,
            "-x",
            f"{world_x:.6f}",
            "-y",
            f"{world_y:.6f}",
            "-z",
            f"{world_z:.6f}",
            "-R",
            "0.0",
            "-P",
            "0.0",
            "-Y",
            f"{world_yaw:.6f}",
        ]

        self.get_logger().info("Ejecutando spawn del cubo...")

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=30.0,
                env=os.environ.copy(),
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            self.get_logger().error(f"Error ejecutando ros_gz_sim create: {exc}")
            self.timer.cancel()
            return

        if result.returncode != 0:
            self.get_logger().error(
                "No se pudo crear el cubo.\nstdout:\n%s\nstderr:\n%s",
                result.stdout,
                result.stderr,
            )
            self.timer.cancel()
            return

        self.get_logger().info("Cubo creado correctamente sobre la bandeja.")
        self.spawned = True
        self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = SpawnCubeOnTray()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
