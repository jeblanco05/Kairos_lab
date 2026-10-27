import sys
import os
import argparse
from isaaclab.app import AppLauncher

# 1. Configuración de argumentos (Añadimos flags para video)
parser = argparse.ArgumentParser(description="Visualizar y grabar el entorno del Kairos+.")
parser.add_argument("--record_video", action="store_true", default=False, help="Grabar video de la simulación en .mp4")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Si queremos grabar video en un servidor sin pantalla, necesitamos forzar las cámaras virtuales
if args_cli.record_video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from isaaclab.envs import ManagerBasedRLEnv
# Importamos la configuración que creamos en el paso anterior
from my_projects.Kairos_lab.train_lab.tasks.kairosplus_env_cfg import KairosEnvCfg
from tasks.Go2.env_cfg import GO2EnvCfg
from my_projects.Kairos_lab.train_lab.tasks.mdp import custom_reset_kairos, projected_gravity
from isaaclab.managers import SceneEntityCfg

def reset_report(uenv):
    r = uenv.scene["robot"]; d = r.data
    o = uenv.scene.env_origins[0]
    feet = r.find_bodies(".*_foot_link")[0]
    legs = r.find_joints(".*_(hip|upper_leg|lower_leg)_joint")[0]
    gap = (d.body_pos_w[0, feet, 2].min() - o[2] - 0.022).item()
    print(f"[] hueco_pie={gap:+.3f}  |v|={d.root_lin_vel_w[0].norm():.2f}  "
          f"|w|={d.root_ang_vel_w[0].norm():.2f}  "
          f"dev_patas={(d.joint_pos[0, legs] - d.default_joint_pos[0, legs]).abs().max():.2f}")
          
def main():
    # 2. Cargar la configuración
    env_cfg = GO2EnvCfg()
    env_cfg.scene.num_envs = 1

    # --- Terreno plano y sin currículum para depurar ---
    # env_cfg.scene.terrain.terrain_type = "plane"
    # env_cfg.scene.terrain.terrain_generator = None
    # if hasattr(env_cfg.curriculum, "terrain_levels"):
    #     env_cfg.curriculum.terrain_levels = None

    # --- Comandos controlados por ti, no aleatorios ---
    cmd = env_cfg.commands.base_velocity
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    cmd.resampling_time_range = (1e9, 1e9)
    cmd.ranges.lin_vel_x = (0.0, 0.0)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.ranges.heading = (0.0, 0.0)

    # --- Cámara que sigue al robot ---
    # env_cfg.viewer.origin_type = "asset_root"
    # env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (-10.0, 0.0, 20.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
    
    # 3. Crear el entorno (Activamos rgb_array para poder renderizar video)
    render_mode = "rgb_array" if args_cli.record_video else None
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=render_mode)
    
    # 4. Configurar el grabador de vídeo
    if args_cli.record_video:
        video_folder = os.path.join(os.getcwd(), "videos_go2")
        print(f"\n[INFO] Grabando video en la carpeta: {video_folder}\n")
        
        env = RecordVideo(
            env, 
            video_folder=video_folder, 
            step_trigger=lambda step: step == 0, # Grabar desde el frame 0
            video_length=400 # Duración: 200 pasos (a 50Hz = 4 segundos de video)
        )

    # 5. Bucle de Simulación
    #print(env.scene.env_origins)
    obs, info = env.reset()
    print("[INFO] Simulación iniciada...")
    
    # Vamos a correr 200 pasos de simulación
    pos = 0.0
    for i in range(400):
        #print(i)
        # if i%100 == 0:
        #     print(f"[INFO] Teletransportando al robot en el frame {i}...")
        #     # Creamos un tensor que apunta a nuestro único robot (índice 0)
        #     #ids_robot = torch.tensor([0], device=env.device)
        #     ids_robot = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        #     # Llamamos a tu función para que lo mueva, ¡sin reiniciar el entorno entero!
        #     custom_reset_kairos(env.unwrapped, ids_robot)
        #     pos = 0.0
        #     reset_report(env)
        # env.action_space.shape[1] será 9 (3 para la base + 6 para el brazo)
        # Inicializamos todo a 0.0 (El robot se quedará completamente quieto)
        acciones = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)
        
        # --- ¡JUEGA CON LAS ACCIONES AQUÍ! ---
        # Si quieres probar a mover la base hacia adelante a 0.5 m/s, descomenta esta línea:
        #acciones[:, 0] = 5.8

        # Si quieres probar a mover la base hacia el lado a 0.5 m/s, descomenta esta:
        #acciones[:, 1] = 0.5
        
        # Si quieres probar a girar la base sobre su eje (Yaw), descomenta esta:
        #acciones[:, 2] = 0.5
        
        # Si quieres mover la primera articulación del brazo (Hombro Pan), descomenta esta:
        pos += 0.2
        # acciones[:, 0] += pos
        # acciones[:, 1] += -pos
        # acciones[:, 2] += -pos
        # acciones[:, 3] += -pos
        # acciones[:, 4] += -pos
        #acciones[:, 5] += -pos
        #print("Accion: ", i)
        # Enviamos la acción al simulador
        obs, rewards, terminated, truncated, info = env.step(acciones)

        dones = terminated | truncated
        if dones.any():
            print("[INFO] El entorno detectó un choque y se ha reiniciado. Limpiando acciones...")
            tm = env.termination_manager
            print(f"[RESET] paso {i}:", [n for n in tm.active_terms if tm.get_term(n).any()])
            pos = 0.0
            
        #     ids_robot = torch.tensor([0], device=env.device)
        #     tray_cfg = SceneEntityCfg("robot", body_names="rg6_tcp_link")
            
        #     # 2. Resolver los IDs contra la escena actual para evitar el error 'slice'
        #     tray_cfg.resolve(env.unwrapped.scene)
            
        #     # 3. Llamar a la función con el asset_cfg configurado
        #     gravedad_cuerpo= projected_gravity(env.unwrapped)
        #     print("gravedad_cuerpo", gravedad_cuerpo)
        
    print("[INFO] Simulación terminada.")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()