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
from tasks.kairosplus_env_cfg import KairosEnvCfg
from tasks.mdp import custom_reset_kairos, projected_gravity
from isaaclab.managers import SceneEntityCfg
def main():
    # 2. Cargar la configuración
    env_cfg = KairosEnvCfg()
    
    # Para probar visualmente, solo necesitamos 1 robot, no 4000.
    env_cfg.scene.num_envs = 4000
    
    # Configuramos dónde se sitúa la cámara virtual que nos grabará el video
    env_cfg.viewer.eye = (48.0, 0.0, 6.0)   # Posición de la cámara (X, Y, Z)
    env_cfg.viewer.lookat = (-10.0, -5.0, -5.0) # A dónde mira la cámara (hacia el robot)
    
    # 3. Crear el entorno (Activamos rgb_array para poder renderizar video)
    render_mode = "rgb_array" if args_cli.record_video else None
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=render_mode)
    
    # 4. Configurar el grabador de vídeo
    if args_cli.record_video:
        video_folder = os.path.join(os.getcwd(), "videos_kairos")
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
    for i in range(600):
        # if i%100 == 0:
        #     print(f"[INFO] Teletransportando al robot en el frame {i}...")
        #     # Creamos un tensor que apunta a nuestro único robot (índice 0)
        #     #ids_robot = torch.tensor([0], device=env.device)
        #     ids_robot = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        #     # Llamamos a tu función para que lo mueva, ¡sin reiniciar el entorno entero!
        #     custom_reset_kairos(env.unwrapped, ids_robot)
        #     pos = 0.0
        #env.action_space.shape[1] será 9 (3 para la base + 6 para el brazo)
        # Inicializamos todo a 0.0 (El robot se quedará completamente quieto)
        acciones = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)
        
        # --- ¡JUEGA CON LAS ACCIONES AQUÍ! ---
        # Si quieres probar a mover la base hacia adelante a 0.5 m/s, descomenta esta línea:
        acciones[:, 0] = 1.5

        # Si quieres probar a mover la base hacia el lado a 0.5 m/s, descomenta esta:
        #acciones[:, 1] = 0.5
        
        # Si quieres probar a girar la base sobre su eje (Yaw), descomenta esta:
        #acciones[:, 2] = 0.5
        
        # Si quieres mover la primera articulación del brazo (Hombro Pan), descomenta esta:
        pos += 0.02
        # acciones[:, 3] += pos
        # acciones[:, 4] += -pos
        # acciones[:, 5] += -pos
        # acciones[:, 6] += -pos
        # acciones[:, 7] += -pos
        #acciones[:, 8] += -pos
        #print("Accion: ", i)
        # Enviamos la acción al simulador
        obs, rewards, terminated, truncated, info = env.step(acciones)

        dones = terminated | truncated
        if dones.any():
            print("[INFO] El entorno detectó un choque y se ha reiniciado. Limpiando acciones...")
            pos = 0.0
            ids_robot = torch.tensor([0], device=env.device)
            tray_cfg = SceneEntityCfg("robot", body_names="rg6_tcp_link")
            
            # 2. Resolver los IDs contra la escena actual para evitar el error 'slice'
            tray_cfg.resolve(env.unwrapped.scene)
            
            # 3. Llamar a la función con el asset_cfg configurado
            gravedad_cuerpo= projected_gravity(env.unwrapped)
            print("gravedad_cuerpo", gravedad_cuerpo)
        
    print("[INFO] Simulación terminada.")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()