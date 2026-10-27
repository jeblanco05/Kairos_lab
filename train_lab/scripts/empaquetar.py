import argparse
from isaaclab.app import AppLauncher

# 1. Inicializar Omniverse en modo headless
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import  UsdUtils
import sys

# Ignoramos argumentos extra que pueda meter el lanzador
args = [arg for arg in sys.argv if not arg.startswith('--')]

if len(args) < 3:
    print("Uso: script.py <origen.usd> <destino.usdz>")
    simulation_app.close()
    sys.exit(1)

origen = args[1]
destino = args[2]

print(f"Analizando y empaquetando dependencias de: {origen}")
success = UsdUtils.CreateNewUsdzPackage(origen, destino)

if success:
    print(f"¡Éxito! Archivo portátil creado en: {destino}")
else:
    print("Error al empaquetar el archivo. Comprueba que el origen existe.")

simulation_app.close()