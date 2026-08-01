import argparse
from isaaclab.app import AppLauncher

# 1. Inicialización
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext # Importante para arrancar físicas
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from pxr import UsdPhysics
from pxr import Usd, UsdPhysics
import omni

def calculate_and_print_robot_mass(prim_path: str):
    # Get the current active stage
    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(prim_path)
    
    if not robot_prim.IsValid():
        return
        
    total_mass = 0.0
    
    print(f"\n--- Mass Breakdown for {prim_path} ---")
    print(f"{'Prim Name':<45} | {'Mass (kg)'}")
    print("-" * 60)
    
    # Traverse all descendant prims
    for prim in Usd.PrimRange(robot_prim):
        # Check if the prim has physical mass properties defined
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            mass_value = mass_api.GetMassAttr().Get()
            
            # Add to total if explicit mass is found
            if mass_value is not None:
                print(f"{prim.GetName():<45} | {mass_value:.3f}")
                total_mass += mass_value
                
    print("-" * 60)
    print(f"{'Total Simulated Mass':<45} | {total_mass:.3f} kg\n")

def main():
    # Inicializar el contexto de simulación
    sim = SimulationContext()

    # 2. Spawn
    ruta_usd = "/mnt/beegfs/home/jesuseliseo.blanco/my_projects/Kairos_lab/assets/rbkairos_plus_rg6_tray/rbkairos_plus_rg6_tray.usd"
    spawn_cfg = sim_utils.UsdFileCfg(usd_path=ruta_usd)
    spawn_cfg.func("/World/Kairos", spawn_cfg)

    # 3. Buscar el nodo físico real (Ignorando la carpeta de materiales)
    stage = omni.usd.get_context().get_stage()
    kairos_root = stage.GetPrimAtPath("/World/Kairos")
    
    robot_base = None
    print("\n--- Analizando estructura ---")
    for child in kairos_root.GetChildren():
        print(f"Encontrado: {child.GetPath()}")
        # Ignoramos la carpeta de texturas
        if child.GetName() != "Looks":
            robot_base = child
            
    if not robot_base:
        print("¡ERROR! No se encontró el cuerpo del robot.")
        return
    
    # Aplicar la API si no la tiene
    if not robot_base.HasAPI(UsdPhysics.ArticulationRootAPI):
        print(f"Aplicando ArticulationRootAPI a: {robot_base.GetPath()}")
        robot_base.ApplyAPI(UsdPhysics.ArticulationRootAPI)
    
    # 4. Configurar la Articulation
    robot_cfg = ArticulationCfg(
        prim_path=str(robot_base.GetPath()), 
        spawn=None,
        actuators={
            "motores": ImplicitActuatorCfg(
                joint_names_expr=[".*"], 
                stiffness=0.0, 
                damping=10.0
            )
        }
    )
    calculate_and_print_robot_mass("/World/Kairos")
    kairos = Articulation(robot_cfg)
    
    # Arrancar las físicas para que se genere el _root_physx_view
    sim.reset()
    
    # 5. Verificación
    print(f"\n[ÉXITO] Articulación física cargada en: {robot_base.GetPath()}")
    print(f"Número de articulaciones (DOFs): {kairos.num_joints}")
    print("Nombres:")
    for i, nombre in enumerate(kairos.joint_names):
        print(f"  [{i}]: {nombre}")

if __name__ == "__main__":
    main()
    simulation_app.close()