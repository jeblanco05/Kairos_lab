import argparse
from isaaclab.app import AppLauncher

# 1. Inicializar Omniverse en modo headless
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

def main():
    # 2. Rutas
    source_usd = "/mnt/beegfs/home/jesuseliseo.blanco/my_projects/Kairos_lab/assets/rbkairos_plus_rg6/rbkairos_plus_rg6.usd"
    target_usd = "/mnt/beegfs/home/jesuseliseo.blanco/my_projects/Kairos_lab/assets/rbkairos_plus_rg6/rbkairos_plus_rg6.usd"
    
    stage = Usd.Stage.Open(source_usd)
    if not stage:
        print(f"Error: No se pudo abrir {source_usd}")
        return
        
    print("USD cargado. Transformando cilindros en esferas de baja fricción...")
    
    # 3. Crear el material de fricción cero
    material_path = "/World/ZeroFrictionMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(0.0)
    physics_material.CreateDynamicFrictionAttr().Set(0.0)
    physics_material.CreateRestitutionAttr().Set(0.0)
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr().Set("multiply")
    
    modificaciones = 0
    radio_rueda = 0.127  # El radio de la rueda del Kairos+
    ruedas_esperadas = 4  # Kairos+ tiene 4 ruedas mecanum -> ajusta si tu USD difiere

    def ancestro_es_rueda(prim):
        p = prim
        while p and p.IsValid() and not p.IsPseudoRoot():
            if "wheel" in p.GetName().lower():
                return True
            p = p.GetParent()
        return False

    # 4. Recorrer el USD buscando las ruedas
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not ("wheel" in prim.GetPath().pathString.lower() or ancestro_es_rueda(prim)):
            continue

        # PASO A: Desactivar la colisión de la malla original (ahora es solo visual)
        prim.RemoveAPI(UsdPhysics.CollisionAPI)

        # PASO B: Crear una nueva esfera colgando del mismo eslabón
        parent_path = prim.GetParent().GetPath()
        sphere_path = parent_path.AppendChild("sphere_collision")

        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.GetRadiusAttr().Set(radio_rueda)

        # PASO C: Ocultar la esfera visualmente sin excluirla de la física.
        # 'invisible' es más fiable que 'guide' para geometría puramente física
        # en Isaac Sim (algunos pipelines de colisión filtran por purpose).
        sphere.GetPurposeAttr().Set(UsdGeom.Tokens.invisible)

        # PASO D: Convertir la esfera en un objeto físico (CollisionAPI)
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())

        # PASO E: Pegarle el material sin fricción a la esfera
        binding_api = UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim())
        binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

        print(f"-> Convertido: {parent_path.name} usa ahora una colisión esférica.")
        modificaciones += 1

    # =======================================================================
    # FIX DE COLISIONES: Ignorar choque entre la muñeca y la pinza
    print("\nBuscando dinámicamente la muñeca y la pinza...")
    
    wrist_prim = None
    gripper_prim = None

    # 1. Recorremos todo el archivo USD buscando las piezas por su nombre exacto
    for prim in stage.Traverse():
        prim_name = prim.GetName()
        if prim_name == "arm_tool0":
            wrist_prim = prim
        elif prim_name == "rg6_base_link":
            gripper_prim = prim

    # 2. Si las hemos encontrado, aplicamos el filtro
    if wrist_prim and gripper_prim:
        print(f"-> Muñeca encontrada en: {wrist_prim.GetPath()}")
        print(f"-> Pinza encontrada en: {gripper_prim.GetPath()}")
        
        # Aplicar la API a la muñeca
        if not wrist_prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(wrist_prim)
        else:
            filtered_pairs_api = UsdPhysics.FilteredPairsAPI(wrist_prim)
        
        # Añadir la pinza como objetivo a ignorar
        filtered_pairs_api.GetFilteredPairsRel().AddTarget(gripper_prim.GetPath())
        
        print("-> ¡Éxito! Colisión fantasma desactivada.")
    else:
        print("-> [ADVERTENCIA] Faltan piezas:")
        if not wrist_prim: print("   - No se encontró 'arm_tool0'")
        if not gripper_prim: print("   - No se encontró 'rg6_base_link'")


    print("\nDesactivando colisiones de la bandeja (y sus sub-mallas) para ahorrar memoria GPU...")
    tray_prim_name = "tray_link" # ¡Asegúrate de poner el nombre correcto aquí!
    
    tray_prim = None
    for prim in stage.Traverse():
        if prim.GetName() == tray_prim_name:
            tray_prim = prim
            break

    # =======================================================================
    # FIX DE COLISIONES: Apagado TOTAL de físicas en la pinza
    # =======================================================================
    print("\n[MODO SEGURO] Buscando y apagando todas las colisiones de la pinza...")
    
    # Nombres clave que identifican las partes de la pinza
    piezas_pinza = [
        "rg6_base_link",
        "rg6_left_finger_link",
        "rg6_right_finger_link",
        "rg6_tcp_link",
        "arm_ft_frame"
    ]
    
    colisiones_apagadas = 0
    
    # Recorremos ABSOLUTAMENTE TODOS los elementos del archivo USD
    for prim in stage.Traverse():
        ruta_str = str(prim.GetPath())
        
        # Si la ruta del prim contiene el nombre de alguna de las piezas de la pinza...
        # (Esto pilla tanto al 'rg6_base_link' como a 'rg6_base_link/collisions/mesh_0')
        if any(pieza in ruta_str for pieza in piezas_pinza):
            
            # Si ese prim tiene la capacidad de chocar, se la quitamos
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI(prim)
                collision_api.GetCollisionEnabledAttr().Set(False)
                colisiones_apagadas += 1

    if colisiones_apagadas > 0:
        print(f"-> ¡Victoria! Se han desactivado {colisiones_apagadas} mallas de colisión ocultas en la pinza.")
        print("-> El robot ya no explotará internamente, pero conservará su peso e inercia.")
    else:
        print("-> [ADVERTENCIA] No se encontró ninguna colisión activa en la pinza.")
    # =======================================================================
        
    if tray_prim:
        print(f"-> Bandeja encontrada en: {tray_prim.GetPath()}")
        colisiones_desactivadas = 0
       
        for child_prim in Usd.PrimRange(tray_prim):
            if child_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI(child_prim)
                collision_api.GetCollisionEnabledAttr().Set(False)
                colisiones_desactivadas += 1
                print(f"   - Colisión desactivada en el sub-nodo: {child_prim.GetName()}")
                
        if colisiones_desactivadas > 0:
            print(f"-> ¡Éxito! Se desactivaron {colisiones_desactivadas} mallas de colisión ocultas dentro de la bandeja.")
            print("-> (La inercia y masa de la bandeja seguirán afectando al UR5e)")
        else:
            print("-> [ADVERTENCIA] No se encontró NINGUNA colisión dentro de la bandeja.")
    else:
         print(f"-> [ADVERTENCIA] No se encontró la bandeja raíz con el nombre '{tray_prim_name}'")

    # =======================================================================

    # 5. Guardar el nuevo archivo, avisando si el número de ruedas no cuadra
    if modificaciones == 0:
        print("\nAdvertencia: No se encontraron colisiones de ruedas. No se ha guardado nada.")
    else:
        if modificaciones != ruedas_esperadas:
            print(
                f"\n¡ATENCIÓN! Se esperaban {ruedas_esperadas} ruedas y se convirtieron "
                f"{modificaciones}. Si tu Kairos+ realmente tiene {ruedas_esperadas} ruedas, "
                "esto indica que alguna colisión de rueda NO se ha detectado (revisa nombres "
                "de prim / jerarquía) y seguirá teniendo fricción real -> movimiento asimétrico."
            )
        stage.GetRootLayer().Export(target_usd)
        print(f"\n¡Éxito! Robot guardado en: {target_usd} ({modificaciones} ruedas convertidas)")

if __name__ == "__main__":
    main()
    simulation_app.close()