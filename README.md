# Kairos Lab

## Transformacion urdf.xacro --> usd

1. En rbkairos_plus.urdf.xacro modificar el argumento `ur_type`  y poner "ur5e".
2. En rbkairos_base.urdf.xacro añadir los siguientes joints despues de importar las ruedas para evitar simular las fisicas complejas de los rodillos de las ruedas. El movimiento en el simulador se realizará a traves de 2 articulaciones prismaticas (x, y) virtuales y una continua para la rotación en el eje z. 
    
    ``` xml
    <link name="world" />

    <joint name="world_to_virtual_base" type="fixed">
      <parent link="world" />
      <child link="${prefix}virtual_base_link" />
    </joint>

    <link name="${prefix}virtual_base_link" />

    <link name="${prefix}virtual_link_x" />
    <joint name="${prefix}virtual_joint_x" type="prismatic">
      <parent link="${prefix}virtual_base_link" />
      <child link="${prefix}virtual_link_x" />
      <axis xyz="1 0 0" />
      <limit lower="-100" upper="100" effort="1000" velocity="10" />
    </joint>

    <link name="${prefix}virtual_link_y" />
    <joint name="${prefix}virtual_joint_y" type="prismatic">
      <parent link="${prefix}virtual_link_x" />
      <child link="${prefix}virtual_link_y" />
      <axis xyz="0 1 0" />
      <limit lower="-100" upper="100" effort="1000" velocity="10" />
    </joint>

    <joint name="${prefix}virtual_joint_yaw" type="continuous">
      <parent link="${prefix}virtual_link_y" />
      <child link="${prefix}base_footprint" />
      <axis xyz="0 0 1" />
    </joint>
    ``` 
3. En el archivo rbkairos_mecanum_wheel.urdf.xacro se cambia el tipo de joint a "fixed" y se elimina limits y axis.
4. Desde el paqute de ROS convertir a .urdf:

``` bash
    xacro rbkairos_plus.urdf.xacro > rbkairos_plus.urdf
```
>Nota: Es necesario borrar los archivos de compilación (build/robotnik_description, install/robotnik_description) y compilar el paquete de ROS (colcon build)

>Nota: Si es necesario ajustar rutas para encontrar los paquetes ur_description y robotnik_sensors.

5. Transformar a usd con el script de conversión de IsaacLab:

``` bash
IsaacLab/scripts/tools/convert_urdf.py /ruta/a/tu/nuevo/rbkairos_plus.urdf /ruta/destino/rbkairos_plus.usd --fix-base
```

## 