#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from pynput import keyboard

msg = """
---------------------------------------------
Controlador de Robot Personalizado (Multi-tecla)
---------------------------------------------
Movimiento de la Base (¡Puedes combinar teclas!):
   w : Adelante
   s : Atrás
   a : Derecha (traslación lateral)
   d : Izquierda (traslación lateral)
   q : Girar a la izquierda
   e : Girar a la derecha
   Espacio : Detener la base

Ajuste de Velocidad de la Base:
   r : Aumentar velocidad (+0.5)
   f : Disminuir velocidad (-0.5)

Control de Offset del Brazo (Limites: X[±0.3], Y[±0.2], Z[±0.2]):
   Flecha Arriba    : +X offset (+0.05)
   Flecha Abajo     : -X offset (-0.05)
   Flecha Izquierda : +Y offset (+0.05)
   Flecha Derecha   : -Y offset (-0.05)
   u                : +Z offset (+0.05)
   j                : -Z offset (-0.05)

CTRL-C o ESC para salir
---------------------------------------------
"""

class CustomTeleopNode(Node):
    def __init__(self):
        super().__init__('custom_teleop_node')
        
        # Publicadores
        self.pub_cmd_vel = self.create_publisher(Twist, '/robot/nav2_cmd_vel', 10)
        self.pub_offset = self.create_publisher(Vector3, '/kairos_deploy/tray_offset_target', 10)

        # Variables de estado y velocidad
        self.speed = 0.5
        
        # Variables de Offset
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0
        self.offset_step = 0.05

        # Conjunto para guardar las teclas que están siendo presionadas simultáneamente
        self.pressed_keys = set()

        print(msg)
        print(f"Velocidad inicial: {self.speed} | Offset inicial: ({self.offset_x:.2f}, {self.offset_y:.2f}, {self.offset_z:.2f})")

        # Configurar el listener del teclado en segundo plano
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

        # Timer para publicar continuamente (10 Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def get_char(self, key):
        """Intenta extraer el caracter de la tecla presionada."""
        try:
            return key.char.lower() if key.char else None
        except AttributeError:
            return None

    def clamp(self, value, min_val, max_val):
        """Mantiene un valor dentro de un rango especificado."""
        return max(min_val, min(value, max_val))

    def on_press(self, key):
        # Para salir con ESC
        if key == keyboard.Key.esc:
            return False 

        # Evitar procesar repetidamente teclas que ya están presionadas
        if key not in self.pressed_keys:
            char = self.get_char(key)
            
            # --- CONTROL DE VELOCIDAD Y PARADA (Solo una vez al presionar) ---
            if char == 'r':
                self.speed += 0.5
                print(f"Modificador de velocidad subido: {self.speed:.1f}")
            elif char == 'f':
                self.speed = max(0.0, self.speed - 0.5)
                print(f"Modificador de velocidad bajado: {self.speed:.1f}")
            elif char == ' ':
                self.pressed_keys.clear() # Limpiar todo movimiento base

            # --- CONTROL DEL OFFSET DEL BRAZO CON LIMITES ---
            elif key == keyboard.Key.up:
                self.offset_x = self.clamp(self.offset_x + self.offset_step, -0.3, 0.3)
                self.print_offsets()
            elif key == keyboard.Key.down:
                self.offset_x = self.clamp(self.offset_x - self.offset_step, -0.3, 0.3)
                self.print_offsets()
            elif key == keyboard.Key.left:
                self.offset_y = self.clamp(self.offset_y + self.offset_step, -0.2, 0.2)
                self.print_offsets()
            elif key == keyboard.Key.right:
                self.offset_y = self.clamp(self.offset_y - self.offset_step, -0.2, 0.2)
                self.print_offsets()
            elif char == 'u':
                self.offset_z = self.clamp(self.offset_z + self.offset_step, -0.2, 0.2)
                self.print_offsets()
            elif char == 'j':
                self.offset_z = self.clamp(self.offset_z - self.offset_step, -0.2, 0.2)
                self.print_offsets()

        # Añadir la tecla al conjunto de teclas activas
        self.pressed_keys.add(key)
        
    def print_offsets(self):
        print(f"Offset actualizado: X={self.offset_x:.2f}, Y={self.offset_y:.2f}, Z={self.offset_z:.2f}")

    def on_release(self, key):
        # Quitar la tecla del conjunto cuando se suelta
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)

    def timer_callback(self):
        # Leer todas las teclas activas en este instante para calcular el movimiento combinado
        x, y, th = 0.0, 0.0, 0.0
        
        active_chars = [self.get_char(k) for k in self.pressed_keys]

        # Sumar/restar movimientos según las teclas activadas
        if 'w' in active_chars: x += 1.0
        if 's' in active_chars: x -= 1.0
        if 'a' in active_chars: y -= 1.0  # -y para derecha en tu configuración
        if 'd' in active_chars: y += 1.0  # +y para izquierda
        if 'q' in active_chars: th += 1.0
        if 'e' in active_chars: th -= 1.0

        # Calcular velocidades teóricas
        calc_linear_x = x * self.speed
        calc_linear_y = y * self.speed
        calc_angular_z = th * self.speed

        # Aplicar límites de seguridad a las velocidades finales
        twist = Twist()
        twist.linear.x = self.clamp(calc_linear_x, -1.5, 1.5)
        twist.linear.y = self.clamp(calc_linear_y, -1.5, 1.5)
        twist.angular.z = self.clamp(calc_angular_z, -1.0, 1.0)
        
        self.pub_cmd_vel.publish(twist)

        # Publicar el offset del brazo (ya limitado en on_press)
        offset_msg = Vector3()
        offset_msg.x = self.offset_x
        offset_msg.y = self.offset_y
        offset_msg.z = self.offset_z
        self.pub_offset.publish(offset_msg)

    def stop_robot(self):
        twist = Twist()
        self.pub_cmd_vel.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    teleop_node = CustomTeleopNode()

    try:
        # Mantener el nodo corriendo
        rclpy.spin(teleop_node)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nDeteniendo el robot...")
        teleop_node.stop_robot()
        teleop_node.listener.stop()
        teleop_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()