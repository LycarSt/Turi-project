import sys
import time

# Ruta del SDK de Unitree
sys.path.append('/home/labia-001/ROBOTDOG/unitree_legged_sdk/lib/python/amd64')
import robot_interface as sdk

# Ruta del txt generado por la OAK-D
DETECCION_FILE = "/home/labia-001/Repo/turi-project/Pruebas OAK-D/detecciones_oakd.txt"

class Custom:
    def __init__(self, level):
        # Comunicación UDP con el robot
        self.udp = sdk.UDP(level, 8090, "192.168.123.161", 8082)
        self.cmd = sdk.HighCmd()
        self.state = sdk.HighState()

        # Inicializar datos
        self.motiontime = 0
        self.pitch_actual = 0.0
        self.pitch_objetivo = 0.0

        # Inicializa el paquete de comandos
        self.udp.InitCmdData(self.cmd)

    def leer_detecciones(self):
        """Lee si la OAK-D detecta una persona en el TXT"""
        try:
            with open(DETECCION_FILE, "r") as f:
                contenido = f.read().lower()
                return "person" in contenido
        except:
            return False

    def suavizar(self, actual, objetivo, factor=0.02):
        """Evita vibraciones – Movimiento suave tipo joystick"""
        return actual + (objetivo - actual) * factor

    def preparar_comando_base(self):
        """Configura la postura base del robot sin movimiento"""
        self.cmd.mode = 1                # posición estable (stand)
        self.cmd.gaitType = 0
        self.cmd.speedLevel = 0
        self.cmd.velocity = [0.0, 0.0]   # sin desplazamiento
        self.cmd.yawSpeed = 0.0
        self.cmd.footRaiseHeight = 0.0
        self.cmd.bodyHeight = 0.0

    def RobotControl(self):
        """Lee detección + aplica inclinación del tronco"""
        self.motiontime += 1

        # Actualizar estado recibido del robot
        self.udp.Recv()
        self.udp.GetRecv(self.state)

        # Configuración base del perro quieto
        self.preparar_comando_base()

        # Verificar detecciones del TXT
        persona_detectada = self.leer_detecciones()

        if persona_detectada:
            print("Persona detectada → Levantar cabeza")
            self.pitch_objetivo = -0.30   # levantar la cabeza (pitch +0.3 rad)
        else:
            print("No hay persona → posición normal")
            self.pitch_objetivo = 0.0

        # Movimiento suave
        self.pitch_actual = self.suavizar(self.pitch_actual, self.pitch_objetivo)

        # Aplicar orientación del cuerpo
        self.cmd.euler = [0.0, self.pitch_actual, 0.0]

        # Enviar comandos al robot
        self.udp.SetSend(self.cmd)
        self.udp.Send()  # Enviar paquete actualizado

def main():
    print("HIGH-LEVEL control iniciado.")
    input("Presiona Enter cuando el robot esté en el suelo y listo...")

    custom = Custom(level=0xee)  # HIGHLEVEL mode

    while True:
        custom.RobotControl()
        time.sleep(0.002)

if __name__ == "__main__":
    main()