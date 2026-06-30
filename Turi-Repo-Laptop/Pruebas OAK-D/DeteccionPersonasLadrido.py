import sys
import time
import simpleaudio as sa

# Ruta del SDK de Unitree
sys.path.append('/home/labia-001/ROBOTDOG/unitree_legged_sdk/lib/python/amd64')
import robot_interface as sdk

# Ruta del txt generado por la OAK-D
DETECCION_FILE = "/home/labia-001/Repo/turi-project/Pruebas OAK-D/detecciones_oakd.txt"

# Ruta del ladrido WAV
LADRIDO_WAV = "/home/labia-001/Repo/turi-project/Pruebas OAK-D/ladrido.wav"

# Cooldown entre ladridos
COOLDOWN_LADRIDO = 8


class Custom:
    def __init__(self, level):
        # Comunicación UDP con el robot
        self.udp = sdk.UDP(level, 8090, "192.168.123.161", 8082)
        self.cmd = sdk.HighCmd()
        self.state = sdk.HighState()

        # Estado inicial
        self.motiontime = 0
        self.pitch_actual = 0.0
        self.pitch_objetivo = 0.0

        # Último ladrido
        self.ultimo_ladrido = 0

        # Pre-cargar audio (más rápido)
        self.ladrido_audio = sa.WaveObject.from_wave_file(LADRIDO_WAV)

        # Inicializa el paquete de comandos
        self.udp.InitCmdData(self.cmd)

    def leer_detecciones(self):
        """Verifica si en el TXT se detectó una persona"""
        try:
            with open(DETECCION_FILE, "r") as f:
                contenido = f.read().lower()
                return "person" in contenido
        except:
            return False

    def reproducir_ladrido(self):
        """Reproduce sonido con cooldown"""
        ahora = time.time()
        if ahora - self.ultimo_ladrido >= COOLDOWN_LADRIDO:
            print("🔊 Ladrido!")
            self.ultimo_ladrido = ahora
            self.ladrido_audio.play()  # No bloquea

    def suavizar(self, actual, objetivo, factor=0.02):
        return actual + (objetivo - actual) * factor

    def preparar_comando_base(self):
        # Postura base sin movimiento
        self.cmd.mode = 1
        self.cmd.gaitType = 0
        self.cmd.speedLevel = 0
        self.cmd.velocity = [0.0, 0.0]
        self.cmd.yawSpeed = 0.0
        self.cmd.footRaiseHeight = 0.0
        self.cmd.bodyHeight = 0.0

    def RobotControl(self):
        self.motiontime += 1

        # Recibir estado
        self.udp.Recv()
        self.udp.GetRecv(self.state)

        self.preparar_comando_base()

        # Leer detección
        persona_detectada = self.leer_detecciones()

        if persona_detectada:
            print("Persona detectada → Levantar cabeza")
            self.pitch_objetivo = -0.30
            self.reproducir_ladrido()
        else:
            print("No hay persona → posición normal")
            self.pitch_objetivo = 0.0

        # Movimiento suave
        self.pitch_actual = self.suavizar(self.pitch_actual, self.pitch_objetivo)

        # Aplicar pitch
        self.cmd.euler = [0.0, self.pitch_actual, 0.0]

        # Enviar comando al robot
        self.udp.SetSend(self.cmd)
        self.udp.Send()


def main():
    print("HIGH-LEVEL control iniciado.")
    input("Presiona Enter cuando el robot esté en el suelo y listo...")

    custom = Custom(level=0xee)

    while True:
        custom.RobotControl()
        time.sleep(0.002)

if __name__ == "__main__":
    main()