import sys
import time
import socket

# Asegúrate de agregar la ruta correcta de tu SDK de Unitree
sys.path.append('/home/labia-001/ROBOTDOG/unitree_legged_sdk/lib/python/amd64')
import robot_interface as sdk

class Custom:
    def __init__(self, level):
        # Configuración inicial de comunicación con el robot
        self.udp = sdk.UDP(level, 8090, "192.168.123.161", 8082)  # Dirección IP y puertos
        self.cmd = sdk.HighCmd()
        self.state = sdk.HighState()
        self.motiontime = 0
        self.dt = 0.002  # 0.001~0.01
        self.command = None
        self.udp.InitCmdData(self.cmd)

    def UDPRecv(self):
        """Recibe los datos del estado del robot"""
        self.udp.Recv()

    def UDPSend(self):
        """Envía los datos de comando al robot"""
        self.udp.Send()

    def RobotControl(self):
        """Controla el movimiento del robot según el comando leído del archivo command.txt"""
        self.motiontime += 1
        self.udp.GetRecv(self.state)
        print(f"{self.motiontime}   {self.state.imu.quaternion[2]}")

        # Leer comando desde el archivo command.txt
        try:
            with open('/home/labia-001/ROBOTDOG/unitree_legged_sdk/example_py/command.txt', 'r') as file:
                self.command = file.read().strip()  # Lee el comando y elimina espacios extra
        except FileNotFoundError:
            print("command.txt no encontrado.")
            self.command = 'n'  # Si no se encuentra el archivo, el comando por defecto es 'n'

        print(f"Comando recibido: {self.command}")

        # Configuración del robot según el comando
        self.cmd.mode = 0  # 0: idle, default stand, 1: forced stand, 2: walk continuously
        self.cmd.gaitType = 0
        self.cmd.speedLevel = 0
        self.cmd.footRaiseHeight = 0
        self.cmd.bodyHeight = 0
        self.cmd.euler = [0, 0, 0]
        self.cmd.velocity = [0.0, 0.0]
        self.cmd.yawSpeed = 0.0
        self.cmd.reserve = 0

        if self.command == 'n':  # 'n' for stand
            print("Stand")
            self.cmd.mode = 1
            self.cmd.bodyHeight = 0.0
        elif self.command == 'd':  # 'b' for sit
            print("Sit")
            self.cmd.mode = 1
            self.cmd.bodyHeight = -0.5
        elif self.command == 'u':  # 'u' for go forward
            print("Go forward")
            self.cmd.mode = 2  # Walk forward
            self.cmd.velocity = [0.2, 0]
            self.cmd.yawSpeed = 0
            self.cmd.footRaiseHeight = 0.1
        elif self.command == 'l':  # 'l' for turn left
            print("Turn left")
            self.cmd.mode = 2
            self.cmd.gaitType = 1
            self.cmd.velocity[0] = 0.2
            self.cmd.yawSpeed = 1
            self.cmd.footRaiseHeight = 0.1
        elif self.command == 'r':  # 'r' for turn right
            print("Turn right")
            self.cmd.mode = 2
            self.cmd.gaitType = 1
            self.cmd.velocity[0] = 0.2
            self.cmd.yawSpeed = -1
            self.cmd.footRaiseHeight = 0.1
        else:
            print("Comando inválido, se mantendrá en modo Stand.")
            self.cmd.mode = 1
            self.cmd.bodyHeight = 0.0

        # Enviar comando al robot
        self.udp.SetSend(self.cmd)

def main():
    print("Communication level is set to HIGH-level.")
    print("WARNING: Make sure the robot is standing on the ground.")
    input("Press Enter to continue...")

    custom = Custom(level=0xee)  # HIGHLEVEL
    while True:
        custom.RobotControl()
        custom.UDPSend()
        custom.UDPRecv()
        time.sleep(0.002)  # Delay entre comandos

if __name__ == "__main__":
    main()
