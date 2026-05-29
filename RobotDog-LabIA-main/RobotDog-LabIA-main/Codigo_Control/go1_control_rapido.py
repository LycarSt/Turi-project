#!/usr/bin/python3
import sys
import time
from pynput import keyboard  # Captura teclas sin root

sys.path.append('/home/labia-001/ROBOTDOG/unitree_legged_sdk/lib/python/amd64')
import robot_interface as sdk


def on_press(key):
    global cmd
    try:
        if key == keyboard.Key.up:
            cmd.velocity = [0.2, 0.0]
            cmd.yawSpeed = 0.0
        elif key == keyboard.Key.down:
            cmd.velocity = [-0.2, 0.0]
            cmd.yawSpeed = 0.0
        elif key == keyboard.Key.left:
            cmd.velocity = [0.0, 0.0]
            cmd.yawSpeed = 0.5
        elif key == keyboard.Key.right:
            cmd.velocity = [0.0, 0.0]
            cmd.yawSpeed = -0.5
        elif key == keyboard.Key.space:
            cmd.velocity = [0.0, 0.0]
            cmd.yawSpeed = 0.0
        elif key.char == 'q':
            print("\n🛑 Saliendo...")
            return False
    except AttributeError:
        pass


if __name__ == '__main__':
    HIGHLEVEL = 0xee
    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)
    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)

    cmd.mode = 2
    cmd.footRaiseHeight = 0.1
    cmd.bodyHeight = 0.0

    print("🐕 Control del Go1 con flechas del teclado")
    print("⬆️ Adelante | ⬇️ Atrás | ⬅️ Izquierda | ➡️ Derecha | Espacio: parar | Q: salir")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while listener.is_alive():
            udp.SetSend(cmd)
            udp.Send()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass

    cmd.mode = 1
    cmd.velocity = [0.0, 0.0]
    cmd.yawSpeed = 0.0
    udp.SetSend(cmd)
    udp.Send()
    print("✅ Robot detenido.")
