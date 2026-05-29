#!/usr/bin/python3
import sys
import time
import math

sys.path.append('/home/labia-001/ROBOTDOG/unitree_legged_sdk/lib/python/amd64')
import robot_interface as sdk

if __name__ == '__main__':
    HIGHLEVEL = 0xee

    # Configurar conexión UDP
    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)
    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)

    print("🐕 Iniciando giro lento hacia la izquierda (Ctrl+C para detener)")
    cmd.mode = 2  # modo caminar
    cmd.velocity = [0.0, 0.0]  # sin movimiento lineal
    cmd.footRaiseHeight = 0.05  # altura pequeña de zancada
    cmd.bodyHeight = 0.0

    # Giro lento: yawSpeed controla la velocidad angular (radianes/segundo aprox.)
    cmd.yawSpeed = 0.1  # izquierda = positivo, derecha = negativo

    try:
        while True:
            udp.SetSend(cmd)
            udp.Send()
            time.sleep(0.02)  # 50 Hz
    except KeyboardInterrupt:
        print("\n🛑 Detenido por el usuario, parando robot...")
        cmd.mode = 1  # modo quieto
        cmd.velocity = [0.0, 0.0]
        cmd.yawSpeed = 0.0
        udp.SetSend(cmd)
        udp.Send()
        time.sleep(0.1)
