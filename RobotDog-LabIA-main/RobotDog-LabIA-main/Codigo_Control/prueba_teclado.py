#!/usr/bin/python

import sys
import time
import math

sys.path.append('/home/labia-001/ROBOTDOG/unitree_legged_sdk/lib/python/amd64')
import robot_interface as sdk


if __name__ == '__main__':

    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff

    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)

    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)

    motiontime = 0
    # Incluir control manual por teclado
    while True:
        key = input("Enter command (n=stand, b=sit, u=forward, l=left, r=right): ")

        if key == 'n':
            cmd.mode = 1  # Stand
            cmd.bodyHeight = 0.0
        elif key == 'b':
            cmd.mode = 1  # Sit
            cmd.bodyHeight = -0.5
        elif key == 'u':
            cmd.mode = 2  # Walk forward
            cmd.velocity = [0.2, 0]
            cmd.yawSpeed = 0
            cmd.footRaiseHeight = 0.1
        elif key == 'l':
            cmd.mode = 2  # Turn left
            cmd.velocity = [0.2, 0]
            cmd.yawSpeed = 1
            cmd.footRaiseHeight = 0.1
        elif key == 'r':
            cmd.mode = 2  # Turn right
            cmd.velocity = [0.2, 0]
            cmd.yawSpeed = -1
            cmd.footRaiseHeight = 0.1
        else:
            print("Invalid command!")

        udp.SetSend(cmd)
        udp.Send()
        time.sleep(0.002)  # Delay between commands

