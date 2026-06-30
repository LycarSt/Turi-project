#!/usr/bin/env python3
import sys
import time
import threading
import cv2
import depthai as dai
import numpy as np
import simpleaudio as sa
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS (Actualízalas si es necesario) ---
# Ruta SDK Unitree
sys.path.append('/home/labia-001/ROBOTDOG/unitree_legged_sdk/lib/python/amd64')
import robot_interface as sdk

# Ruta Modelo YOLO
NN_PATH = str((Path(__file__).parent / Path('/home/labia-001/Repo/turi-project/Pruebas OAK-D/yolov8n_coco_640x352.blob')).resolve().absolute())
# Ruta Audio
LADRIDO_WAV = "/home/labia-001/Repo/turi-project/Pruebas OAK-D/ladrido.wav"

# --- VARIABLES COMPARTIDAS (Comunicación entre Hilos) ---
# Esta variable reemplaza al archivo .txt
# False = No hay persona, True = Hay persona
GLOBAL_PERSONA_DETECTADA = False
LOCK = threading.Lock() # Para evitar conflictos de lectura/escritura

# --- HILO 1: CÁMARA OAK-D (Visión) ---
def hilo_camara():
    global GLOBAL_PERSONA_DETECTADA
    
    # 1. Configuración OAK-D
    labelMap = [ "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]

    pipeline = dai.Pipeline()
    camRgb = pipeline.create(dai.node.ColorCamera)
    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    nnOut = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    nnOut.setStreamName("nn")

    camRgb.setPreviewSize(640, 352)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(40)

    detectionNetwork.setConfidenceThreshold(0.5)
    detectionNetwork.setNumClasses(80)
    detectionNetwork.setCoordinateSize(4)
    detectionNetwork.setIouThreshold(0.5)
    detectionNetwork.setBlobPath(NN_PATH)
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)

    camRgb.preview.link(detectionNetwork.input)
    detectionNetwork.passthrough.link(xoutRgb.input)
    detectionNetwork.out.link(nnOut.input)

    # 2. Bucle de Visión
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        while True:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

            if inDet is not None:
                detections = inDet.detections
                # Verificamos si hay alguna "person" (clase person suele ser index 0 en COCO, o string "person")
                hay_persona = False
                for detection in detections:
                    if labelMap[detection.label] == "person":
                        hay_persona = True
                        break
                
                # Actualizamos la variable compartida de forma segura
                with LOCK:
                    GLOBAL_PERSONA_DETECTADA = hay_persona

            if inRgb is not None:
                # Opcional: Mostrar imagen (puede ralentizar si no hay monitor)
                cv2.imshow("Vista Robot", inRgb.getCvFrame())

            if cv2.waitKey(1) == ord('q'):
                break
        
        # Al cerrar ventana, matamos todo el script
        sys.exit()

# --- HILO 2: CONTROL ROBOT (Movimiento y Audio) ---
class RobotController:
    def __init__(self):
        self.udp = sdk.UDP(0xee, 8090, "192.168.123.161", 8082)
        self.cmd = sdk.HighCmd()
        self.state = sdk.HighState()
        self.udp.InitCmdData(self.cmd)

        self.pitch_actual = 0.0
        self.pitch_objetivo = 0.0
        
        # Audio
        self.ultimo_ladrido = 0
        self.COOLDOWN_LADRIDO = 8
        self.ladrido_audio = sa.WaveObject.from_wave_file(LADRIDO_WAV)

    def suavizar(self, actual, objetivo, factor=0.02):
        return actual + (objetivo - actual) * factor

    def run(self):
        print("Iniciando control del robot...")
        while True:
            self.udp.Recv()
            self.udp.GetRecv(self.state)

            # Configuración base (High Level)
            self.cmd.mode = 1
            self.cmd.gaitType = 0
            self.cmd.speedLevel = 0
            self.cmd.velocity = [0.0, 0.0]
            self.cmd.yawSpeed = 0.0
            self.cmd.footRaiseHeight = 0.0
            self.cmd.bodyHeight = 0.0

            # --- LEER VISIÓN ---
            # Leemos la variable que actualiza el otro hilo
            detectado = False
            with LOCK:
                detectado = GLOBAL_PERSONA_DETECTADA

            # --- LÓGICA DE REACCIÓN ---
            if detectado:
                self.pitch_objetivo = -0.30
                
                # Lógica de ladrido
                ahora = time.time()
                if ahora - self.ultimo_ladrido >= self.COOLDOWN_LADRIDO:
                    print("¡Persona vista! -> GUAU!")
                    self.ultimo_ladrido = ahora
                    self.ladrido_audio.play()
            else:
                self.pitch_objetivo = 0.0

            # Suavizado de movimiento
            self.pitch_actual = self.suavizar(self.pitch_actual, self.pitch_objetivo)
            self.cmd.euler = [0.0, self.pitch_actual, 0.0]

            self.udp.SetSend(self.cmd)
            self.udp.Send()
            
            # El robot necesita este sleep preciso
            time.sleep(0.002)

# --- MAIN ---
if __name__ == "__main__":
    # 1. Iniciar el hilo de la cámara (se ejecuta en paralelo)
    t_camara = threading.Thread(target=hilo_camara)
    t_camara.daemon = True # Se cierra si el programa principal se cierra
    t_camara.start()

    # 2. Iniciar el control del robot en el hilo principal
    robot = RobotController()
    try:
        robot.run()
    except KeyboardInterrupt:
        print("Apagando...")