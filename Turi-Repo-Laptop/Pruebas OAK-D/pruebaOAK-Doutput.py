#!/usr/bin/env python3
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
from datetime import datetime

# Path del modelo
nnPath = str((Path(__file__).parent / Path('/home/labia-001/Repo/turi-project/Pruebas OAK-D/yolov8n_coco_640x352.blob')).resolve().absolute())
if not Path(nnPath).exists():
    raise FileNotFoundError(f'Blob no encontrado. Ejecuta install_requirements.py')

# LabelMap YOLO
labelMap = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light","fire hydrant",
    "stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant",
    "bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster",
    "sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

syncNN = True
# Archivo donde se guardan las detecciones
LOG_FILE = "Pruebas OAK-D/detecciones_oakd.txt"

def guardar_detecciones_vivas(lista_detecciones):
    conteo = {}

    # Contar detecciones
    for clase in lista_detecciones:
        if clase not in conteo:
            conteo[clase] = 0
        conteo[clase] += 1

    # Sobrescribir archivo
    with open(LOG_FILE, "w") as f:
        for clase, cantidad in conteo.items():
            f.write(f"{clase}: {cantidad}\n")


# Pipeline
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
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

camRgb.preview.link(detectionNetwork.input)
if syncNN:
    detectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

detectionNetwork.out.link(nnOut.input)

# Connect to device
with dai.Device(pipeline) as device:

    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

            # Obtener clase detectada
            detected_class = labelMap[detection.label]

            # Dibujar en pantalla
            cv2.putText(frame, detected_class, (bbox[0] + 10, bbox[1] + 20),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%",
                        (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), color, 2)

        cv2.imshow(name, frame)

    while True:
        inRgb = qRgb.get()
        inDet = qDet.get()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if inDet is not None:
            detections = inDet.detections
            counter += 1

            # Obtener clases detectadas
            clases_detectadas = [labelMap[d.label] for d in detections]

            # Guardar detecciones actuales en vivo
            guardar_detecciones_vivas(clases_detectadas)

        if frame is not None:
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
