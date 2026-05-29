import depthai as dai
import cv2
import socket
import numpy as np

UDP_IP = "192.168.12.135"  # IP de tu laptop
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

pipeline = dai.Pipeline()

# Nodos de cámaras estéreo
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")

monoLeft.setCamera("left")
monoRight.setCamera("right")
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Configurar el nodo de profundidad
depth.setConfidenceThreshold(200)   # mejora la calidad del mapa
depth.setMedianFilter(dai.MedianFilter.KERNEL_7x7)  # suaviza el mapa de profundidad

# Enlazar cámaras estéreo → nodo de profundidad → salida
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.depth.link(xoutDepth.input)

with dai.Device(pipeline) as device:
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        frame = qDepth.get().getFrame()  # 16-bit depth map (en milímetros)
        
        # Recortar valores extremos para mejorar contraste visual
        frame = np.clip(frame, 500, 5000)  # límites en mm (0.5m - 5m)

        # Convertir a rango 0-255 para visualización
        frame_vis = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame_vis = np.uint8(frame_vis)

        # Aplicar un mapa de color para mejor visualización
        frame_colored = cv2.applyColorMap(frame_vis, cv2.COLORMAP_JET)

        # Comprimir a JPEG (para transmisión UDP)
        ret, buffer = cv2.imencode(".jpg", frame_colored, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            sock.sendto(buffer.tobytes(), (UDP_IP, UDP_PORT))
