import cv2
import depthai as dai
import socket


UDP_IP = "192.168.12.135"   
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Crear pipeline
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(640, 480)
cam.setInterleaved(False)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.preview.link(xout.input)

with dai.Device(pipeline) as device:
    q = device.getOutputQueue("video", maxSize=4, blocking=False)

    while True:
        frame = q.get().getCvFrame()
        # Comprimir frame a JPEG
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            sock.sendto(buffer.tobytes(), (UDP_IP, UDP_PORT))
