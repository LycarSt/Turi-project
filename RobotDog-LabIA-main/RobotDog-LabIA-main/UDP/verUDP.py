import cv2
import socket
import numpy as np

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while True:
    data, _ = sock.recvfrom(65536)  # recibir paquetes
    npdata = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)

    if frame is not None:
        cv2.imshow("OAK-D Go1 Stream", frame)

    if cv2.waitKey(1) == ord('q'):
        break
