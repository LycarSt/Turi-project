import depthai as dai
import cv2

# --- CONFIGURAR AQUÍ ---
# Cambia entre: dai.MonoCameraProperties.SensorResolution.THE_400_P,
#               dai.MonoCameraProperties.SensorResolution.THE_720_P,
#               dai.MonoCameraProperties.SensorResolution.THE_800_P
RESOLUCION = dai.MonoCameraProperties.SensorResolution.THE_720_P  
# -----------------------

# Crear pipeline
pipeline = dai.Pipeline()

# Cámaras mono izquierda y derecha
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoLeft.setResolution(RESOLUCION)
monoRight.setResolution(RESOLUCION)

# Nodo de profundidad
stereo = pipeline.create(dai.node.StereoDepth)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# Salida de disparity/depth
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

# Iniciar dispositivo
with dai.Device(pipeline) as device:
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        inDepth = qDepth.get()
        depth_frame = inDepth.getFrame()  # matriz en mm

        # Mostrar resolución real
        h, w = depth_frame.shape
        texto = f"Resolucion profundidad: {w} x {h}"
        vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        cv2.putText(vis, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Mapa de profundidad", vis)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
