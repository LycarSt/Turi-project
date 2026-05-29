"""
capture_and_tsdf_fixed.py
Mapeado 3D con cámara OAK-D sin ROS (DepthAI + Open3D)
Corrige tamaños RGB/Depth y crea mapa 3D estable
"""

import depthai as dai
import numpy as np
import open3d as o3d
import cv2
import time
import os

# ====== CONFIGURACIÓN ======
VOXEL_LENGTH = 0.02
SDF_TRUNC = 0.06
N_FRAMES = 300
SAVE_DIR = "map_output"
os.makedirs(SAVE_DIR, exist_ok=True)

FX, FY = 600.0, 600.0
CX, CY = 320.0, 240.0

# ====== PIPELINE DepthAI ======
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(640, 480)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoL = pipeline.create(dai.node.MonoCamera)
monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

monoR = pipeline.create(dai.node.MonoCamera)
monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setSubpixel(True)

monoL.out.link(stereo.left)
monoR.out.link(stereo.right)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
camRgb.preview.link(xout_rgb.input)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# ====== TSDF ======
vol = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=VOXEL_LENGTH,
    sdf_trunc=SDF_TRUNC,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

print("Iniciando dispositivo OAK-D...")
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue("depth", maxSize=4, blocking=False)
    print("Capturando y fusionando frames... (Ctrl+C para detener)")

    frame_count = 0
    while frame_count < N_FRAMES:
        inRgb = qRgb.tryGet()
        inDepth = qDepth.tryGet()
        if inRgb is None or inDepth is None:
            time.sleep(0.01)
            continue

        rgb = inRgb.getCvFrame()                      # (480x640)
        depth = inDepth.getFrame().astype(np.float32) # (1080x1920)

        # --- Ajustar tamaño de profundidad ---
        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

        print(f"RGB: {rgb.shape}, Depth: {depth.shape}")

        # --- Limpiar y convertir profundidad ---
        depth_m = depth / 1000.0
        depth_m[np.isnan(depth_m)] = 0
        depth_m[depth_m < 0.1] = 0
        depth_m[depth_m > 5.0] = 0

        depth_uint16 = (depth_m * 1000).astype(np.uint16)
        if np.count_nonzero(depth_uint16) == 0:
            print("⚠️ Depth vacía, omitiendo frame.")
            continue

        # --- Crear RGBD válido ---
        rgb_c = np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        color_o3d = o3d.geometry.Image(rgb_c)
        depth_o3d = o3d.geometry.Image(depth_uint16)

        try:
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=color_o3d,
                depth=depth_o3d,
                depth_scale=1000.0,
                depth_trunc=5.0,
                convert_rgb_to_intensity=False
            )
        except Exception as e:
            print("❌ Error creando RGBD:", e)
            continue

        # --- Integrar en TSDF ---
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=640, height=480, fx=FX, fy=FY, cx=CX, cy=CY
        )
        pose = np.eye(4)
        vol.integrate(rgbd, intrinsic, np.linalg.inv(pose))

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"→ Integrados {frame_count} frames")

    print("Captura terminada, extrayendo resultados...")

# ====== EXPORTAR ======
mesh = vol.extract_triangle_mesh()
mesh.compute_vertex_normals()
pcd = mesh.sample_points_uniformly(number_of_points=200000)

mesh_path = os.path.join(SAVE_DIR, "mesh.ply")
pcd_path = os.path.join(SAVE_DIR, "map.ply")

o3d.io.write_triangle_mesh(mesh_path, mesh)
o3d.io.write_point_cloud(pcd_path, pcd)

# ====== VISUALIZAR NUBE DE PUNTOS ======
print("Mostrando nube de puntos...")
o3d.visualization.draw_geometries([pcd],
                                  window_name="Mapa 3D OAK-D",
                                  width=960,
                                  height=720,
                                  point_show_normal=False)

print("✅ Guardado:")
print(" -", mesh_path)
print(" -", pcd_path)
