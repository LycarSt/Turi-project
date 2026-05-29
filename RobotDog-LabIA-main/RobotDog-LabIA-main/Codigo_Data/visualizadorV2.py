import open3d as o3d
import glob
import os

# Carpeta donde tienes los .pcd
pcd_folder = "/home/labia-001/pcd_output/Pruebas_posicion_2"
pcd_files = sorted(glob.glob(os.path.join(pcd_folder, "*.pcd")))

print(f"Encontrados {len(pcd_files)} archivos .pcd")

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="Visualizador PCD", width=1280, height=720)

pcd = o3d.io.read_point_cloud(pcd_files[0])
vis.add_geometry(pcd)

# Configuración inicial de cámara
ctr = vis.get_view_control()
bbox = pcd.get_axis_aligned_bounding_box()
ctr.set_front([0, 1, 1])        # Cámara en Z+, mirando hacia abajo
ctr.set_lookat(bbox.get_center())
ctr.set_up([0, -1, 0])          # Ajusta la orientación para que no quede invertida
ctr.set_zoom(0.2)



idx = {"value": 0}  # índice mutable

def next_pcd(vis):
    idx["value"] = (idx["value"] + 1) % len(pcd_files)
    new_pcd = o3d.io.read_point_cloud(pcd_files[idx["value"]])
    vis.clear_geometries()
    vis.add_geometry(new_pcd)

    # Reajustar cámara en cada nube
    ctr = vis.get_view_control()
    bbox = new_pcd.get_axis_aligned_bounding_box()
    ctr.set_front([0, 1, 1])        # Cámara en Z+, mirando hacia abajo
    ctr.set_lookat(bbox.get_center())
    ctr.set_up([0, -1, 0])          # Ajusta la orientación para que no quede invertida
    ctr.set_zoom(0.2)



    print(f"Mostrando {pcd_files[idx['value']]}")
    return False


# Espacio = siguiente nube
vis.register_key_callback(ord(" "), next_pcd)

vis.run()
vis.destroy_window()