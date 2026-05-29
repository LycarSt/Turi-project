import open3d as o3d
import glob
import os
import numpy as np
from pcd_parser import PCDParser, PCDProcessor

def get_pcd_folder():
    """
    Busca de manera inteligente la carpeta de archivos PCD, 
    gestionando la portabilidad entre la máquina Linux del perro y tu PC de Windows.
    """
    paths = [
        # Ruta en tu PC de desarrollo Windows (OneDrive)
        r"c:\Users\curay\OneDrive\Documentos\LABIAR\pcd_output\pcd_output\Pruebas_posicion_1_up",
        # Ruta por defecto de Linux del manual
        "/home/labime puedea-001/pcd_output/Pruebas_posicion_2",
        "/home/labia-001/pcd_output/Pruebas_posicion_1_up",
        # Rutas relativas
        "../pcd_output/pcd_output/Pruebas_posicion_1_up",
        "./pcd_output/pcd_output/Pruebas_posicion_1_up"
    ]
    
    for p in paths:
        if os.path.exists(p):
            print(f"[*] Directorio de datos PCD encontrado en: {p}")
            return p
            
    # Fallback si no se encuentra
    print("⚠️  No se encontró ninguna ruta de PCD predefinida. Usando el directorio actual...")
    return "."

def process_and_color_pcd(file_path):
    """
    Carga un PCD binario, ejecuta RANSAC y DBSCAN en tiempo real, 
    y retorna la nube coloreada junto con las cajas 3D de personas.
    """
    # 1. Leer y filtrar NaNs
    points_dict = PCDParser.read_pcd(file_path)
    xyz = np.stack((points_dict['x'], points_dict['y'], points_dict['z']), axis=1)
    
    # 2. Inicializar procesador
    processor = PCDProcessor(points_dict)
    
    # 3. RANSAC para Suelo Seguro
    ground_mask, obstacle_mask, _ = processor.segment_ground_ransac()
    
    # 4. DBSCAN para Obstáculos y Personas
    obs_xyz, labels, person_candidates = processor.cluster_obstacles_dbscan(obstacle_mask)
    
    # 5. Crear objeto Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # 6. Colorear la nube en 3D
    # Colores normalizados en float [0.0, 1.0]
    colors = np.zeros((len(xyz), 3))
    
    # Suelo seguro -> Verde suave
    colors[ground_mask] = [0.15, 0.68, 0.38] 
    
    # Ruido / Obstáculos generales -> Gris oscuro
    colors[obstacle_mask] = [0.25, 0.25, 0.25]
    
    # Colorear obstáculos agrupados por clústeres DBSCAN
    if len(obs_xyz) > 0 and len(labels) > 0:
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
            
        # Generar una paleta de colores vibrantes para objetos
        # Usamos una paleta cíclica simple
        colormap_colors = [
            [0.1, 0.6, 0.8], [0.9, 0.6, 0.1], [0.6, 0.2, 0.8], 
            [0.1, 0.8, 0.2], [0.8, 0.1, 0.2], [0.9, 0.8, 0.1],
            [0.1, 0.9, 0.7], [0.7, 0.1, 0.9], [0.4, 0.4, 0.4]
        ]
        
        # Mapear los índices originales a los puntos en la nube total
        obstacle_indices = np.where(obstacle_mask & (processor.z < 1.2))[0]
        
        for i, label in enumerate(unique_labels):
            # Encontrar si este clúster es persona
            is_person = any(c['label'] == label for c in person_candidates)
            
            cluster_mask_local = (labels == label)
            global_indices_in_obstacles = obstacle_indices[cluster_mask_local]
            
            if is_person:
                # Color rojo brillante para personas
                colors[global_indices_in_obstacles] = [1.0, 0.1, 0.1]
            else:
                # Color único del clúster
                c_color = colormap_colors[i % len(colormap_colors)]
                colors[global_indices_in_obstacles] = c_color
                
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 7. Crear cajas contenedoras 3D (Bounding Boxes) para sobrevivientes
    geometries_to_draw = [pcd]
    
    for person in person_candidates:
        # Calcular límites de la caja
        center = np.array(person['center'])
        dims = np.array(person['dimensions'])
        
        min_bound = center - dims/2.0
        max_bound = center + dims/2.0
        
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        bbox.color = (1.0, 0.0, 0.0) # Caja roja para personas
        geometries_to_draw.append(bbox)
        
    return geometries_to_draw, person_candidates

def main():
    pcd_folder = get_pcd_folder()
    pcd_files = sorted(glob.glob(os.path.join(pcd_folder, "*.pcd")))
    
    if not pcd_files:
        print(f"❌ Error: No se encontraron archivos .pcd en {pcd_folder}")
        return
        
    print(f"======================================================================")
    print(f"    VISUALIZADOR INTELIGENTE 3D EN TIEMPO REAL - UNITREE GO1")
    print(f"======================================================================")
    print(f" -> Encontrados: {len(pcd_files)} barridos LiDAR.")
    print(f" -> Teclas:")
    print(f"    [ESPACIO] : Cargar siguiente barrido .pcd (Simulación de Video)")
    print(f"    [R]       : Reiniciar vista de la cámara")
    print(f"    [Q]       : Salir del programa")
    print(f" -> Código de Colores:")
    print(f"    - Verde Claro  : ZONA SEGURA (Suelo Transitable)")
    print(f"    - Rojo y Cajas : ¡PERSONAS DETECTADAS (Sobrevivientes)!")
    print(f"    - Otros Colores: Muebles, Paredes y Objetos individuales (DBSCAN)")
    print(f"    - Gris Oscuro  : Ruido / Puntos sueltos")
    print(f"======================================================================\n")

    # Iniciar la ventana interactiva de Open3D
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Unitree Go1 - IA Real-Time Mapeo 3D", width=1280, height=720)
    
    idx = {"value": 0}  # índice mutable
    
    # Procesar primer archivo
    geometrias, personas = process_and_color_pcd(pcd_files[0])
    for g in geometrias:
        vis.add_geometry(g)
        
    # Imprimir detecciones en consola
    print(f"[*] Mostrando {os.path.basename(pcd_files[0])}")
    if personas:
        print(f"    ⚠️  ¡ATENCION! Detectadas {len(personas)} personas en este cuadro.")
        
    # Configuración inicial de cámara de alta estética
    ctr = vis.get_view_control()
    pcd_only = geometrias[0]
    bbox = pcd_only.get_axis_aligned_bounding_box()
    ctr.set_front([0, 1, 1])        # Cámara inclinada mirando hacia adelante y abajo
    ctr.set_lookat(bbox.get_center())
    ctr.set_up([0, -1, 0])          # Orientación vertical correcta
    ctr.set_zoom(0.25)
    
    def next_pcd(vis):
        idx["value"] = (idx["value"] + 1) % len(pcd_files)
        current_file = pcd_files[idx["value"]]
        
        # Procesar y calcular la nueva geometría segmentada
        new_geometrias, new_personas = process_and_color_pcd(current_file)
        
        # Limpiar geometrías anteriores de la escena
        vis.clear_geometries()
        
        # Añadir las nuevas geometrías (nube coloreada + cajas 3D)
        for g in new_geometrias:
            vis.add_geometry(g)
            
        # Reajustar la cámara para mantener la perspectiva fluida
        ctr = vis.get_view_control()
        bbox = new_geometrias[0].get_axis_aligned_bounding_box()
        ctr.set_front([0, 1, 1])
        ctr.set_lookat(bbox.get_center())
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.25)
        
        print(f"[*] Mostrando {os.path.basename(current_file)}")
        if new_personas:
            print(f"    ⚠️  ¡ATENCION! Detectadas {len(new_personas)} personas en este cuadro.")
            for p in new_personas:
                print(f"        -> ID_{p['label']}: Centro X={p['center'][0]:.2f}m, Y={p['center'][1]:.2f}m | Altura: {p['height']:.2f}m")
                
        return False  # Indica que requiere actualizar la pantalla

    # Registrar evento de la tecla ESPACIO
    vis.register_key_callback(ord(" "), next_pcd)
    
    # Ejecutar bucle visualizador interactivo
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
