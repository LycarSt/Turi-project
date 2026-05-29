import os
import glob
import time
from pcd_parser import PCDParser, PCDProcessor
from visualize_map import PCDVisualizer

def main():
    import sys
    import io
    # Forzar codificación UTF-8 en Windows para evitar caídas por caracteres unicode/emojis
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
        
    print("======================================================================")
    # Título estilizado y premium
    print("   UNITREE GO1 - PIPELINE DE DETECCION DE ZONAS SEGURAS Y RESCATE")
    print("======================================================================\n")
    
    # 1. Ubicar carpetas y archivos en el workspace de Windows
    pcd_folder = r"c:\Users\curay\OneDrive\Documentos\LABIAR\pcd_output\pcd_output\Pruebas_posicion_1_up"
    
    # Si la carpeta no existe, intentar buscar en el directorio superior
    if not os.path.exists(pcd_folder):
        pcd_folder = r"..\pcd_output\pcd_output\Pruebas_posicion_1_up"
        
    pcd_files = sorted(glob.glob(os.path.join(pcd_folder, "*.pcd")))
    
    if not pcd_files:
        print(f"ERROR: No se encontraron archivos .pcd en la ruta: {pcd_folder}")
        return
        
    test_file = pcd_files[0]
    print(f"[*] Archivos PCD detectados: {len(pcd_files)}")
    print(f"[*] Procesando archivo de prueba: {os.path.basename(test_file)}\n")
    
    # 2. Benchmarks de Rendimiento
    times = {}
    
    # ETAPA 1: Lectura y Filtrado NaN
    t_start = time.time()
    points_dict = PCDParser.read_pcd(test_file)
    times['Parseo y Limpieza'] = (time.time() - t_start) * 1000
    print(f"[+] 1. Lectura exitosa: {len(points_dict['x'])} puntos válidos filtrados.")
    
    # ETAPA 2: Segmentación RANSAC
    processor = PCDProcessor(points_dict)
    t_start = time.time()
    ground_mask, obstacle_mask, plane_coeffs = processor.segment_ground_ransac()
    times['RANSAC Terreno'] = (time.time() - t_start) * 1000
    A, B, C, D = plane_coeffs
    print(f"[+] 2. RANSAC completado: Plano de suelo detectado en Z = {-D/C:.3f}m. Ecuación: {A:.2f}x + {B:.2f}y + {C:.2f}z + {D:.2f} = 0")
    
    # ETAPA 3: Agrupamiento DBSCAN de Obstáculos e Identificación Humana
    t_start = time.time()
    obs_xyz, labels, person_candidates = processor.cluster_obstacles_dbscan(obstacle_mask)
    times['Clustering DBSCAN'] = (time.time() - t_start) * 1000
    print(f"[+] 3. DBSCAN completado: {len(np.unique(labels)) - (1 if -1 in labels else 0)} obstáculos físicos detectados.")
    
    # Mostrar alertas de supervivientes si existen
    if person_candidates:
        print("\n\t⚠️  ¡ALERTA DE RESCATE: SOBREVIVIENTE(S) DETECTADO(S)!")
        for c in person_candidates:
            print(f"\t -> Persona {c['label']}: Centro X={c['center'][0]:.2f}m, Y={c['center'][1]:.2f}m | Altura: {c['height']:.2f}m | Puntos: {c['num_points']}")
        print("")
    else:
        print("\t[i] No se detectaron firmas humanas de búsqueda y rescate en este barrido.\n")
        
    # ETAPA 4: Generación de Rejilla de Ocupación 2D (Grid Map)
    t_start = time.time()
    grid, x_bins, y_bins = processor.generate_occupancy_grid(ground_mask, obstacle_mask)
    times['Grid de Ocupación 2D'] = (time.time() - t_start) * 1000
    
    # Calcular porcentaje de navegabilidad del aula
    total_cells = grid.size
    free_cells = np.sum(grid == 0)
    occupied_cells = np.sum(grid == 1)
    unknown_cells = np.sum(grid == -1)
    
    print(f"[+] 4. Grid generado: Celda unitaria de 15cm x 15cm | Tamaño: {grid.shape[0]}x{grid.shape[1]}")
    print(f"\t - Zona Segura (Navegable): {free_cells} celdas ({free_cells/total_cells*100:.1f}%)")
    print(f"\t - Zona de Peligro (Obstáculo): {occupied_cells} celdas ({occupied_cells/total_cells*100:.1f}%)")
    print(f"\t - Zona no Mapeada (Unknown): {unknown_cells} celdas ({unknown_cells/total_cells*100:.1f}%)")
    
    # 3. Guardar reportes gráficos interactivos
    output_dir = os.path.dirname(test_file)
    # Guardamos los gráficos en la misma carpeta Codigo_Data para acceso rápido
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    img_3d_path = os.path.join(demo_dir, "demo_result_3d.png")
    img_2d_path = os.path.join(demo_dir, "demo_result_2d.png")
    
    print("\n[*] Generando visualizaciones y renderizados...")
    PCDVisualizer.plot_3d_segmentation(processor, ground_mask, obs_xyz, labels, person_candidates, save_path=img_3d_path)
    PCDVisualizer.plot_2d_occupancy_grid(grid, x_bins, y_bins, person_candidates, save_path=img_2d_path)
    
    # 4. Tabla de Tiempos y Benchmark Final
    print("\n======================================================================")
    print("                  BENCHMARK DE RENDIMIENTO EN TIEMPO REAL")
    print("======================================================================")
    total_pipeline_time = sum(times.values())
    for etapa, ms in times.items():
        print(f" - {etapa.ljust(25)}: {ms:8.2f} ms")
    print("----------------------------------------------------------------------")
    print(f" * TIEMPO TOTAL PIPELINE  : {total_pipeline_time:8.2f} ms  (Frecuencia: {1000/total_pipeline_time:.1f} Hz)")
    print("======================================================================\n")
    
    print(f"[!] PROCESO COMPLETADO. Las visualizaciones se guardaron en:")
    print(f"    1. Renderizado 3D Segmentado: {os.path.basename(img_3d_path)}")
    print(f"    2. Mapa de Ocupación 2D     : {os.path.basename(img_2d_path)}\n")

if __name__ == "__main__":
    import numpy as np
    main()
