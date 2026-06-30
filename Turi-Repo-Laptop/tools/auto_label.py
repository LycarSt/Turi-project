#!/usr/bin/env python3
"""
Script de etiquetado automático para labelCloud.
Procesa todas las nubes de puntos .ply en pointclouds/, ejecuta la inferencia
(RANSAC + DBSCAN + PointNet ONNX) y genera archivos .json en labels/
con el formato exacto requerido por labelCloud (centroid_abs).
"""
import os
import glob
import json
import numpy as np
import open3d as o3d
import onnxruntime as ort
from pathlib import Path

IDX_TO_CLASS = {
    0: "Pedestrian",
    1: "Car",
    2: "Obstacle",
    3: "Furniture"
}

def preprocess_cloud(pcd, voxel_size=0.02, remove_outliers=(20, 2.0)):
    # 1. Eliminar outliers
    if remove_outliers:
        nb, std = remove_outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=int(nb), std_ratio=float(std))
        
    # 2. Voxel downsample
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        
    return pcd

def segment_and_cluster(pcd, eps=0.12, min_points=25, distance_threshold=0.06):
    # RANSAC para el suelo
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=1000
    )
    
    # Obstáculos (puntos que no son suelo)
    rest = pcd.select_by_index(inliers, invert=True)
    
    # Agrupar en clusters usando DBSCAN
    labels = np.array(rest.cluster_dbscan(eps=eps, min_points=min_points))
    
    clusters = []
    max_label = labels.max()
    for i in range(max_label + 1):
        idx = np.where(labels == i)[0]
        cluster = rest.select_by_index(idx)
        clusters.append(cluster)
        
    return clusters

def main():
    project_dir = Path("/home/labia-001/Repo/turi-project")
    onnx_path = project_dir / "models/exports/detector_obstaculos.onnx"
    pointclouds_dir = project_dir / "pointclouds"
    labels_dir = project_dir / "labels"
    
    # Crear carpetas si no existen
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Verificar ONNX
    if not onnx_path.exists():
        print(f"ERROR: Modelo ONNX no encontrado en {onnx_path}")
        return
        
    # Buscar archivos .ply
    ply_files = sorted(glob.glob(str(pointclouds_dir / "*.ply")))
    if not ply_files:
        print(f"ERROR: No se encontraron archivos .ply en {pointclouds_dir}")
        return
        
    print(f"Iniciando auto-etiquetado. Cargando modelo ONNX...")
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    
    print(f"Se procesarán {len(ply_files)} archivos de nube de puntos...")
    
    success_count = 0
    for filepath in ply_files:
        filepath = Path(filepath)
        pcd = o3d.io.read_point_cloud(str(filepath))
        
        # Preprocesar nube
        pcd_clean = preprocess_cloud(pcd)
        
        # Segmentación y agrupamiento
        clusters = segment_and_cluster(pcd_clean)
        
        objects = []
        for cluster in clusters:
            points = np.asarray(cluster.points, dtype=np.float32)
            n_pts = points.shape[0]
            
            # Filtro por cantidad mínima de puntos en el cluster
            if n_pts < 30:
                continue
                
            # Obtener Bounding Box (AABB)
            bbox = cluster.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            extent = bbox.get_extent() # [largo, ancho, alto]
            
            # Filtro de volumen mínimo
            volume = extent[0] * extent[1] * extent[2]
            if volume < 0.003:
                continue
                
            # Filtro de altura mínima
            if extent[2] < 0.08:
                continue
                
            # Centrar localmente para PointNet
            centered_points = points - center
            
            # Remuestrear a 256 puntos
            if n_pts >= 256:
                choice = np.random.choice(n_pts, 256, replace=False)
            else:
                choice = np.random.choice(n_pts, 256, replace=True)
            sampled_points = centered_points[choice]
            
            # Formato de entrada ONNX [1, 256, 3]
            input_data = np.expand_dims(sampled_points, axis=0).astype(np.float32)
            
            # Inferencia ONNX
            outputs = session.run(None, {input_name: input_data})
            probs = outputs[0][0]
            class_idx = np.argmax(probs)
            class_name = IDX_TO_CLASS.get(class_idx, "Obstacle")
            confidence = float(np.exp(probs[class_idx]) / np.sum(np.exp(probs)))
            
            # Gating de Confianza e Heurísticas
            if confidence < 0.75:
                class_name = "Obstacle"
                
            # Pedestrian debe medir >= 0.8m
            if class_name == "Pedestrian" and extent[2] < 0.8:
                class_name = "Obstacle"
                
            # Car debe medir >= 1.2m
            if class_name == "Car" and max(extent[0], extent[1]) < 1.2:
                class_name = "Obstacle"
                
            # Registrar objeto en el formato de labelCloud (centroid_abs)
            obj = {
                "name": class_name,
                "centroid": {
                    "x": round(float(center[0]), 6),
                    "y": round(float(center[1]), 6),
                    "z": round(float(center[2]), 6)
                },
                "dimensions": {
                    "length": round(float(extent[0]), 6),
                    "width": round(float(extent[1]), 6),
                    "height": round(float(extent[2]), 6)
                },
                "rotations": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0
                }
            }
            objects.append(obj)
            
        # Armar el esquema JSON final de labelCloud
        label_data = {
            "folder": "pointclouds",
            "filename": filepath.name,
            "path": str(filepath),
            "objects": objects
        }
        
        # Guardar en labels/<nombre_archivo>.json
        json_path = labels_dir / f"{filepath.stem}.json"
        with open(json_path, "w") as jf:
            json.dump(label_data, jf, indent=4)
            
        success_count += 1
        
    print(f"¡Terminado! Se procesaron {success_count} archivos y se generaron sus etiquetas preestablecidas en: {labels_dir}/")

if __name__ == "__main__":
    main()
