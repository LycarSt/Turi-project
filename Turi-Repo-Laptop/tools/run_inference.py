#!/usr/bin/env python3
"""
Ejecuta el pipeline de inferencia completo en tiempo real usando el modelo PointNet ONNX.
Preprocesa, aplica RANSAC para remover suelo, DBSCAN para agrupar obstáculos
y ONNX para clasificar cada objeto detectado.
"""
import os
import argparse
import json
import time
import numpy as np
import open3d as o3d
import onnxruntime as ort
from pathlib import Path

IDX_TO_CLASS = {
    0: "Pedestrian",
    1: "Car",
    2: "Obstacle",
    3: "Furniture",
    4: "DontCare"
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
    # Segmentación del plano del suelo usando RANSAC
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=1000
    )
    
    # Extraer el resto de puntos (obstáculos)
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

def run_inference(onnx_path, pcd_path, output_report, eps=0.12, min_points=25, distance_threshold=0.06):
    # Cargar sesión de ONNX
    print(f"Loading ONNX model from {onnx_path}...")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    # Cargar nube de puntos
    print(f"Loading point cloud from {pcd_path}...")
    pcd_path = Path(pcd_path)
    if pcd_path.suffix == '.npy':
        pts = np.load(str(pcd_path))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
    else:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        
    # Preprocesar
    pcd = preprocess_cloud(pcd)
    
    # Segmentar y agrupar en clusters
    print(f"Segmenting ground plane (threshold={distance_threshold}) and clustering (eps={eps}, min_points={min_points})...")
    clusters = segment_and_cluster(pcd, eps=eps, min_points=min_points, distance_threshold=distance_threshold)
    print(f"Found {len(clusters)} obstacle clusters.")
    
    detections = []
    
    for i, cluster in enumerate(clusters):
        points = np.asarray(cluster.points, dtype=np.float32)
        n_pts = points.shape[0]
        
        # 1. Filtro por cantidad de puntos
        if n_pts < 30:
            continue
            
        # Calcular centro y dimensiones (Bounding Box)
        bbox = cluster.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent() # [dx, dy, dz]
        
        # 2. Filtro de volumen tridimensional mínimo
        volume = extent[0] * extent[1] * extent[2]
        if volume < 0.003:
            continue
            
        # 3. Filtro de altura mínima (eje Z)
        if extent[2] < 0.08:
            continue
            
        # Centrar localmente los puntos para la red PointNet
        centered_points = points - center
        
        # Remuestrear a 256 puntos
        if n_pts >= 256:
            choice = np.random.choice(n_pts, 256, replace=False)
        else:
            choice = np.random.choice(n_pts, 256, replace=True)
        sampled_points = centered_points[choice]
        
        # Preparar entrada para ONNX [1, 256, 3]
        input_data = np.expand_dims(sampled_points, axis=0).astype(np.float32)
        
        # Ejecutar inferencia ONNX
        t_start = time.time()
        outputs = session.run(None, {input_name: input_data})
        inference_time_ms = (time.time() - t_start) * 1000.0
        
        probs = outputs[0][0]
        class_idx = np.argmax(probs)
        class_name = IDX_TO_CLASS.get(class_idx, "Unknown")
        confidence = float(np.exp(probs[class_idx]) / np.sum(np.exp(probs))) # softmax simple
        
        # 4. Gating de Confianza e Heurísticas de Tamaño Físico
        if confidence < 0.75:
            class_name = "Obstacle"
            
        # Sanity check: Pedestrian debe medir >= 0.8m
        if class_name == "Pedestrian" and extent[2] < 0.8:
            class_name = "Obstacle"
            
        # Sanity check: Car debe medir >= 1.2m
        if class_name == "Car" and max(extent[0], extent[1]) < 1.2:
            class_name = "Obstacle"
            
        # Distancia euclidiana respecto al robot (origen 0,0,0)
        distance = float(np.linalg.norm(center))
        
        detection = {
            "object_id": len(detections) + 1,
            "class": class_name,
            "confidence": confidence,
            "center": [float(c) for c in center],
            "dimensions": [float(e) for e in extent],
            "distance_meters": distance,
            "num_points": n_pts,
            "inference_time_ms": inference_time_ms
        }
        detections.append(detection)
        
    # Imprimir reporte en consola
    print("\n" + "="*50)
    print(" DETECTIONS SUMMARY")
    print("="*50)
    for det in detections:
        print(f"ID {det['object_id']}: {det['class']} (Conf: {det['confidence']*100:.1f}%) "
              f"at dist: {det['distance_meters']:.2f}m | Loc: {det['center']} | Size: {det['dimensions']}")
    print("="*50 + "\n")
    
    # Guardar reporte JSON
    if output_report:
        report_path = Path(output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as rf:
            json.dump({
                "timestamp": time.time(),
                "file_processed": str(pcd_path),
                "num_detections": len(detections),
                "detections": detections
            }, rf, indent=2)
        print(f"Report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Run real-time inference using PointNet ONNX model")
    parser.add_argument('--onnx', default='models/exports/detector_obstaculos.onnx')
    parser.add_argument('--input', default='data/preprocessed/1758208492.275396347.npy')
    parser.add_argument('--report', default='reports/inference_report.json')
    parser.add_argument('--eps', type=float, default=0.12, help='DBSCAN epsilon parameter')
    parser.add_argument('--min-points', type=int, default=25, help='DBSCAN min points parameter')
    parser.add_argument('--ransac-threshold', type=float, default=0.06, help='RANSAC ground distance threshold')
    args = parser.parse_args()
    
    run_inference(args.onnx, args.input, args.report, eps=args.eps, min_points=args.min_points, distance_threshold=args.ransac_threshold)

if __name__ == '__main__':
    main()
