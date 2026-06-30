#!/usr/bin/env python3
"""
Convierte nubes de puntos .npy y etiquetas JSON al formato estándar KITTI.
Crea la estructura del dataset en dataset/kitti/
"""
import os
import json
import glob
import argparse
from pathlib import Path
import numpy as np

# Mapeo de clases del proyecto a KITTI
CLASS_MAPPING = {
    "PERSON": "Pedestrian",
    "VEHICLE": "Car",
    "OBSTACLE": "Obstacle",
    "FURNITURE": "Furniture",
    "BACKGROUND": "DontCare"
}

def convert_dataset(preprocessed_dir, labels_dir, output_dir):
    preprocessed_dir = Path(preprocessed_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    
    # Crear carpetas de salida conforme a KITTI
    velodyne_dir = output_dir / "training" / "velodyne"
    label_dir = output_dir / "training" / "label_2"
    image_dir = output_dir / "training" / "image_2"
    imagesets_dir = output_dir / "training" / "ImageSets"
    
    for d in [velodyne_dir, label_dir, image_dir, imagesets_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Buscar todos los JSON de etiquetas
    label_files = sorted(labels_dir.glob("*.json"))
    if not label_files:
        print(f"No JSON label files found in {labels_dir}. Please run mock label generator first.")
        return
        
    print(f"Found {len(label_files)} label files. Converting to KITTI format...")
    
    id_map = {}
    train_ids = []
    
    for idx, label_path in enumerate(label_files):
        stem = label_path.stem  # ej. 1758208492.275396347
        npy_path = preprocessed_dir / f"{stem}.npy"
        
        if not npy_path.exists():
            print(f"Warning: Preprocessed cloud {npy_path} not found for label {label_path}. Skipping.")
            continue
            
        kitti_id = f"{idx:06d}"
        train_ids.append(kitti_id)
        id_map[kitti_id] = stem
        
        # 1. Convertir nube de puntos a .bin (X, Y, Z, Intensity)
        xyz = np.load(str(npy_path))  # shape: [N, 3]
        
        # Agregar columna de intensidad de ceros
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        xyzi = np.hstack((xyz.astype(np.float32), intensity))  # shape: [N, 4]
        
        bin_out_path = velodyne_dir / f"{kitti_id}.bin"
        xyzi.tofile(str(bin_out_path))
        
        # 2. Convertir etiquetas JSON a KITTI .txt
        with open(label_path, 'r') as f:
            label_data = json.load(f)
            
        txt_out_path = label_dir / f"{kitti_id}.txt"
        with open(txt_out_path, 'w') as out_f:
            for ann in label_data.get('annotations', []):
                obj_type = CLASS_MAPPING.get(ann.get('type'), ann.get('type', 'Obstacle'))
                
                # Obtener atributos de oclusión y truncamiento
                attrs = ann.get('attributes', {})
                truncated = 1.0 if attrs.get('truncated', False) else 0.0
                occluded = 1 if attrs.get('occluded', False) else 0 # 0=visible, 1=occluded
                
                # KITTI bounding box 2D (Dummy values since we are lidar-only)
                bbox2d = [0.0, 0.0, 50.0, 50.0]
                
                # 3D bounding box dimensions (height, width, length)
                # En nuestro esquema docs/label_schema.md: dimensions = [dx, dy, dz]
                # En KITTI: height (z), width (y), length (x)
                dims = ann.get('bbox3d', {}).get('dimensions', [1.0, 1.0, 1.0])
                dx, dy, dz = dims[0], dims[1], dims[2]
                kitti_dims = [dz, dy, dx]  # height, width, length
                
                # 3D bounding box center [x, y, z]
                center = ann.get('bbox3d', {}).get('center', [0.0, 0.0, 0.0])
                
                # rotation y
                ry = ann.get('bbox3d', {}).get('rotation_y', 0.0)
                
                # Alpha (observation angle): standard dummy value is -10
                alpha = -10.0
                
                # Formato KITTI label:
                # 1. type
                # 2. truncated
                # 3. occluded
                # 4. alpha
                # 5. bbox 2d (4 floats: left, top, right, bottom)
                # 6. dimensions 3d (3 floats: height, width, length)
                # 7. location 3d (3 floats: x, y, z)
                # 8. rotation_y
                line_parts = [
                    obj_type,
                    f"{truncated:.2f}",
                    str(occluded),
                    f"{alpha:.2f}",
                    f"{bbox2d[0]:.2f}", f"{bbox2d[1]:.2f}", f"{bbox2d[2]:.2f}", f"{bbox2d[3]:.2f}",
                    f"{kitti_dims[0]:.2f}", f"{kitti_dims[1]:.2f}", f"{kitti_dims[2]:.2f}",
                    f"{center[0]:.2f}", f"{center[1]:.2f}", f"{center[2]:.2f}",
                    f"{ry:.2f}"
                ]
                out_f.write(" ".join(line_parts) + "\n")
                
        # 3. Crear una imagen ficticia vacía de 0 bytes si es necesario para compatibilidad
        dummy_img_path = image_dir / f"{kitti_id}.png"
        if not dummy_img_path.exists():
            dummy_img_path.touch()

    # Escribir ImageSets/train.txt
    train_txt_path = imagesets_dir / "train.txt"
    with open(train_txt_path, 'w') as f:
        for tid in train_ids:
            f.write(f"{tid}\n")
            
    # Guardar mapa de IDs
    map_path = output_dir / "kitti_id_map.json"
    with open(map_path, 'w') as f:
        json.dump(id_map, f, indent=2)
        
    print(f"Successfully converted {len(train_ids)} pairs to KITTI dataset format at {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert preprocessed PCD to KITTI dataset format")
    parser.add_argument('--preprocessed-dir', default='data/preprocessed')
    parser.add_argument('--labels-dir', default='data/labels')
    parser.add_argument('--output-dir', default='dataset/kitti')
    args = parser.parse_args()
    
    convert_dataset(args.preprocessed_dir, args.labels_dir, args.output_dir)

if __name__ == '__main__':
    main()
