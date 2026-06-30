#!/usr/bin/env python3
"""
Script de entrenamiento para clasificador PointNet 3D usando dataset KITTI.
Extrae objetos individuales usando Bounding Boxes 3D y entrena un modelo PointNet.
"""
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Agregar el directorio raíz al path para poder importar models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pointnet_detector import PointNetClassifier

CLASS_TO_IDX = {
    "Pedestrian": 0,
    "Car": 1,
    "Obstacle": 2,
    "Furniture": 3,
    "DontCare": 4
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

class KittiObjectDataset(Dataset):
    def __init__(self, kitti_dir, split='train', num_points=256):
        self.kitti_dir = Path(kitti_dir)
        self.num_points = num_points
        self.samples = []
        
        # Cargar los IDs de entrenamiento
        imagesets_file = self.kitti_dir / "training" / "ImageSets" / f"{split}.txt"
        if not imagesets_file.exists():
            raise FileNotFoundError(f"ImageSets file not found: {imagesets_file}")
            
        with open(imagesets_file, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
            
        print(f"Loading objects from {len(ids)} frames...")
        
        for kitti_id in ids:
            bin_path = self.kitti_dir / "training" / "velodyne" / f"{kitti_id}.bin"
            label_path = self.kitti_dir / "training" / "label_2" / f"{kitti_id}.txt"
            
            if not bin_path.exists() or not label_path.exists():
                continue
                
            # Cargar nube de puntos (X, Y, Z, Intensity) -> conservar XYZ
            points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)[:, :3]
            
            # Cargar etiquetas
            with open(label_path, 'r') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if not parts:
                        continue
                        
                    obj_type = parts[0]
                    if obj_type not in CLASS_TO_IDX:
                        continue
                        
                    class_idx = CLASS_TO_IDX[obj_type]
                    
                    # Cargar bounding box 3D
                    # dimensions: height (z), width (y), length (x)
                    h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
                    # center: cx, cy, cz
                    cx, cy, cz = float(parts[11]), float(parts[12]), float(parts[13])
                    # rotation ry
                    ry = float(parts[14])
                    
                    # Cortar puntos dentro de la caja delimitadora orientada
                    # Traslación de puntos respecto al centro del box
                    dx = points[:, 0] - cx
                    dy = points[:, 1] - cy
                    dz = points[:, 2] - cz
                    
                    # Rotación inversa alrededor de Z (vertical en Lidar)
                    cos_ry = np.cos(-ry)
                    sin_ry = np.sin(-ry)
                    
                    rot_x = dx * cos_ry - dy * sin_ry
                    rot_y = dx * sin_ry + dy * cos_ry
                    
                    # Filtro inside/outside
                    mask = (
                        (np.abs(rot_x) <= l / 2.0) &
                        (np.abs(rot_y) <= w / 2.0) &
                        (np.abs(dz) <= h / 2.0)
                    )
                    
                    crop_points = points[mask]
                    
                    # Saltar si el objeto contiene muy pocos puntos
                    if len(crop_points) < 10:
                        continue
                        
                    # Centrar localmente los puntos
                    crop_points = crop_points - np.array([cx, cy, cz])
                    
                    self.samples.append({
                        'points': crop_points,
                        'label': class_idx,
                        'frame_id': kitti_id,
                        'obj_type': obj_type
                    })
                    
        print(f"Extracted {len(self.samples)} valid objects for training.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pts = sample['points']
        label = sample['label']
        
        # Remuestrear puntos a un número fijo (self.num_points)
        n_pts = pts.shape[0]
        if n_pts >= self.num_points:
            choice = np.random.choice(n_pts, self.num_points, replace=False)
        else:
            choice = np.random.choice(n_pts, self.num_points, replace=True)
            
        pts_sampled = pts[choice]  # shape: [num_points, 3]
        
        # Retornar como tensores de PyTorch
        # PointNet espera [3, num_points] o el clasificador maneja la transposición interna
        pts_tensor = torch.tensor(pts_sampled, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return pts_tensor, label_tensor

def main():
    parser = argparse.ArgumentParser(description="Train PointNet classifier on KITTI objects")
    parser.add_argument('--kitti-dir', default='dataset/kitti')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-points', type=int, default=256)
    parser.add_argument('--save-dir', default='models/checkpoints')
    args = parser.parse_args()
    
    # Crear directorio de checkpoints
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Crear dataset y dataloader
    try:
        dataset = KittiObjectDataset(args.kitti_dir, split='train', num_points=args.num_points)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    if len(dataset) == 0:
        print("No training objects extracted. Check your data/labels and data/preprocessed directories.")
        return
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # Definir dispositivo (GPU o CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Inicializar modelo
    model = PointNetClassifier(num_classes=len(CLASS_TO_IDX))
    model.to(device)
    
    # Optimizador y función de pérdida
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_loss = float('inf')
    loss_history = []
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for pts, labels in dataloader:
            # pts shape: [B, num_points, 3] -> transponer en el forward de PointNet
            pts = pts.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(pts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * pts.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = epoch_loss / total
        accuracy = correct / total
        loss_history.append(avg_loss)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy * 100:.2f}%")
        
        # Guardar mejor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = save_path / "best_model.pth"
            torch.save(model.state_dict(), str(best_model_path))
            print(f"  --> Saved new best model to {best_model_path}")
            
    print("Training finished.")
    
    # Guardar historial de pérdida
    history_path = save_path / "loss_history.json"
    with open(history_path, 'w') as f:
        json.dump(loss_history, f)
        
    # Intentar graficar e historial de pérdida si matplotlib está instalado
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(loss_history, label='Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(str(save_path / "loss_curve.png"))
        print(f"Saved loss curve to {save_path / 'loss_curve.png'}")
    except ImportError:
        print("Matplotlib not installed. Skipping loss curve plot.")

if __name__ == '__main__':
    main()
