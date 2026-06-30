#!/usr/bin/env python3
"""
Exporta el clasificador PointNet entrenado a formato ONNX.
"""
import os
import argparse
import torch
from pathlib import Path

# Agregar el directorio raíz al path para poder importar models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pointnet_detector import PointNetClassifier

def main():
    parser = argparse.ArgumentParser(description="Export PointNet to ONNX")
    parser.add_argument('--checkpoint', default='models/checkpoints/best_model.pth')
    parser.add_argument('--output', default='models/exports/detector_obstaculos.onnx')
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--num-points', type=int, default=256)
    args = parser.parse_args()
    
    # Asegurar que la carpeta de exportación existe
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar el modelo
    print(f"Loading PointNet model with {args.num_classes} classes...")
    model = PointNetClassifier(num_classes=args.num_classes)
    
    # 2. Cargar los pesos
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} not found. Please train the model first.")
        return
        
    print(f"Loading weights from {args.checkpoint}...")
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()
    
    # 3. Definir entrada dummy [BatchSize, NumPoints, XYZ]
    dummy_input = torch.randn(1, args.num_points, 3, dtype=torch.float32)
    
    # 4. Exportar a ONNX
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input_cloud'],
        output_names=['output_probs'],
        dynamic_axes={
            'input_cloud': {0: 'batch_size', 1: 'num_points'},
            'output_probs': {0: 'batch_size'}
        }
    )
    
    # 5. Validar con la biblioteca ONNX
    import onnx
    print("Validating exported ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("✔ Model successfully exported and checked!")

if __name__ == '__main__':
    main()
