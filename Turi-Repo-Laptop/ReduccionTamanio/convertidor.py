import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os

# Ruta al archivo .off (descargado de ModelNet40 por ejemplo)
off_file_path = "sofa_0001.off"  # Cambia esto por el archivo que tengas

# Verifica que el archivo exista
if not os.path.exists(off_file_path):
    raise FileNotFoundError(f"No se encontró el archivo: {off_file_path}")

# Leer archivo .off como malla triangular
mesh = o3d.io.read_triangle_mesh(off_file_path)

# Convertir a nube de puntos desde los vértices de la malla
pointcloud = mesh.sample_points_uniformly(number_of_points=2048)

# Convertir a numpy array
points = np.asarray(pointcloud.points)

# Separar coordenadas X, Y, Z
x, y, z = points[:, 0], points[:, 1], points[:, 2]

# Normalizar Z para usarlo como intensidad entre 0 y 1
z_normalized = (z - z.min()) / (z.max() - z.min())

# Crear figura y graficar proyección XY con intensidad Z
plt.figure(figsize=(6, 6))
plt.scatter(x, y, c=z_normalized, cmap='gray', s=0.5)
plt.axis('off')
plt.axis('equal')

# Guardar imagen
output_path = "projected_2d.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"✅ Imagen proyectada guardada en: {output_path}")
