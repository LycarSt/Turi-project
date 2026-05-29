import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class PCDVisualizer:
    """
    Visualizador de mapas de ocupación 2D y segmentación 3D del LiDAR
    utilizando Matplotlib nativo y compatible con Python estándar (py).
    """
    
    @staticmethod
    def plot_3d_segmentation(processor, ground_mask, obs_xyz, labels, person_candidates, save_path=None, show=False):
        """
        Dibuja la segmentación 3D completa de la nube de puntos:
        - Suelo Seguro: Verde claro (downsampled para rendimiento).
        - Obstáculos agrupados: Colores por clúster.
        - Candidatos a Personas: Cajas contenedoras destacadas en rojo.
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. Graficar el Suelo Seguro (Zonas Seguras)
        # Hacemos submuestreo de 1 cada 5 puntos de suelo para acelerar la renderización 3D
        ground_pts = processor.xyz[ground_mask]
        if len(ground_pts) > 0:
            step = max(1, len(ground_pts) // 3000) # Límite de ~3000 puntos para visualización fluida
            ground_view = ground_pts[::step]
            ax.scatter(ground_view[:, 0], ground_view[:, 1], ground_view[:, 2], 
                       c='#a2d1a9', s=2, alpha=0.4, label='Suelo Seguro (Walkable)')
            
        # 2. Graficar Obstáculos y Clústeres
        if len(obs_xyz) > 0:
            # Puntos de ruido (etiqueta -1) en gris oscuro
            noise_pts = obs_xyz[labels == -1]
            if len(noise_pts) > 0:
                ax.scatter(noise_pts[:, 0], noise_pts[:, 1], noise_pts[:, 2], 
                           c='#555555', s=4, alpha=0.3, label='Ruido/Obstáculo General')
                
            # Clústeres reales en colores
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
                
            cmap = plt.get_cmap('tab20')
            for i, label in enumerate(unique_labels):
                cluster_pts = obs_xyz[labels == label]
                
                # Verificar si este clúster es un humano
                is_person = any(c['label'] == label for c in person_candidates)
                color = '#ff3333' if is_person else cmap(i % 20)
                marker_size = 12 if is_person else 6
                label_text = f'Objeto {label}' if not is_person else f'ALERTA: Persona {label}'
                
                ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], cluster_pts[:, 2], 
                           color=color, s=marker_size, alpha=0.8, label=label_text)
                
                # Dibujar un cubo/caja contenedora para las personas detectadas
                if is_person:
                    min_b = cluster_pts.min(axis=0)
                    max_b = cluster_pts.max(axis=0)
                    # Añadir texto destacado en el centro de la persona
                    center = (min_b + max_b) / 2.0
                    ax.text(center[0], center[1], max_b[2] + 0.3, "⚠️ SURVIVOR!", 
                            color='red', fontsize=12, fontweight='bold', 
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.3'))
        
        ax.set_title('Mapeo de Zonas Seguras 3D & Detección de Personas (Unitree Go1)', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (Frente - Metros)', fontsize=10)
        ax.set_ylabel('Y (Lados - Metros)', fontsize=10)
        ax.set_zlabel('Z (Altura - Metros)', fontsize=10)
        ax.set_xlim(-6, 4)
        ax.set_ylim(-4, 14)
        ax.set_zlim(-1.0, 2.0)
        ax.legend(loc='upper right', markerscale=2, fontsize=8)
        
        # Ajustar ángulo de cámara inicial de alta estética
        ax.view_init(elev=28, azim=-145)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"Visualización 3D guardada con éxito en {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_2d_occupancy_grid(grid, x_bins, y_bins, person_candidates=None, save_path=None, show=False):
        """
        Dibuja un mapa de ocupación 2D limpio y estético (Grid Map) apto para navegación:
        - Suelo Seguro (Celdas Libres): Blanco o Gris muy claro.
        - Obstáculos (Celdas Ocupadas): Negro.
        - Desconocido (Celdas no mapeadas): Gris oscuro.
        - Personas: Alertas tipo sonar en rojo.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Rotar la rejilla para alinearla visualmente con la orientación clásica del robot (X adelante, Y a los lados)
        # imshow espera el origen arriba a la izquierda por defecto, lo ajustamos
        grid_display = grid.T[::-1, :] # Traspuesta para alinear ejes
        
        # Mapear colores de alta calidad:
        # -1 (Desconocido) -> #2e2e2e (Gris oscuro)
        # 0 (Libre) -> #e8f5e9 (Verde claro ultra limpio)
        # 1 (Obstáculo) -> #b71c1c (Rojo oscuro) o #000000 (Negro)
        # Creamos una colormap personalizada
        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(['#2e2e2e', '#f5f5f5', '#2c3e50']) # Desconocido, Libre, Obstáculo
        
        # Graficar la rejilla de ocupación
        im = ax.imshow(grid_display, extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]], 
                       cmap=custom_cmap, vmin=-1, vmax=1)
        
        # Overlay de la posición del robot (siempre el origen 0, 0 en su propio frame de LiDAR)
        ax.plot(0, 0, marker='>', color='#ff9800', markersize=12, label='Posición Unitree Go1')
        # Círculo de seguridad interna alrededor del perro robot (radio de giro seguro de ~0.4m)
        safety_radius = Circle((0, 0), 0.4, color='#ff9800', fill=False, linestyle='--', linewidth=1.5)
        ax.add_patch(safety_radius)
        
        # Overlay de las personas detectadas en la rejilla 2D (Alertas Sonar de Rescate)
        if person_candidates:
            for c in person_candidates:
                px, py = c['center'][0], c['center'][1]
                # Graficar marcador de alerta en el mapa
                ax.plot(px, py, marker='*', color='#ff3333', markersize=14, label='SOBREVIVIENTE DETECTADO')
                
                # Círculo de radar de advertencia en rojo
                sonar = Circle((px, py), 0.6, color='#ff3333', fill=False, linestyle=':', linewidth=2, alpha=0.7)
                ax.add_patch(sonar)
                
                # Etiqueta de la persona
                ax.text(px + 0.15, py + 0.15, f"HUMAN_{c['label']}", color='#ff3333', 
                        fontweight='bold', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
        
        ax.set_title('Rejilla de Navegabilidad 2D (Zonas Seguras & Obstáculos)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Eje X - Frente del Robot (Metros)', fontsize=12)
        ax.set_ylabel('Eje Y - Lateral del Robot (Metros)', fontsize=12)
        ax.grid(True, which='both', color='#555555', linestyle='-', linewidth=0.3, alpha=0.5)
        
        # Leyendas personalizadas
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#f5f5f5', edgecolor='gray', label='Zona Segura (Walkable)'),
            Patch(facecolor='#2c3e50', edgecolor='gray', label='Obstáculo (Evitar Colisión)'),
            Patch(facecolor='#2e2e2e', edgecolor='gray', label='Zona Desconocida (Por Explorar)'),
            ax.plot([], [], marker='>', color='#ff9800', linestyle='None', markersize=10, label='Robot (Origen)')[0]
        ]
        
        if person_candidates:
            legend_elements.append(ax.plot([], [], marker='*', color='#ff3333', linestyle='None', markersize=10, label='Persona Detetada')[0])
            
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"Mapa 2D de navegación guardado con éxito en {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
