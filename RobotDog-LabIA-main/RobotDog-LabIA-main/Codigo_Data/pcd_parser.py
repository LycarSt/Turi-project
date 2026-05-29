import numpy as np
import time

class PCDParser:
    """
    Clase especializada en el parseo y lectura de archivos de nube de puntos (.pcd)
    en formato binario de 32 bytes por punto, típico de escáneres de 16 canales.
    """
    
    @staticmethod
    def read_pcd(file_path):
        """
        Lee un archivo PCD binario y extrae sus campos de manera ultra-rápida.
        Filtra los valores NaN de las coordenadas espaciales.
        
        Args:
            file_path (str): Ruta al archivo .pcd
            
        Returns:
            dict: Diccionario con arrays de numpy para 'x', 'y', 'z', 'intensity', 'ring', 'timestamp'.
        """
        with open(file_path, 'rb') as f:
            content = f.read()
            
        # Encontrar el inicio de los datos binarios en la cabecera
        data_idx = content.find(b"DATA binary")
        if data_idx == -1:
            raise ValueError("El archivo no está en formato binario válido (.pcd DATA binary)")
            
        newline_pos = content.find(b"\n", data_idx)
        binary_start = newline_pos + 1
        
        # Estructura del punto (32 bytes):
        # FIELDS x y z _ intensity _ ring _ timestamp
        # SIZE   4 4 4 1 1         1 2    1 8
        # TYPE   F F F U U         U U    U F
        # COUNT  1 1 1 4 1         1 1    4 1
        dt = np.dtype([
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('_pad1', 'u1', 4),
            ('intensity', 'u1'),
            ('_pad2', 'u1'),
            ('ring', 'u2'),
            ('_pad3', 'u1', 4),
            ('timestamp', 'f8')
        ])
        
        # El barrido tiene exactamente 28800 puntos
        num_points = 28800
        itemsize = 32
        expected_size = num_points * itemsize
        
        binary_data = content[binary_start:binary_start + expected_size]
        points = np.frombuffer(binary_data, dtype=dt)
        
        # Filtrar valores NaN (puntos vacíos donde el LiDAR se perdió o no tuvo retorno)
        x = points['x']
        y = points['y']
        z = points['z']
        
        valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
        
        return {
            'x': x[valid_mask],
            'y': y[valid_mask],
            'z': z[valid_mask],
            'intensity': points['intensity'][valid_mask],
            'ring': points['ring'][valid_mask],
            'timestamp': points['timestamp'][valid_mask]
        }


class PCDProcessor:
    """
    Procesador geométrico y de Machine Learning 3D para la nube de puntos.
    Incluye RANSAC para suelo seguro, DBSCAN para objetos y Occupancy Grid 2D.
    """
    
    def __init__(self, points_dict):
        """
        Inicializa el procesador con los puntos válidos.
        """
        self.x = points_dict['x']
        self.y = points_dict['y']
        self.z = points_dict['z']
        self.intensity = points_dict['intensity']
        self.ring = points_dict['ring']
        self.timestamp = points_dict['timestamp']
        
        # Apilar en matriz N x 3 para procesamiento geométrico
        self.xyz = np.stack((self.x, self.y, self.z), axis=1)
        self.num_points = len(self.xyz)
        
    def segment_ground_ransac(self, distance_threshold=0.06, max_iterations=100, min_ground_z=-0.8, max_ground_z=-0.3):
        """
        Segmenta el suelo seguro (zona transitable) usando RANSAC optimizado para robótica.
        Restringe la búsqueda a Zs lógicas donde el suelo debería estar respecto a la altura del perro.
        
        Returns:
            ground_mask (ndarray): Máscara booleana de puntos correspondientes al suelo.
            obstacle_mask (ndarray): Máscara booleana de puntos correspondientes a obstáculos.
            plane_coeffs (tuple): Coeficientes (A, B, C, D) del plano del suelo.
        """
        t0 = time.time()
        best_inliers = np.array([], dtype=int)
        best_plane = (0, 0, 1, 0.6)  # Plano horizontal por defecto
        
        # Filtrar candidatos al suelo (reducir espacio de búsqueda para velocidad extrema)
        ground_candidates = np.where((self.z >= min_ground_z) & (self.z <= max_ground_z))[0]
        
        if len(ground_candidates) < 3:
            # Fallback en caso de que no haya suficientes puntos
            ground_mask = self.z < -0.4
            return ground_mask, ~ground_mask, best_plane
            
        for _ in range(max_iterations):
            # Seleccionar 3 puntos aleatorios candidatos
            sample_indices = np.random.choice(ground_candidates, 3, replace=False)
            p1, p2, p3 = self.xyz[sample_indices]
            
            # Calcular vectores coplanares y su producto cruzado (normal del plano)
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
                
            normal = normal / norm
            A, B, C = normal
            
            # Restricción robótica: El plano de suelo debe ser predominantemente horizontal.
            # La componente vertical C de la normal debe ser muy cercana a 1 (o -1)
            if abs(C) < 0.85:
                continue
                
            D = -np.dot(normal, p1)
            
            # Calcular distancias de todos los puntos de la nube al plano ajustado
            # dist = |Ax + By + Cz + D| (el denominador es 1 porque normal ya está normalizada)
            distances = np.abs(np.dot(self.xyz, normal) + D)
            inliers = np.where(distances < distance_threshold)[0]
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = (A, B, C, D)
                
        # Si no se encontró un buen plano, aplicar un umbral simple basado en altura
        if len(best_inliers) == 0:
            ground_mask = (self.z >= min_ground_z) & (self.z <= max_ground_z)
        else:
            ground_mask = np.zeros(self.num_points, dtype=bool)
            ground_mask[best_inliers] = True
            
        obstacle_mask = ~ground_mask
        
        # Filtro de seguridad adicional: Puntos extremadamente altos (techo) no son obstáculos navegables
        # Los guardamos como obstáculos pero no afectarán la pisada del perro
        
        print(f"RANSAC Suelo Seguro completado en {(time.time() - t0)*1000:.2f}ms. Inliers de suelo: {np.sum(ground_mask)} / {self.num_points}")
        return ground_mask, obstacle_mask, best_plane
        
    def cluster_obstacles_dbscan(self, obstacle_mask, eps=0.25, min_samples=12, max_obstacle_height=1.2):
        """
        Agrupa los obstáculos no transitables usando el algoritmo DBSCAN.
        Permite aislar mesas, sillas, paredes y personas en clusters individuales.
        
        Args:
            obstacle_mask (ndarray): Máscara booleana de los obstáculos.
            eps (float): Distancia de vecindad de DBSCAN (en metros).
            min_samples (int): Mínimo número de puntos para formar un grupo.
            max_obstacle_height (float): Altura máxima a considerar para obstáculos (evita el techo).
            
        Returns:
            obstacle_xyz (ndarray): Matriz de coordenadas N_obs x 3 de los puntos filtrados.
            labels (ndarray): Etiquetas de clústeres para cada punto de obstáculo (-1 es ruido).
            person_candidates (list): Lista de diccionarios con metadatos de clústeres candidatos a personas.
        """
        t0 = time.time()
        
        # Filtrar obstáculos para evitar computar el techo o puntos extremadamente lejanos irrelevantes
        obstacle_indices = np.where(obstacle_mask & (self.z < max_obstacle_height))[0]
        obs_xyz = self.xyz[obstacle_indices]
        
        if len(obs_xyz) == 0:
            return np.empty((0, 3)), np.array([]), []
            
        labels = np.array([])
        
        # Intentar clustering híbrido prioritario con Open3D
        try:
            import open3d as o3d
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(obs_xyz)
            # cluster_dbscan retorna una lista de etiquetas int vectorizada
            labels = np.array(pcd_o3d.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False))
            print("Clustering DBSCAN completado usando el motor ultra-rápido de Open3D.")
        except ImportError:
            # Fallback nativo estable a Scikit-Learn
            from sklearn.cluster import DBSCAN
            db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = db.fit_predict(obs_xyz)
            print("Clustering DBSCAN completado usando Scikit-Learn.")
            
        # Análisis de clústeres para encontrar candidatos a seres humanos (Rescate)
        person_candidates = []
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1) # Ignorar ruido
            
        for label in unique_labels:
            class_mask = (labels == label)
            cluster_pts = obs_xyz[class_mask]
            
            # Calcular caja contenedora alineada a los ejes (Axis-Aligned Bounding Box)
            min_bound = cluster_pts.min(axis=0)
            max_bound = cluster_pts.max(axis=0)
            dims = max_bound - min_bound # [Ancho_X, Largo_Y, Alto_Z]
            
            width_x, length_y, height_z = dims
            center = (min_bound + max_bound) / 2.0
            
            # Criterio de Búsqueda y Rescate para Humano (De pie, agachado o acostado):
            # - Altura física del clúster entre 0.3m (acostado) y 1.9m (de pie)
            # - Volumen y dimensiones lógicas (no es una pared masiva de 5 metros ni un punto pequeño)
            is_person = False
            
            # Caso 1: Persona de pie o sentada
            if 0.5 <= height_z <= 1.8 and width_x < 1.0 and length_y < 1.0:
                is_person = True
            # Caso 2: Persona acostada/atrapada en suelo (volumen concentrado pero bajo)
            elif height_z < 0.5 and (0.8 <= max(width_x, length_y) <= 2.0):
                is_person = True
                
            if is_person:
                person_candidates.append({
                    'label': int(label),
                    'center': center.tolist(),
                    'dimensions': dims.tolist(),
                    'num_points': int(np.sum(class_mask)),
                    'height': float(height_z)
                })
                
        print(f"Clustering completado en {(time.time() - t0)*1000:.2f}ms. Encontrados {len(unique_labels)} objetos y {len(person_candidates)} candidatos a personas.")
        return obs_xyz, labels, person_candidates
        
    def generate_occupancy_grid(self, ground_mask, obstacle_mask, resolution=0.15, max_obstacle_height=1.0):
        """
        Genera una rejilla de ocupación 2D (Occupancy Grid Map) proyectando los puntos 3D.
        Crucial para que el Unitree Go1 pueda trazar rutas sin chocar.
        
        Args:
            ground_mask (ndarray): Máscara de puntos del suelo seguro.
            obstacle_mask (ndarray): Máscara de puntos de obstáculos.
            resolution (float): Tamaño del pixel de la rejilla en metros (ej. 0.15m = 15cm).
            max_obstacle_height (float): Altura máxima de obstáculos para evitar colisiones corporales.
            
        Returns:
            grid (ndarray): Matriz 2D. Valores: 0 = Libre, 1 = Obstáculo, -1 = Desconocido.
            x_edges (ndarray): Bordes de las celdas en el eje X.
            y_edges (ndarray): Bordes de las celdas en el eje Y.
        """
        t0 = time.time()
        
        # Filtro de puntos del robot (descartar puntos muy cercanos a la base del robot/LiDAR para evitar auto-escanearse)
        dist_from_origin = np.linalg.norm(self.xyz[:, :2], axis=1)
        valid_points_mask = dist_from_origin > 0.2
        
        # Filtrar puntos de suelo y de obstáculos dentro de la altura de colisión del perro
        ground_pts = self.xyz[ground_mask & valid_points_mask]
        
        # Los obstáculos deben estar en la altura física del cuerpo del robot para ser colisiones reales (ej. no el techo)
        collision_obstacle_mask = obstacle_mask & valid_points_mask & (self.z < max_obstacle_height) & (self.z > -0.6)
        obstacle_pts = self.xyz[collision_obstacle_mask]
        
        # Definir los límites espaciales del mapa
        all_pts = np.vstack((ground_pts[:, :2], obstacle_pts[:, :2])) if len(obstacle_pts) > 0 else ground_pts[:, :2]
        
        min_x, min_y = all_pts.min(axis=0) - 0.5
        max_x, max_y = all_pts.max(axis=0) + 0.5
        
        # Crear los contenedores de las celdas
        x_bins = np.arange(min_x, max_x + resolution, resolution)
        y_bins = np.arange(min_y, max_y + resolution, resolution)
        
        # Crear grid inicializado como Desconocido (-1)
        grid = np.full((len(x_bins)-1, len(y_bins)-1), -1, dtype=int)
        
        # Rellenar celdas con puntos de suelo (0 = libre)
        if len(ground_pts) > 0:
            x_idx_g = np.digitize(ground_pts[:, 0], x_bins) - 1
            y_idx_g = np.digitize(ground_pts[:, 1], y_bins) - 1
            
            # Asegurar índices dentro de límites
            valid_g = (x_idx_g >= 0) & (x_idx_g < grid.shape[0]) & (y_idx_g >= 0) & (y_idx_g < grid.shape[1])
            grid[x_idx_g[valid_g], y_idx_g[valid_g]] = 0
            
        # Rellenar celdas con puntos de obstáculos (1 = ocupado, sobreescribe libre)
        if len(obstacle_pts) > 0:
            x_idx_o = np.digitize(obstacle_pts[:, 0], x_bins) - 1
            y_idx_o = np.digitize(obstacle_pts[:, 1], y_bins) - 1
            
            valid_o = (x_idx_o >= 0) & (x_idx_o < grid.shape[0]) & (y_idx_o >= 0) & (y_idx_o < grid.shape[1])
            grid[x_idx_o[valid_o], y_idx_o[valid_o]] = 1
            
        print(f"Mapa de Ocupación 2D generado en {(time.time() - t0)*1000:.2f}ms. Rejilla: {grid.shape[0]}x{grid.shape[1]} celdas.")
        return grid, x_bins, y_bins
