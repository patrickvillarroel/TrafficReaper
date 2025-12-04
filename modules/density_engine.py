# modules/density_engine.py
import numpy as np

class DensityEngine:

    def __init__(self, min_points=4):
        self.min_points = min_points

    def evaluate_cluster(self, cluster_points):
        density = len(cluster_points)

        # Calcular tamaño físico del cluster (radio aprox)
        if len(cluster_points) > 1:
            pts = np.array(cluster_points)
            center = pts.mean(axis=0)
            dist = np.linalg.norm(pts - center, axis=1)
            cluster_size = dist.max()
        else:
            cluster_size = 1.0

        return density, cluster_size
