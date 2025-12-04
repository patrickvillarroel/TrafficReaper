# heatmap/heatmap_builder.py
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter

class HeatmapBuilder:
    def __init__(self, width, height, decay=0.96):
        self.width = width
        self.height = height
        self.decay = decay
        self.heatmap = np.zeros((height, width), dtype=np.float32)
        self.points = []  # historial de puntos recientes (para clustering)
        self.max_points_history = 2000  # limita memoria

    def add_points(self, centroids, weight=4, radius=10):
        """
        centroids: list of (cx, cy)
        weight: value to add at the circle
        radius: radius for drawing small blob per detection
        """
        # decaimiento
        self.heatmap *= self.decay

        for (cx, cy) in centroids:
            # validar
            if not (0 <= cx < self.width and 0 <= cy < self.height):
                continue
            cv2.circle(self.heatmap, (int(cx), int(cy)), radius, weight, -1)
            self.points.append([cx, cy])

        # mantener history limitado
        if len(self.points) > self.max_points_history:
            # quitar oldest
            excess = len(self.points) - self.max_points_history
            self.points = self.points[excess:]

    def _cluster_points(self, eps=40, min_samples=5):
        if len(self.points) < min_samples:
            return []
        pts = np.array(self.points)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
        labels = clustering.labels_
        clusters = []
        for label in set(labels):
            if label == -1:
                continue
            mask = (labels == label)
            cluster_pts = pts[mask]
            clusters.append(cluster_pts)
        return clusters

    def get_clusters(self, eps=40, min_samples=5):
        return self._cluster_points(eps=eps, min_samples=min_samples)

    def get_max_intensity(self):
        return float(self.heatmap.max())

    def get_heatmap_image(self, cluster_boost=True):
        temp = self.heatmap.copy()

        if cluster_boost:
            clusters = self._cluster_points()
            for cluster in clusters:
                cx = int(cluster[:,0].mean())
                cy = int(cluster[:,1].mean())
                # boost cluster area
                val = float(temp.max() * 0.6 + 40)
                cv2.circle(temp, (cx, cy), 60, val, -1)

        smooth = gaussian_filter(temp, sigma=25)
        norm = cv2.normalize(smooth, None, 0, 255, cv2.NORM_MINMAX)
        img_u8 = norm.astype(np.uint8)
        heat_img = cv2.applyColorMap(img_u8, cv2.COLORMAP_JET)
        return heat_img

    def reset(self):
        self.heatmap.fill(0)
        self.points.clear()
