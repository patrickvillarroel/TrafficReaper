from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=2, max_distance=5):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        if len(detections) == 0:
            remove = []
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    remove.append(obj_id)
            for rid in remove:
                self.deregister(rid)
            return self.objects

        input_centroids = np.zeros((len(detections), 2), dtype="int")

        for (i, (x1, y1, x2, y2)) in enumerate(detections):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids[i] = (cx, cy)

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = dist.cdist(np.array(object_centroids), input_centroids)

        # REEMPLAZO: Usar Algoritmo Húngaro para asignación óptima global
        rows, cols = linear_sum_assignment(D)

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            # El algoritmo húngaro garantiza unicidad, pero mantenemos la verificación por seguridad
            if row in used_rows or col in used_cols:
                continue

            if D[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            self.objects[obj_id] = input_centroids[col]
            self.disappeared[obj_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        unused_cols = set(range(0, input_centroids.shape[0])).difference(used_cols)
        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects
