# traffic_analyzer.py

import cv2
import time
import torch
from pathlib import Path
from ultralytics import YOLO

from tracker.centroid_tracker import CentroidTracker
from tracker.kalman import KalmanFilter2D
from heatmap.heatmap_builder import HeatmapBuilder
from modules.alert_system import AlertSystem
from modules.density_engine import DensityEngine
from modules.smoothing import TemporalSmoother
from modules.json_writer import JSONWriter

# Configuracion Global
MODEL_PATH = "yolov8x.pt"
CONF_THRESH = 0.45
ALERT_INTENSITY_THRESHOLD = 120.0
ALERT_CLUSTER_COUNT = 15
SAVE_DIR = Path("outputs")
SAVE_DIR.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
model.to(device)

# Módulos persistentes entre llamadas
alert_system = AlertSystem()
density_engine = DensityEngine()
smoother = TemporalSmoother(window=5)
writer = JSONWriter()

# Trackers globales entre llamadas
global_tracker = CentroidTracker(max_disappeared=10, max_distance=200)
global_kf_map = {}
global_kf_last = {}

heatmap_instance = None
H = W = None


def analyze_image(frame, save_outputs=False):
    """
    Procesa una sola imagen.
    frame: numpy array (BGR) leído por cv2.imdecode o cv2.imread
    save_outputs: si True guarda overlay y heatmap

    Return: dict con resultados listos para API JSON
    """

    global heatmap_instance, H, W, global_kf_map, global_kf_last

    h, w = frame.shape[:2]

    if H != h or W != w:
        H, W = h, w
        heatmap_instance = HeatmapBuilder(width=W, height=H, decay=0.96)

    # Reinicia heatmap solo para esta imagen
    heatmap_instance.reset()

    results = model(frame, imgsz=4000, conf=CONF_THRESH)
    r = results[0]

    dets = []
    if r.boxes is not None:
        for b in r.boxes:
            cls = int(b.cls[0])
            if cls in [2]:  # vehículos
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                dets.append((x1, y1, x2, y2))

    # Tracker
    tracked = global_tracker.update(dets)
    smoothed_centroids = {}
    now = time.time()

    for tid, (cx, cy) in tracked.items():

        if tid not in global_kf_map:
            global_kf_map[tid] = KalmanFilter2D(cx, cy, 0, 0, dt=1.0)
            global_kf_last[tid] = now
        else:
            kf = global_kf_map[tid]
            kf.predict()
            kf.update([cx, cy])
            global_kf_last[tid] = now

        sx, sy = map(int, global_kf_map[tid].get_state()[:2])
        smoothed_centroids[tid] = (sx, sy)

    # Eliminar KF viejos (5s sin actualizar)
    for tid in list(global_kf_map.keys()):
        if tid not in tracked and now - global_kf_last.get(tid, 0) > 5:
            del global_kf_map[tid]
            del global_kf_last[tid]

    # Heatmap
    heatmap_instance.add_points(
        list(smoothed_centroids.values()),
        weight=40,
        radius=50
    )

    heat_img = heatmap_instance.get_heatmap_image(cluster_boost=True)
    overlay = cv2.addWeighted(frame, 0.6, heat_img, 0.4, 0)

    for tid, (sx, sy) in smoothed_centroids.items():
        cv2.circle(overlay, (sx, sy), 5, (255, 255, 255), -1)
        cv2.putText(
            overlay,
            f"ID{tid}",
            (sx + 8, sy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    max_intensity = heatmap_instance.get_max_intensity()
    clusters = heatmap_instance.get_clusters()

    # Evaluar cada cluster
    cluster_results = []

    for idx, cpts in enumerate(clusters):

        density, csize = density_engine.evaluate_cluster(cpts)
        smooth_density = smoother.push(density)

        priority = alert_system.classify_priority(
            density=smooth_density,
            cluster_size=csize,
            speed_variance=0.1
        )

        cluster_results.append({
            "cluster_id": idx,
            "density": float(density),
            "smoothed_density": float(smooth_density),
            "size": int(csize),
            "priority": priority
        })

    # Alertas
    alert_flag = (
        max_intensity >= ALERT_INTENSITY_THRESHOLD or
        len(clusters) >= ALERT_CLUSTER_COUNT
    )

    alert_reason = None
    if alert_flag:
        alert_reason = f"I={max_intensity:.1f}, clusters={len(clusters)}"

    # Guardar imagen
    saved_overlay = None
    saved_heat = None

    if save_outputs:
        ts = time.strftime("%Y%m%d_%H%M%S")
        saved_overlay = str(SAVE_DIR / f"overlay_{ts}.png")
        saved_heat = str(SAVE_DIR / f"heat_{ts}.png")

        cv2.imwrite(saved_overlay, overlay)
        cv2.imwrite(saved_heat, heat_img)

    # JSON de respuesta
    return {
        "vehicles_detected": len(dets),
        "tracks": len(smoothed_centroids),
        "max_intensity": float(max_intensity),
        "clusters": len(clusters),
        "cluster_details": cluster_results,
        "alert": alert_flag,
        "alert_reason": alert_reason,
        "saved_overlay": saved_overlay,
        "saved_heatmap": saved_heat
    }
