# main_pro.py
import cv2
import time
import csv
from pathlib import Path
import torch

# importa tus módulos (adapta rutas si corresponde)
from ultralytics import YOLO

from tracker.centroid_tracker import CentroidTracker  # tu tracker simple
from tracker.kalman import KalmanFilter2D
from heatmap.heatmap_builder import HeatmapBuilder
from modules.alert_system import AlertSystem
from modules.density_engine import DensityEngine
from modules.smoothing import TemporalSmoother

alert_system = AlertSystem()
density_engine = DensityEngine()
smoother = TemporalSmoother(window=5)
cluster_id_counter = 0

# -----------------------------
# CONFIGURACION
# -----------------------------
VIDEO_SRC = "data/cinta.mp4"  # o RTMP url
MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.45

SAVE_DIR = Path("outputs")
SAVE_DIR.mkdir(exist_ok=True)

SNAPSHOT_INTERVAL = 10.0  # segundos
ALERT_INTENSITY_THRESHOLD = 120.0  # ajustar según escala (heat.get_max_intensity)
ALERT_CLUSTER_COUNT = 3  # si hay >= clusters -> alerta
ALERT_SAVE_PREFIX = SAVE_DIR / "alert"

# -----------------------------
# Inicializacion
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH)
model.to(device)
print("Modelo en", device)

cap = cv2.VideoCapture(VIDEO_SRC)
if not cap.isOpened():
    raise SystemExit("No se puede abrir fuente de video")

ret, frame = cap.read()
if not ret:
    raise SystemExit("No se puede leer primer frame")

H, W = frame.shape[:2]

# trackers & heatmap
tracker = CentroidTracker(max_disappeared=30, max_distance=120)
kf_map = dict()  # track_id -> KalmanFilter2D
kf_last_update = dict()  # track_id -> timestamp of last measurement
heat = HeatmapBuilder(width=W, height=H, decay=0.96)

last_snapshot = time.time()
frame_count = 0

# CSV log for alerts/snapshots
csvfile = SAVE_DIR / "events_log.csv"
if not csvfile.exists():
    with open(csvfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "event_type", "value", "image_path"])

print("Iniciando procesamiento... presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video/stream")
        break
    frame_count += 1
    t0 = time.time()

    # yolo inference
    results = model(frame, conf=CONF_THRESH, verbose=False)
    r = results[0]

    # gather vehicle detections
    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls = int(b.cls[0])
            if cls in [2,3,5,7]:
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                dets.append((x1,y1,x2,y2))


    # update centroid tracker -> returns dict id->centroid
    tracked = tracker.update(dets)

    # update centroid tracker -> returns dict id->centroid
    tracked = tracker.update(dets)
    centroids = list(tracked.values())

    # KALMAN per track: create/update/filter
    smoothed_centroids = {}
    now = time.time()
    for tid, centroid in tracked.items():
        cx, cy = int(centroid[0]), int(centroid[1])
        if tid not in kf_map:
            # init KF with zero velocity
            kf_map[tid] = KalmanFilter2D(x=cx, y=cy, vx=0, vy=0, dt=1.0, process_var=1.0, meas_var=20.0)
            kf_last_update[tid] = now
        else:
            # predict then update
            kf = kf_map[tid]
            kf.predict()
            kf.update([cx, cy])
            kf_last_update[tid] = now

        state = kf_map[tid].get_state()
        sx, sy = int(state[0]), int(state[1])
        smoothed_centroids[tid] = (sx, sy)

    # remove stale KF entries (tracks gone)
    stale_ids = []
    for tid in list(kf_map.keys()):
        if tid not in tracked:
            # if no observation for > 5s remove
            if now - kf_last_update.get(tid, 0) > 5.0:
                stale_ids.append(tid)
    for tid in stale_ids:
        del kf_map[tid]
        del kf_last_update[tid]

    # update heatmap using smoothed centroids (better stability)
    heat.add_points(list(smoothed_centroids.values()), weight=5, radius=10)

    # generate heatmap image (with clustering boost)
    heat_img = heat.get_heatmap_image(cluster_boost=True)

    # overlay
    overlay = cv2.addWeighted(frame, 0.6, heat_img, 0.4, 0)

    # draw smoothed centroids + ids
    for tid, (sx, sy) in smoothed_centroids.items():
        cv2.circle(overlay, (sx, sy), 5, (255,255,255), -1)
        cv2.putText(overlay, f"ID{tid}", (sx+6, sy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    # metrics & display
    fps = 1.0 / (time.time() - t0 + 1e-9)
    num_tracked = len(smoothed_centroids)
    max_intensity = heat.get_max_intensity()
    clusters = heat.get_clusters()

    ###########################################
    # SISTEMA DE DENSIDAD + PRIORIDADES + JSON
    ###########################################
    for cluster_idx, cluster_points in enumerate(clusters):
        density, cluster_size = density_engine.evaluate_cluster(cluster_points)
        smoothed_density = smoother.push(density)
        speed_var = 0.1

        priority = alert_system.classify_priority(
            density=smoothed_density,
            cluster_size=cluster_size,
            speed_variance=speed_var
        )

        if priority:
            alert_system.push_alert(
                cluster_id=cluster_idx,
                priority=priority,
                info={
                    "raw_density": density,
                    "smoothed_density": float(smoothed_density),
                    "cluster_size": float(cluster_size),
                    "speed_variance": float(speed_var),
                    "frame": frame_count
                }
            )


    ###########################################
    #   SNAPSHOT AUTOMÁTICO (cada X segundos)
    ###########################################
    if time.time() - last_snapshot >= SNAPSHOT_INTERVAL:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_path = SAVE_DIR / f"snapshot_{timestamp}.png"
        heat_path = SAVE_DIR / f"snapshot_heat_{timestamp}.png"

        cv2.imwrite(str(img_path), overlay)
        cv2.imwrite(str(heat_path), heat_img)

        # Log en CSV
        with open(csvfile, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, "snapshot", num_tracked, str(img_path)])

        print("📸 Snapshot guardado:", img_path)
        last_snapshot = time.time()

    ###########################################
    #   ALERTAS POR INTENSIDAD / CLUSTERS
    ###########################################
    alert_triggered = False

    if max_intensity >= ALERT_INTENSITY_THRESHOLD:
        alert_triggered = True
        reason = f"High intensity: {max_intensity:.1f}"

    if len(clusters) >= ALERT_CLUSTER_COUNT:
        alert_triggered = True
        reason = f"Clusters: {len(clusters)}"

    if alert_triggered:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        with open(csvfile, "a", newline="") as f:
            writer = csv.writer(f)


    text = f"Tracks: {num_tracked}  FPS: {fps:.1f}  MaxI: {max_intensity:.1f}  Clusters: {len(clusters)}"
    cv2.putText(overlay, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Heatmap PRO - KF + Clusters", overlay)
    key = cv2.waitKey(1) & 0xFF

    # condición 2: cantidad de clusters grandes
    if len(clusters) >= ALERT_CLUSTER_COUNT:
        alert_triggered = True
        reason = f"Clusters: {len(clusters)}"

    if alert_triggered:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        with open(csvfile, "a", newline="") as f:
            writer = csv.writer(f)

        # opcional: reset parcial del heatmap si quieres evitar alertas continuas
        # heat.reset()

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()