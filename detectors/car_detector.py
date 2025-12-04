import cv2
from ultralytics import YOLO

class CarDetector:
    def __init__(self, model_path="yolov8s.pt", conf=0.45):
        self.model = YOLO(model_path)
        self.conf = conf
        self.classes_allowed = {2, 3, 5, 7}  # car, motorcycle, bus, truck

    def detect(self, frame):
        results = self.model(frame, conf=self.conf)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls)
            if cls not in self.classes_allowed:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append((int(x1), int(y1), int(x2), int(y2)))

        return detections
