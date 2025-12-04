import cv2

class HeatmapVisualizer:
    @staticmethod
    def blend(frame, heatmap, alpha=0.45):
        return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
