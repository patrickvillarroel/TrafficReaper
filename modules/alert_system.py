# modules/alert_system.py
import json
from datetime import datetime
from pathlib import Path

class AlertSystem:
    def __init__(self, out_file="output/alerts.json"):
        self.out_path = Path(out_file)
        self.alerts = []

    def classify_priority(self, density, cluster_size, speed_variance):
        """
        density: número de vehículos en el cluster
        cluster_size: tamaño físico del cluster
        speed_variance: cuánta movilidad hay (0 = quieto)
        """

        # Regla 1: Congestión fuerte
        if density >= 12 and speed_variance < 0.2:
            return "P1"   # Máxima prioridad

        # Regla 2: Congestión moderada
        if density >= 7:
            return "P2"

        # Regla 3: Inicio de aglomeración
        if density >= 4:
            return "P3"

        return None  # No genera alerta

    def push_alert(self, cluster_id, priority, info):
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "cluster_id": cluster_id,
            "priority": priority,
            "details": info
        }
        self.alerts.append(alert)

        # Guardarlo en JSON
        self.save()

    def save(self):
        self.out_path.parent.mkdir(exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(self.alerts, f, indent=4)
