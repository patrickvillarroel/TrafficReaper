import json
from pathlib import Path
import time

class JSONWriter:
    def __init__(self, base_dir="storage"):
        self.base = Path(base_dir)
        self.alert_dir = self.base / "alerts"
        self.snap_dir = self.base / "snapshots"

        self.alert_dir.mkdir(parents=True, exist_ok=True)
        self.snap_dir.mkdir(parents=True, exist_ok=True)

    def save_alert(self, data: dict) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self.alert_dir / f"alert_{ts}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        return path

    def save_snapshot(self, data: dict) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self.snap_dir / f"snapshot_{ts}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        return path
