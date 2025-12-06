from fastapi import FastAPI, HTTPException
from pathlib import Path
import json

from starlette.responses import FileResponse

app = FastAPI()

BASE_DIR = Path("storage")
ALERT_DIR = BASE_DIR / "alerts"
SNAP_DIR = BASE_DIR / "snapshots"


def load_latest_json(directory: Path, prefix: str):
    """Devuelve el JSON más reciente según el prefijo."""
    files = sorted(
        directory.glob(f"{prefix}_*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )

    if not files:
        return None

    with open(files[0], "r") as f:
        return json.load(f)


# ---------------------------------------------------
# 🟥 Última alerta generada
# ---------------------------------------------------
@app.get("/alerts/latest")
def get_latest_alert():
    data = load_latest_json(ALERT_DIR, "alert")
    if not data:
        raise HTTPException(status_code=404, detail="No hay alertas todavía.")
    return data


# ---------------------------------------------------
# 🟦 Último snapshot generado
# ---------------------------------------------------
@app.get("/snapshots/latest")
def get_latest_snapshot():
    data = load_latest_json(SNAP_DIR, "snapshot")
    if not data:
        raise HTTPException(status_code=404, detail="No hay snapshots todavía.")
    return data

@app.get("/imagen/{timestamp}")
def get_imagen(timestamp: str):
    ruta = BASE_DIR / "outputs" / f"alert_{timestamp}.png"
    if not ruta.exists():
        raise HTTPException(status_code=404, detail="No existe la imagen.")
    return FileResponse(ruta, media_type="image/png")