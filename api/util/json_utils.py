import json
from datetime import datetime
from pathlib import Path

from api.model.TaskStatus import TaskStatus
from api.model.models import TaskResult, DetectionResult


def convert_paths_to_urls(task_id: str, result: DetectionResult, base_url: str):
    """Convierte rutas de disco a URLs de la API."""
    result_copy = result.model_copy()

    # Convertir saved_overlay
    if result_copy.saved_overlay:
        overlay_path = Path(result_copy.saved_overlay)
        if overlay_path.exists():
            result_copy.saved_overlay = f"{base_url}/images/{task_id}/overlay"

    # Convertir saved_heatmap
    if result_copy.saved_heatmap:
        heatmap_path = Path(result_copy.saved_heatmap)
        if heatmap_path.exists():
            result_copy.saved_heatmap = f"{base_url}/images/{task_id}/heatmap"

    return result_copy


def load_task_from_json(task_id: str, output_dir: Path) -> TaskResult | None:
    """Intenta cargar el estado de una tarea desde JSON en disco."""
    # Buscar en el directorio de outputs
    pattern = f"result_{task_id}_*.json"
    files = list(output_dir.glob(pattern))

    if not files:
        return None

    # Tomar el más reciente
    latest_file = max(files, key=lambda f: f.stat().st_mtime)

    try:
        with open(latest_file, "r") as file:
            result = json.load(file)

        # Reconstruir el formato de task_results
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            created_at=datetime.fromtimestamp(latest_file.stat().st_ctime).isoformat(),
            completed_at=datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat(),
            result=DetectionResult(**result),
            original_filename=result.get("original_filename", None)
        )
    except Exception as e:
        print(f"Error loading task from JSON: {e}")
        return None


def get_all_tasks_from_disk(output_dir: Path) -> list[TaskResult]:
    """Recupera todas las tareas desde los JSONs guardados en disco."""
    tasks = []

    # Patrón: result_{task_id}_{timestamp}.json
    for json_file in output_dir.glob("result_*.json"):
        try:
            with open(json_file, "r") as f:
                result = json.load(f)

            task_id = result.get("task_id")
            if not task_id:
                # Intentar extraer del nombre del archivo
                parts = json_file.stem.split("_")
                if len(parts) >= 2:
                    task_id = parts[1]

            task_data = TaskResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                created_at=datetime.fromtimestamp(json_file.stat().st_ctime).isoformat(),
                completed_at=datetime.fromtimestamp(json_file.stat().st_mtime).isoformat(),
                result=DetectionResult(**result),
                original_filename=result.get("original_filename", None)
            )
            tasks.append(task_data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    return tasks
