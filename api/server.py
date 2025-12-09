import json
import logging.config

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
import numpy as np
import uuid
import cv2
import io

from starlette.requests import Request

from api.model.TaskStatus import TaskStatus
from api.model.models import TaskResponse, TaskResult, PaginatedResponse, DetectionResult
from api.util.image_utils import compress_image_to_bytes, validate_image_file, compress_image_if_needed
from api.util.json_utils import convert_paths_to_urls, load_task_from_json

# ==================== CONFIGURACIÓN ====================
_project_root = Path(__file__).parent.parent.resolve()
_STORAGE_PATH = _project_root / "storage"
_ALERT_DIR = _STORAGE_PATH / "alerts"
_SNAPSHOT_DIR = _STORAGE_PATH / "snapshots"
_OUTPUTS_DIR = _STORAGE_PATH / "outputs"

# Crear directorios si no existen
for _directory in [_ALERT_DIR, _SNAPSHOT_DIR, _OUTPUTS_DIR]:
    _directory.mkdir(parents=True, exist_ok=True)

# Configuración de imagen
_MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
_CONTENT_TYPE_IMAGE_JPEG = "image/jpeg"
_COMPRESSED_DESCRIPTION = "Retornar imagen comprimida"

# Mensajes de error
_TASK_NO_ENCONTRADA = "Task no encontrada"
_logger = logging.getLogger("API")
_logger.setLevel(logging.INFO)
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "default",
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["default"],
        "level": "INFO",
    },
})

# Cola de procesamiento (en producción usar Celery/Redis)
task_results: dict[str, TaskResult] = {}  # {task_id: result_dict}


def process_image_task(task_id: str, image_bytes: bytes):
    """Procesa la imagen en background directamente desde bytes."""
    try:
        from traffic_analyzer import analyze_image

        # Actualizar estado
        _logger.info(f"Processing task {task_id}")
        task_results[task_id].status = TaskStatus.PROCESSING

        # Convertir bytes a numpy array (sin guardar en disco)
        _logger.debug("Decoding image bytes")
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            _logger.error("Error decodificando imagen")
            raise ValueError("No se pudo decodificar la imagen")

        # Comprimir si es necesario
        _logger.debug("Compressing image if needed")
        frame = compress_image_if_needed(frame)

        # Analizar
        _logger.debug("Analyzing image with YOLO")
        result = analyze_image(frame, save_outputs=True, save_dir=_OUTPUTS_DIR)
        _logger.info(f"Analysis completed for task {task_id}")

        # Guardar resultado con task_id y timestamps
        result["task_id"] = task_id
        result["created_at"] = task_results[task_id].created_at
        result["completed_at"] = datetime.now().isoformat()
        result["original_filename"] = task_results[task_id].original_filename

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Guardar JSON del resultado
        result_path = _OUTPUTS_DIR / f"result_{task_id}_{timestamp}.json"
        _logger.info(f"Saving result to {result_path}")
        with io.open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        # Actualizar task_results
        actual = task_results[task_id]
        actual.status = TaskStatus.COMPLETED
        actual.completed_at = result["completed_at"]
        actual.result = DetectionResult(**result)
        task_results[task_id] = actual
        _logger.info(f"Task {task_id} completed successfully")

    except Exception as e:
        _logger.error(f"Error procesando task {task_id}", exc_info=e)
        actual = task_results[task_id]
        actual.status = TaskStatus.FAILED
        actual.completed_at = datetime.now().isoformat()
        actual.error = str(e)
        task_results[task_id] = actual


# ==================== API ====================
app = FastAPI(
    title="Traffic Reaper API",
    description="API para análisis de tráfico vehicular con YOLO",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def root():
    """Endpoint de health check."""
    return {
        "status": "online",
        "service": "Traffic Reaper API",
        "version": "1.0.0"
    }


@app.get("/model/info", tags=["Health"])
def get_model_info():
    """Obtiene información sobre el modelo YOLO."""
    try:
        import sys
        from pathlib import Path

        # Agregar el directorio raíz al path
        project_root = Path(__file__).parent.parent.resolve()
        if str(project_root) not in sys.path:
            _logger.warning(f"Agregando {project_root} a sys.path para cargar el modelo")
            sys.path.insert(0, str(project_root))

        from traffic_analyzer import get_model_info

        return get_model_info()
    except Exception as e:
        _logger.error("Error cargando modelo YOLO", exc_info=e)
        return {
            "error": str(e),
            "loaded": False
        }


# ==================== ENDPOINTS DE ANÁLISIS ====================
@app.post("/analyze", response_model=TaskResponse, tags=["Analysis"])
async def analyze_image_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., media_type="image/*", description="Imagen a analizar")
):
    """
    Encola una imagen para análisis.
    Retorna un task_id para consultar el resultado posteriormente.
    La imagen se procesa directamente en memoria sin guardar archivo temporal.
    """
    # Validar archivo
    is_valid, message = validate_image_file(file)
    if not is_valid:
        _logger.error(f"Invalid image file: {message}")
        raise HTTPException(status_code=400, detail=message)

    # Leer contenido en memoria
    contents = await file.read()
    if len(contents) > _MAX_IMAGE_SIZE:
        _logger.error(f"Image size exceeds limit of {_MAX_IMAGE_SIZE} bytes")
        raise HTTPException(
            status_code=400,
            detail=f"Imagen muy grande. Máximo: {_MAX_IMAGE_SIZE / 1024 / 1024} MB"
        )

    # Generar task_id
    task_id = str(uuid.uuid4())

    # Crear registro de tarea
    task_results[task_id] = TaskResult(
        task_id=task_id,
        status=TaskStatus.PENDING,
        created_at=datetime.now().isoformat(),
        original_filename=file.filename
    )

    _logger.info(f"Task {task_id} enqueue")
    # Agregar a background tasks (pasamos bytes directamente)
    background_tasks.add_task(process_image_task, task_id, contents)

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        created_at=task_results[task_id].created_at,
        message="Imagen encolada para procesamiento"
    )


@app.get("/tasks/{task_id}", response_model=TaskResult, tags=["Analysis"])
def get_task_result(task_id: str, request: Request):
    """
    Obtiene el resultado de una tarea de análisis.
    Busca primero en memoria, luego en disco.
    """
    # Buscar en memoria
    if task_id in task_results:
        task_data = task_results[task_id].model_copy(deep=True)
        _logger.info(f"Task {task_id} found in memory with status {task_data.status}")

        # Convertir paths a URLs si la tarea está completada
        if task_data.status == TaskStatus.COMPLETED and task_data.result:
            base_url = str(request.base_url).rstrip("/")
            task_data.result = convert_paths_to_urls(task_id, task_data.result, base_url)

        return task_data

    # Buscar en disco
    _logger.info(f"Task {task_id} not found in memory, trying disk")
    task_data = load_task_from_json(task_id, _OUTPUTS_DIR)
    if task_data:
        _logger.info(f"Task {task_id} FOUND in disk with status {task_data.status}, adding to memory for future requests")
        task_results[task_id] = task_data.model_copy(deep=True)
        # Convertir paths a URLs
        base_url = str(request.base_url).rstrip("/")
        task_data.result = convert_paths_to_urls(task_id, task_data.result, base_url)

        return task_data

    _logger.error(f"Task {task_id} not found in memory or disk")
    raise HTTPException(status_code=404, detail="Task no encontrada")


@app.get("/tasks", tags=["Analysis"])
def list_tasks(
    status: TaskStatus | None = None,
    page: int = Query(1, ge=1, description="Número de página"),
    page_size: int = Query(20, ge=1, le=100, description="Tamaño de página"),
    include_disk: bool = Query(True, description="Incluir tareas guardadas en disco"),
    request: Request = None
):
    """
    Lista todas las tareas con paginación, opcionalmente filtradas por estado.
    Combina tareas en memoria y en disco.
    """
    tasks = list(task_results.values())

    # Cargar tareas desde disco si se solicita
    if include_disk:
        _logger.info("Including disk tasks in list results")
        seen_task_ids = {t.task_id for t in tasks}

        # Buscar todos los JSON de resultados
        for json_file in _OUTPUTS_DIR.glob("result_*_*.json"):
            try:
                # Extraer task_id del nombre del archivo
                filename = json_file.stem  # result_task-id_timestamp
                parts = filename.split("_", 2)
                if len(parts) >= 2:
                    file_task_id = parts[1]

                    # Evitar duplicados
                    if file_task_id not in seen_task_ids:
                        task_data = load_task_from_json(file_task_id, _OUTPUTS_DIR)
                        if task_data:
                            _logger.debug(f"Adding task {file_task_id} from disk to memory for future requests")
                            task_results[file_task_id] = task_data.model_copy(deep=True)
                            tasks.append(task_data)
                            seen_task_ids.add(file_task_id)
            except Exception as e:
                print(f"Error loading task from {json_file}: {e}")
                continue

    # Filtrar por estado
    if status:
        tasks = [t for t in tasks if t.status == status]

    # Ordenar por fecha de creación (más reciente primero)
    tasks.sort(key=lambda x: x.created_at, reverse=True)

    # Calcular paginación
    total = len(tasks)
    total_pages = (total + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # Obtener tareas de la página actual
    page_tasks = tasks[start_idx:end_idx]
    result_list = []

    # Convertir paths a URLs en los resultados
    if request:
        base_url = str(request.base_url).rstrip("/")
        for task in page_tasks:
            task_copy = task.model_copy(deep=True)
            if task_copy.status == TaskStatus.COMPLETED and task_copy.result:
                task_copy.result = convert_paths_to_urls(task.task_id, task.result, base_url)
            result_list.append(task_copy)

    return PaginatedResponse(
        data=result_list,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
    )


# ==================== ENDPOINTS DE IMÁGENES ====================
def _get_result(task_id: str):
    # Buscar en memoria
    task_data = task_results.get(task_id)

    # Si no está en memoria, buscar en disco
    if not task_data:
        _logger.info(f"Task {task_id} not found in memory for image, trying disk")
        task_data = load_task_from_json(task_id, _OUTPUTS_DIR)
        if task_data:
            _logger.info(f"Task {task_id} FOUND in disk with status {task_data.status} for image, adding to memory for future requests")
            task_results[task_id] = task_data

    if not task_data:
        _logger.warning(f"Task {task_id} not found in memory or disk to load images result")
        raise HTTPException(status_code=404, detail=_TASK_NO_ENCONTRADA)

    if task_data.status != TaskStatus.COMPLETED:
        _logger.warning(f"Task {task_id} not completed yet and trying to access image results")
        raise HTTPException(status_code=400, detail="Task no completada aún")

    return task_data.result


def _deliver_image(path: str, compressed: bool):
    if not compressed:
        return FileResponse(path, media_type=_CONTENT_TYPE_IMAGE_JPEG)

    # Comprimir imagen
    image = cv2.imread(path)
    compressed_bytes = compress_image_to_bytes(image)

    return StreamingResponse(
        io.BytesIO(compressed_bytes),
        media_type=_CONTENT_TYPE_IMAGE_JPEG
    )


@app.get("/images/{task_id}/overlay", tags=["Images"])
def get_overlay_image(
    task_id: str,
    compressed: bool = Query(True, description=_COMPRESSED_DESCRIPTION)
):
    """Obtiene la imagen overlay (con detecciones) de una tarea."""
    result = _get_result(task_id)
    overlay_path = result.saved_overlay if result is not None else None

    if not overlay_path or not Path(overlay_path).exists():
        if overlay_path:
            _logger.error(f"Overlay image found: {overlay_path}, but is not a existing path in server")
        raise HTTPException(status_code=404, detail="Imagen overlay no encontrada")

    return _deliver_image(overlay_path, compressed)


@app.get("/images/{task_id}/heatmap", tags=["Images"])
def get_heatmap_image(
    task_id: str,
    compressed: bool = Query(True, description=_COMPRESSED_DESCRIPTION)
):
    """Obtiene el heatmap de una tarea."""
    result = _get_result(task_id)
    heat_path = result.saved_heatmap if result is not None else None

    if not heat_path or not Path(heat_path).exists():
        if heat_path:
            _logger.error(f"Heatmap image found: {heat_path}, but is not a existing path in server")
        raise HTTPException(status_code=404, detail="Heatmap no encontrado")

    return _deliver_image(heat_path, compressed)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
