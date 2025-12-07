from pydantic import BaseModel

from api.model.TaskStatus import TaskStatus


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: str
    message: str


class ClusterResult(BaseModel):
    cluster_id: int
    density: float
    smoothed_density: float
    size: int
    priority: str | None


class DetectionResult(BaseModel):
    saved_overlay: str | None = None   # Ahora siempre URL /images/{task_id}/overlay
    saved_heatmap: str | None = None   # URL /images/{task_id}/heatmap
    vehicles_detected: int
    tracks: int
    max_intensity: float
    clusters: int
    cluster_details: list[ClusterResult]
    alert: bool
    alert_reason: str | None


class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: str
    completed_at: str | None = None
    result: DetectionResult | None = None
    error: str | None = None
    original_filename: str | None = None


class PaginatedResponse(BaseModel):
    data: list[TaskResult]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool
