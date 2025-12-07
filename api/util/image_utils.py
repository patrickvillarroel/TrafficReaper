from pathlib import Path

import cv2
import numpy as np
from fastapi import UploadFile

_MAX_DIMENSION = 4000  # Máximo ancho/alto
_COMPRESSION_QUALITY = 85
_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def compress_image_if_needed(image: np.ndarray, max_dimension: int = _MAX_DIMENSION) -> np.ndarray:
    """Comprime la imagen si excede las dimensiones máximas."""
    h, w = image.shape[:2]

    if h <= max_dimension and w <= max_dimension:
        return image

    # Calcular nuevo tamaño manteniendo aspect ratio
    if h > w:
        new_h = max_dimension
        new_w = int(w * (max_dimension / h))
    else:
        new_w = max_dimension
        new_h = int(h * (max_dimension / w))

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compress_image_to_bytes(image: np.ndarray, quality: int = _COMPRESSION_QUALITY) -> bytes:
    """Comprime imagen a byte JPEG."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', image, encode_param)
    return buffer.tobytes()


def validate_image_file(file: UploadFile) -> tuple[bool, str]:
    """Valida el archivo de imagen."""
    # Verificar extensión
    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return False, f"Extensión no permitida. Use: {', '.join(_ALLOWED_EXTENSIONS)}"

    # Verificar tamaño (se hace en el endpoint)
    return True, "OK"
