# test_analyze.py

import cv2
from main import analyze_image

IMAGE_PATH = "data/images/UTP4.jpg"

frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print("Error: no se pudo cargar la imagen, revisa la ruta.")
    exit()

print("Analizando imagen...")
result = analyze_image(frame, save_outputs=True)

import json
print(json.dumps(result, indent=4))
