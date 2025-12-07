# test_analyze.py

import cv2
from traffic_analyzer import analyze_image

if __name__ == "__main__":
    IMAGE_PATH = "data/images/UTP4.jpg"

    frame = cv2.imread(IMAGE_PATH)

    if frame is None:
        print("Error: no se pudo cargar la imagen, revisa la ruta.")
        exit()

    print("Analizando imagen...")
    result = analyze_image(frame, save_outputs=True)

    import json
    print(json.dumps(result, indent=4))
