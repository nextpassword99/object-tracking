import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as py
from mediapipe.tasks.python import vision as vs


class Tracker:
    def __init__(self):
        self.detector = None

    def _configure_options(self):
        base_options = py.BaseOptions(
            model_asset_path='efficientdet_lite0.tflite')
        options = vs.ObjectDetectorOptions(
            base_options, max_results=10, score_threshold=0.5)
        self.detector = vs.ObjectDetector.create_from_options(options)

    def detect(self, frame):
        result = self.detector.detect(mp_image)
        return result

    def download_model(self):
        model_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
        if not os.path.exists(self.model_path):
            print("Descargando modelo...")
            try:
                response = requests.get(model_url)
                response.raise_for_status()

                with open(self.model_path, "wb") as f:
                    f.write(response.content)

                print(f"Modelo descargado: {self.model_path}")

            except requests.exceptions.RequestException as e:
                print(f"Error al descargar el modelo: {e}")
