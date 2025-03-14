import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as py
from mediapipe.tasks.python import vision as vs
import os
import requests


class Tracker:
    def __init__(self, model_path='data/models/efficientdet_lite0.tflite'):
        self.detector = None
        self.model_path = model_path
        self.download_model()
        self.configure_options()

    def configure_options(self):
        base_options = py.BaseOptions(model_asset_path=self.model_path)
        options = vs.ObjectDetectorOptions(
            base_options=base_options,
            max_results=10,
            score_threshold=0.5)
        self.detector = vs.ObjectDetector.create_from_options(options)

    def detect(self, frame):
        if not isinstance(frame, mp.Image):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        else:
            mp_image = frame

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
