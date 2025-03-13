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
        result = self.detector.detect(frame)
