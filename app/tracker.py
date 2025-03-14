import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as py
from mediapipe.tasks.python import vision as vs

from app.downloader import Downloader


class Tracker:
    def __init__(self):
        self.detector = None
        self.model_url = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite'
        self.model_path = None
        self.downloader = Downloader()
        self.model_path = self.downloader.download_model(self.model_url)
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
