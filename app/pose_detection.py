import mediapipe as mp
import cv2
from mediapipe.tasks import python as py
from mediapipe.tasks.python import vision as vs

from app.downloader import Downloader
from app.camera import Camera
from app.display import Display


class PoseDetection:
    def __init__(self):
        model_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
        self.model_path = Downloader().download_model(model_url)
        self.detector = None
        self.configure_options()

    def start(self):
        camera = Camera()
        display = Display()

        try:
            camera.start()
            print("Cámara iniciada correctamente")
        except RuntimeError as e:
            print(f"Error al iniciar la cámara: {e}")
            return

        print("Presiona ESC para salir")

        running = True
        while running:
            frame = camera.capture_frame()
            if frame is None:
                print("Error al capturar frame")
                break

            landmarks_result = self.landmark(frame)

            frame_with_landmarks = display.visualize_pose(
                frame, landmarks_result)

            running = display.display_video(frame_with_landmarks)

        camera.release()
        display.close_windows()
        print("Programa finalizado")

    def configure_options(self):
        base_options = py.BaseOptions(model_asset_path=self.model_path)

        options = vs.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = vs.PoseLandmarker.create_from_options(options)

    def landmark(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        detection_result = self.detector.detect(mp_image)
        return detection_result
