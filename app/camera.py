import cv2


class camera:
    def __init__(self):
        self.cap = None

    def start(self):
        self._initialize_camera()

    def _initialize_camera(self) -> cv2.VideoCapture:
        self.cap = cv2.VideoCapture(0)

    def _capture_frame(self):
        ret, frame = self.cap.read()
        return frame
