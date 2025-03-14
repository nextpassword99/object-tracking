import cv2


class Camera:
    def __init__(self, camera_id=0, width=640, height=480):
        self.cap = None
        self.camera_id = camera_id
        self.width = width
        self.height = height

    def start(self):
        self._initialize_camera()
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("No se pudo inicializar la cámara")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True

    def _initialize_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)

    def capture_frame(self):
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("La cámara no está inicializada")

        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.release()
