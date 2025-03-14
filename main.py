from app.tracker import Tracker
from app.camera import Camera
from app.display import Display


def main():
    tracker = Tracker()
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

        mp_image = display.frame_to_mediapipe(frame)

        detection_result = tracker.detect(mp_image)

        frame_with_detections = display.visualize(frame, detection_result)

        running = display.display_video(frame_with_detections)

    camera.release()
    display.close_windows()
    print("Programa finalizado")


if __name__ == "__main__":
    main()
