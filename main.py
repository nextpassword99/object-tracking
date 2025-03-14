from app.tracker import Tracker
from app.pose_detection import PoseDetection


def main():
    print("""
          Usar:
          1 ) Detección de objetos
          2 ) Detección de pose
          """)
    option = input("Selecciona: ")

    if option == "1":
        tracker = Tracker()
        tracker.start()
    elif option == "2":
        pose = PoseDetection()
        pose.start()


if __name__ == "__main__":
    main()
