import mediapipe as mp
import cv2


class Display:
    def __init__(self):
        pass

    def frame_to_mediapipe(self, frame):
        return mp.Image.create_from_array(frame)

    def visualize(self, frame, detection_result):
        for detection in detection_result.detections:
            self._create_rectangle(detection, frame)

    def _create_rectangle(detection, frame):
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 3)

    def _create_labels(self, detection, frame, bbox):
        if not detection.categories:
            return

        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"

        text_location = (bbox.origin_x + 10, bbox.origin_y + 20)

        text_size = cv2.getTextSize(
            result_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(
            frame,
            (text_location[0] - 5, text_location[1] - text_size[1] - 5),
            (text_location[0] + text_size[0] + 5, text_location[1] + 5),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    def display_video(self, frame):
        if frame is None:
            return False

        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != 27

    def close_windows(self):
        cv2.destroyAllWindows()
