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

    def _create_labels(detection, frame, bbox):
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (10 + bbox.origin_x,
                         10 + 10 + bbox.origin_y)
        cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 0, 0), 1)

    def display_video(self, frame):
        cv2.imshow('Video', frame)
