import mediapipe as mp
import cv2
import numpy as np


class Display:
    def __init__(self):
        self.window_name = 'Video'
        self.pose_connections = mp.solutions.pose.POSE_CONNECTIONS

    def frame_to_mediapipe(self, frame):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    def visualize(self, frame, detection_result):
        if detection_result is None or not hasattr(detection_result, 'detections'):
            return frame

        for detection in detection_result.detections:
            self._create_rectangle(detection, frame)
            bbox = detection.bounding_box
            self._create_labels(detection, frame, bbox)

        return frame

    def _create_rectangle(self, detection, frame):
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

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

    def visualize_pose(self, frame, pose_result):
        if pose_result is None or not hasattr(pose_result, 'pose_landmarks'):
            return frame

        annotated_frame = frame.copy()

        if pose_result.pose_landmarks:
            for pose_landmarks in pose_result.pose_landmarks:
                self._draw_landmarks_and_connections(
                    annotated_frame, pose_landmarks)

        return annotated_frame

    def _draw_landmarks_and_connections(self, frame, pose_landmarks):
        h, w, _ = frame.shape
        landmarks_px = []

        for landmark in pose_landmarks:
            landmarks_px.append((int(landmark.x * w), int(landmark.y * h)))

        for connection in self.pose_connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks_px) and end_idx < len(landmarks_px):
                cv2.line(frame, landmarks_px[start_idx], landmarks_px[end_idx],
                         (255, 255, 255), 2)

    def display_video(self, frame):
        if frame is None:
            return False

        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != 27

    def close_windows(self):
        cv2.destroyAllWindows()
