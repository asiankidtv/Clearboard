import cv2 as cv
import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(
        self,
        threshold,
        history_size,
        keyboard_tracker,
        press_cooldown_ms,
        smoothing_alpha,
    ):
        self.text = ""

        self.threshold = threshold
        self.history_size = history_size
        self.keyboard_tracker = keyboard_tracker
        self.press_cooldown_ms = press_cooldown_ms
        self.smoothing_alpha = smoothing_alpha
        self.current_frame = None
        self.last_timestamp = 0
        self.fingertip_pos = {
            "Left": {},
            "Right": {},
        }
        self.fingertip_image_pos = {
            "Left": {},
            "Right": {},
        }
        self.fingertip_keyboard_pos = {
            "Left": {},
            "Right": {},
        }
        self.timestamps = {
            "Left": {},
            "Right": {},
        }
        self.last_press_timestamps = {
            "Left": {},
            "Right": {},
        }
        self.smoothed_fingertip_y = {
            "Left": {},
            "Right": {},
        }

    def handle_result(self, result, output_image, timestamp_ms):
        image_data = output_image.numpy_view()
        image_height, image_width = image_data.shape[:2]

        if result.hand_world_landmarks and self.last_timestamp < timestamp_ms:
            self.last_timestamp = timestamp_ms
            self.update_fingertips(result, timestamp_ms, image_width, image_height)

        drawn_frame = image_data.copy()
        drawn_frame = self.annotate_hands(drawn_frame, result)
        drawn_frame = cv.cvtColor(drawn_frame, cv.COLOR_RGB2BGR)
        drawn_frame = self.keyboard_tracker.draw_key_zones(drawn_frame)

        self.current_frame = drawn_frame

    def update_fingertips(self, result, timestamp_ms, image_width, image_height):
        for hand_index, hand in enumerate(result.handedness):
            handedness = hand[0].category_name

            for lm_id, landmark in enumerate(result.hand_world_landmarks[hand_index]):
                if not self.is_fingertip(lm_id):
                    continue

                image_landmark = result.hand_landmarks[hand_index][lm_id]
                self.update_fingertip(
                    handedness,
                    lm_id,
                    landmark,
                    image_landmark,
                    timestamp_ms,
                    image_width,
                    image_height,
                )

    def update_fingertip(
        self,
        handedness,
        lm_id,
        world_landmark,
        image_landmark,
        timestamp_ms,
        image_width,
        image_height,
    ):
        image_point = self.image_landmark_to_point(
            image_landmark,
            image_width,
            image_height,
        )
        keyboard_point = self.keyboard_tracker.map_image_point_to_keyboard(image_point)

        self.fingertip_image_pos[handedness][lm_id] = image_point
        self.fingertip_keyboard_pos[handedness][lm_id] = keyboard_point

        self.ensure_fingertip_history(handedness, lm_id)
        smoothed_y = self.smooth_fingertip_y(
            handedness,
            lm_id,
            world_landmark.y,
        )
        self.add_fingertip_sample(
            handedness,
            lm_id,
            smoothed_y,
            timestamp_ms,
        )
        self.detect_keypress(handedness, lm_id, timestamp_ms, image_point, keyboard_point)

    def is_fingertip(self, lm_id):
        return lm_id != 0 and lm_id % 4 == 0

    def ensure_fingertip_history(self, handedness, lm_id):
        if self.fingertip_pos[handedness].get(lm_id, -1) == -1:
            self.fingertip_pos[handedness][lm_id] = []
            self.timestamps[handedness][lm_id] = []

    def add_fingertip_sample(self, handedness, lm_id, y_value, timestamp_ms):
        positions = self.fingertip_pos[handedness][lm_id]
        timestamps = self.timestamps[handedness][lm_id]

        if len(positions) >= self.history_size:
            positions.pop(0)

        if len(timestamps) >= self.history_size:
            timestamps.pop(0)

        positions.append(y_value)

        if timestamp_ms not in timestamps:
            timestamps.append(timestamp_ms)

    def smooth_fingertip_y(self, handedness, lm_id, y_value):
        previous_y = self.smoothed_fingertip_y[handedness].get(lm_id)
        if previous_y is None:
            smoothed_y = y_value
        else:
            smoothed_y = (
                self.smoothing_alpha * y_value
                + (1 - self.smoothing_alpha) * previous_y
            )

        self.smoothed_fingertip_y[handedness][lm_id] = smoothed_y
        return smoothed_y

    def detect_keypress(self, handedness, lm_id, timestamp_ms, image_point, keyboard_point):
        positions = self.fingertip_pos[handedness][lm_id]
        timestamps = self.timestamps[handedness][lm_id]

        if len(positions) < 3 or len(timestamps) < 3:
            return

        first_velocity = self.compute_velocity(positions, timestamps, -3, -2)
        second_velocity = self.compute_velocity(positions, timestamps, -2, -1)

        velocity_change = abs(second_velocity - first_velocity)
        is_press_motion = first_velocity < 0 and second_velocity > 0

        if velocity_change > self.threshold and is_press_motion:
            if self.is_in_cooldown(handedness, lm_id, timestamp_ms):
                return

            self.last_press_timestamps[handedness][lm_id] = timestamp_ms
            if keyboard_point is not None:
                self.detect_key(keyboard_point)

    def is_in_cooldown(self, handedness, lm_id, timestamp_ms):
        last_press_timestamp = self.last_press_timestamps[handedness].get(lm_id)
        if last_press_timestamp is None:
            return False

        return timestamp_ms - last_press_timestamp < self.press_cooldown_ms

    def detect_key(self, keyboard_point):
        keyboard_layout = self.keyboard_tracker.keyboard_layout
        for key, dimensions in keyboard_layout.items():
            if key in ("KeyboardWidth", "KeyboardHeight"):
                continue

            x, y, w, h = self.normalize_key(dimensions)
            is_inside_x = keyboard_point[0] >= x and keyboard_point[0] <= x + w
            is_inside_y = keyboard_point[1] >= y and keyboard_point[1] <= y + h
            if is_inside_x and is_inside_y:
                self.parse_key(key)

    def parse_key(self, key):
        print(f"{key} Detected")
        if key == "Space":
            self.text = self.text + " "
        elif key == "Backspace":
            if len(self.text) >= 2:
                self.text = self.text[0:-1]
        elif len(key) == 1:
            self.text = self.text + key
        else:
            return

    def normalize_key(self, dimensions):
        keyboard_layout = self.keyboard_tracker.keyboard_layout
        x, y, w, h = dimensions
        return (
            x / keyboard_layout["KeyboardWidth"],
            y / keyboard_layout["KeyboardHeight"],
            w / keyboard_layout["KeyboardWidth"],
            h / keyboard_layout["KeyboardHeight"],
        )

    def compute_velocity(self, positions, timestamps, first_index, second_index):
        return (
            (positions[second_index] - positions[first_index])
            / ((timestamps[second_index] - timestamps[first_index]))
        )

    def image_landmark_to_point(self, landmark, image_width, image_height):
        return np.array([
            landmark.x * image_width,
            landmark.y * image_height,
        ], dtype=np.float32)

    def annotate_hands(self, rgb_image, detection_result):
        mp_hands = mp.tasks.vision.HandLandmarksConnections
        mp_drawing = mp.tasks.vision.drawing_utils
        mp_drawing_styles = mp.tasks.vision.drawing_styles

        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness

        for hand_index in range(len(handedness_list)):
            hand_landmarks = hand_landmarks_list[hand_index]
            mp_drawing.draw_landmarks(
                rgb_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        return rgb_image
