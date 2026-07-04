import cv2 as cv
from keyboardLayouts import LAPTOP_KEYS
import mediapipe as mp
import numpy as np
from time import sleep, time
from textblob import TextBlob
from ultralytics import YOLO

DETECTED_KEYBOARD = LAPTOP_KEYS

HAND_TASK_PATH = "./hand_landmarker.task"
KEYBOARD_MODEL_PATH = "best.pt"
CAMERA_ID = 0
CONFIDENCE_REQ = 0.65
THRESHOLD = 1 * pow(10, -4)
FINGERTIP_HISTORY_SIZE = 3
PRESS_COOLDOWN_MS = 150
SMOOTHING_ALPHA = 0.5
MANUAL_CORNER_LABELS = (
    "top-left",
    "top-right",
    "bottom-right",
    "bottom-left",
)
MANUAL_CORNER_SHORT_LABELS = ("TL", "TR", "BR", "BL")

class KeyboardTracker:
    def __init__(self, confidence_req, keyboard_layout):
        self.confidence_req = confidence_req
        self.keyboard_layout = keyboard_layout
        self.corners = None
        self.locked = False
        self.homography = None
        self.inverse_homography = None

    def detect(self, frame, model):
        if self.locked:
            self.draw_key_zones(frame)
            return frame

        results = model.track(
            frame,
            persist=True,
            conf=self.confidence_req,
            verbose=False,
        )

        for result in results:
            if result.obb is None:
                continue

            for corners in result.obb.xyxyxyxy:
                self.corners = self.extract_corners(corners)
                self.draw_corners(frame)

        return frame
    
    def lock(self):
        if not self.corners_are_valid(self.corners):
            return False

        self.locked = True
        self.homography = self.compute_homography(self.corners)
        self.inverse_homography = self.compute_inverse_homography(self.corners)
        return True

    def lock_from_points(self, points):
        if len(points) != 4:
            return False

        self.corners = np.array(points, dtype=np.float32).reshape(4, 2)
        return self.lock()

    def extract_corners(self, obb_corners):
        points = obb_corners.cpu().numpy().reshape(4, 2).astype(np.float32)
        return self.order_corners(points)

    def order_corners(self, points):
        ordered = np.zeros((4, 2), dtype=np.float32)

        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).reshape(4)

        ordered[0] = points[np.argmin(sums)]
        ordered[1] = points[np.argmin(diffs)]
        ordered[2] = points[np.argmax(sums)]
        ordered[3] = points[np.argmax(diffs)]

        return ordered

    def corners_are_valid(self, corners):
        if corners is None or corners.shape != (4, 2):
            return False

        if len(np.unique(corners, axis=0)) != 4:
            return False

        contour = corners.reshape((-1, 1, 2)).astype(np.float32)
        signed_area = cv.contourArea(contour, oriented=True)
        if signed_area < 1000:
            return False

        return cv.isContourConvex(contour)

    def compute_homography(self, corners):
        normalized_keyboard = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        return cv.getPerspectiveTransform(corners, normalized_keyboard)

    def compute_inverse_homography(self, corners):
        normalized_keyboard = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        return cv.getPerspectiveTransform(normalized_keyboard, corners)

    def draw_corners(self, frame):
        if self.corners is None:
            return

        points = self.corners.reshape((-1, 1, 2)).astype(int)
        cv.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    def draw_key_zones(self, frame):
        if self.inverse_homography is None:
            return frame

        self.draw_corners(frame)

        for key, dimensions in self.keyboard_layout.items():
            if key in ("KeyboardWidth", "KeyboardHeight"):
                continue

            key_corners = self.get_normalized_key_corners(dimensions)
            image_corners = self.map_keyboard_points_to_image(key_corners)
            if image_corners is None:
                continue

            points = image_corners.reshape((-1, 1, 2)).astype(int)
            cv.polylines(frame, [points], isClosed=True, color=(0, 255, 255), thickness=1)

            label_x, label_y = image_corners.mean(axis=0).astype(int)
            cv.putText(
                frame,
                key,
                (label_x - 6, label_y + 4),
                cv.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 255, 255),
                1,
                cv.LINE_AA,
            )

        return frame

    def get_normalized_key_corners(self, dimensions):
        x, y, w, h = dimensions
        keyboard_width = self.keyboard_layout["KeyboardWidth"]
        keyboard_height = self.keyboard_layout["KeyboardHeight"]

        return np.array([
            [x / keyboard_width, y / keyboard_height],
            [(x + w) / keyboard_width, y / keyboard_height],
            [(x + w) / keyboard_width, (y + h) / keyboard_height],
            [x / keyboard_width, (y + h) / keyboard_height],
        ], dtype=np.float32)

    def map_keyboard_points_to_image(self, keyboard_points):
        if self.inverse_homography is None:
            return None

        points = np.array([keyboard_points], dtype=np.float32)
        image_points = cv.perspectiveTransform(points, self.inverse_homography)
        return image_points[0]

    def map_image_point_to_keyboard(self, image_point):
        if self.homography is None:
            return None

        point = np.array([[image_point]], dtype=np.float32)
        keyboard_point = cv.perspectiveTransform(point, self.homography)
        return keyboard_point[0][0]


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
            # print(f"Key Press for {handedness} hand, tip {lm_id} detected at {timestamp_ms}")
            if keyboard_point is not None:
                # print(f"Keyboard point: {keyboard_point}")
                self.detect_key(keyboard_point)

    def is_in_cooldown(self, handedness, lm_id, timestamp_ms):
        last_press_timestamp = self.last_press_timestamps[handedness].get(lm_id)
        if last_press_timestamp is None:
            return False

        return timestamp_ms - last_press_timestamp < self.press_cooldown_ms

    def detect_key(self, keyboard_point):
        for i, key in enumerate(list(DETECTED_KEYBOARD)):
            if i != 0 and i != 1:
                x, y, w, h = self.normalize_key(DETECTED_KEYBOARD[key])
                if keyboard_point[0] >= x and keyboard_point[0] <= x + w and keyboard_point[1] >= y and keyboard_point[1] <= y + h:
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
        x, y, w, h = dimensions
        return x / DETECTED_KEYBOARD["KeyboardWidth"], y / DETECTED_KEYBOARD["KeyboardHeight"], w / DETECTED_KEYBOARD["KeyboardWidth"], h / DETECTED_KEYBOARD["KeyboardHeight"]


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


class ClearboardApp:
    def __init__(self):
        self.window_name = "Hand Result"
        self.hand_task_path = HAND_TASK_PATH
        self.keyboard_model_path = KEYBOARD_MODEL_PATH
        self.camera_id = CAMERA_ID
        self.confidence_req = CONFIDENCE_REQ
        self.threshold = THRESHOLD
        self.fingertip_history_size = FINGERTIP_HISTORY_SIZE
        self.press_cooldown_ms = PRESS_COOLDOWN_MS
        self.smoothing_alpha = SMOOTHING_ALPHA

        self.cam = None
        self.keyboard_model = None
        self.manual_corner_points = []
        self.keyboard = KeyboardTracker(self.confidence_req, DETECTED_KEYBOARD)
        self.hand_tracker = HandTracker(
            self.threshold,
            self.fingertip_history_size,
            self.keyboard,
            self.press_cooldown_ms,
            self.smoothing_alpha,
        )

    def run(self):
        try:
            self.setup()
            if not self.warm_up_camera():
                print("Camera could not start")
                return

            with self.create_hand_landmarker() as landmarker:
                self.frame_loop(landmarker)
        finally:
            self.cleanup()

    def setup(self):
        self.cam = self.create_camera()
        self.keyboard_model = self.create_keyboard_model()
        cv.namedWindow(self.window_name)
        cv.setMouseCallback(self.window_name, self.handle_mouse_click)

    def create_camera(self):
        return cv.VideoCapture(self.camera_id, cv.CAP_AVFOUNDATION)

    def create_keyboard_model(self):
        return YOLO(self.keyboard_model_path)

    def create_hand_landmarker_options(self):
        base_options = mp.tasks.BaseOptions
        hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions
        vision_running_mode = mp.tasks.vision.RunningMode

        return hand_landmarker_options(
            base_options=base_options(model_asset_path=self.hand_task_path),
            running_mode=vision_running_mode.LIVE_STREAM,
            result_callback=self.hand_tracker.handle_result,
            num_hands=2,
        )

    def create_hand_landmarker(self):
        hand_landmarker = mp.tasks.vision.HandLandmarker
        options = self.create_hand_landmarker_options()
        return hand_landmarker.create_from_options(options)

    def warm_up_camera(self):
        start_time = time()

        while time() - start_time < 5:
            started, _frame = self.cam.read()
            if started:
                return True

            print("Warming Up, frame not found")
            sleep(0.1)

        return False

    """
        Order:
            - Annotates Keyboard Overlay (If camera not locked)
            - Processes hand data with both hand overlay and stores data
            - Shows Frame
    """
    def frame_loop(self, landmarker):
        while True:
            frame_exists, frame = self.read_frame()
            if not frame_exists:
                print("Camera Frame not Found.")
                break

            frame = self.process_keyboard_detection(frame)
            self.process_hand_detection(frame, landmarker)
            self.show_frame(frame)

            if self.handle_keypress():
                break

    def read_frame(self):
        return self.cam.read()

    def process_keyboard_detection(self, frame):
        return self.keyboard.detect(frame, self.keyboard_model)

    def process_hand_detection(self, frame, landmarker):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        landmarker.detect_async(mp_image, int(time() * 1000))

    def show_frame(self, fallback_frame):
        if self.hand_tracker.current_frame is not None:
            frame = self.hand_tracker.current_frame
        else:
            frame = fallback_frame

        frame = frame.copy()
        self.draw_manual_corner_points(frame)
        cv.imshow(self.window_name, frame)

    def handle_mouse_click(self, event, x, y, _flags, _param):
        if event == cv.EVENT_RBUTTONDOWN:
            self.manual_corner_points = []
            print("Manual corner selection reset")
            return

        if event != cv.EVENT_LBUTTONDOWN:
            return

        if self.keyboard.locked:
            return

        if len(self.manual_corner_points) >= 4:
            self.manual_corner_points = []

        corner_label = MANUAL_CORNER_LABELS[len(self.manual_corner_points)]
        self.manual_corner_points.append((x, y))
        print(f"Manual {corner_label} corner selected: ({x}, {y})")

        if len(self.manual_corner_points) == 4:
            self.lock_keyboard_from_manual_points()

    def lock_keyboard_from_manual_points(self):
        if not self.keyboard.lock_from_points(self.manual_corner_points):
            print("Cannot lock keyboard: manual corners were invalid")
            print("Expected order: top-left, top-right, bottom-right, bottom-left")
            self.manual_corner_points = []
            return

        for corner in self.keyboard.corners:
            print(corner)
        print("Keyboard homography computed")

    def draw_manual_corner_points(self, frame):
        if self.keyboard.locked:
            return

        for index, point in enumerate(self.manual_corner_points):
            x, y = point
            cv.circle(frame, (x, y), 6, (0, 0, 255), -1)
            cv.putText(
                frame,
                MANUAL_CORNER_SHORT_LABELS[index],
                (x + 8, y - 8),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )

    def handle_keypress(self):
        key = cv.waitKey(1)

        if key & 0xFF == ord("q"):
            self.correctText(self.hand_tracker.text)
            return True

        if key & 0xFF == ord("c"):
            self.lock_keyboard()

        return False

    def lock_keyboard(self):
        if not self.keyboard.lock():
            print("Cannot lock keyboard: no valid corners detected")
            return

        print("Keyboard Corners Locked")
        for corner in self.keyboard.corners:
            print(corner)
        print("Keyboard homography computed")

    def cleanup(self):
        if self.cam is not None:
            self.cam.release()
        cv.destroyAllWindows()

    def correctText(self, text):
        if text:
            textBlob = TextBlob(text)
            print(textBlob.correct())
        else:
            print("No text detected")

def main():
    app = ClearboardApp()
    app.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
