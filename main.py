import cv2 as cv
import mediapipe as mp
import numpy as np
from time import sleep, time
from ultralytics import YOLO


HAND_TASK_PATH = "./hand_landmarker.task"
KEYBOARD_MODEL_PATH = "best.pt"
CAMERA_ID = 0
CONFIDENCE_REQ = 0.65
THRESHOLD = 1 * pow(10, -7)
FINGERTIP_HISTORY_SIZE = 15

class KeyboardTracker:
    def __init__(self, confidence_req):
        self.confidence_req = confidence_req
        self.corners = None
        self.locked = False
        self.homography = None

    def detect(self, frame, model):
        if self.locked:
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
        if self.corners is None or self.corners.shape != (4, 2):
            return False

        self.locked = True
        self.homography = self.compute_homography(self.corners)
        return True

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

    def compute_homography(self, corners):
        normalized_keyboard = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        return cv.getPerspectiveTransform(corners, normalized_keyboard)

    def draw_corners(self, frame):
        if self.corners is None:
            return

        points = self.corners.reshape((-1, 1, 2)).astype(int)
        cv.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    def map_image_point_to_keyboard(self, image_point):
        if self.homography is None:
            return None

        point = np.array([[image_point]], dtype=np.float32)
        keyboard_point = cv.perspectiveTransform(point, self.homography)
        return keyboard_point[0][0]


class HandTracker:
    def __init__(self, threshold, history_size, keyboard_tracker):
        self.threshold = threshold
        self.history_size = history_size
        self.keyboard_tracker = keyboard_tracker
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

    def handle_result(self, result, output_image, timestamp_ms):
        image_data = output_image.numpy_view()
        image_height, image_width = image_data.shape[:2]

        if result.hand_world_landmarks and self.last_timestamp < timestamp_ms:
            self.last_timestamp = timestamp_ms
            self.update_fingertips(result, timestamp_ms, image_width, image_height)

        drawn_frame = image_data.copy()
        drawn_frame = self.annotate_hands(drawn_frame, result)
        drawn_frame = cv.cvtColor(drawn_frame, cv.COLOR_RGB2BGR)

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
        self.add_fingertip_sample(
            handedness,
            lm_id,
            world_landmark.y,
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

    def detect_keypress(self, handedness, lm_id, timestamp_ms, image_point, keyboard_point):
        positions = self.fingertip_pos[handedness][lm_id]
        timestamps = self.timestamps[handedness][lm_id]

        if len(positions) <= 4 or len(timestamps) <= 4:
            return

        first_velocity = self.compute_velocity(positions, timestamps, -3, -2)
        second_velocity = self.compute_velocity(positions, timestamps, -2, -1)

        velocity_change = abs(second_velocity - first_velocity)
        is_press_motion = first_velocity < 0 and second_velocity > 0

        if velocity_change > self.threshold and is_press_motion:
            print(f"Key Press for {handedness} hand, tip {lm_id} detected at {timestamp_ms}")
            print(f"Image point: {image_point}")
            if keyboard_point is not None:
                print(f"Keyboard point: {keyboard_point}")

    def compute_velocity(self, positions, timestamps, first_index, second_index):
        return (
            (positions[second_index] - positions[first_index])
            / ((timestamps[second_index] - timestamps[first_index]) * 1000)
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
        self.hand_task_path = HAND_TASK_PATH
        self.keyboard_model_path = KEYBOARD_MODEL_PATH
        self.camera_id = CAMERA_ID
        self.confidence_req = CONFIDENCE_REQ
        self.threshold = THRESHOLD
        self.fingertip_history_size = FINGERTIP_HISTORY_SIZE

        self.cam = None
        self.keyboard_model = None
        self.keyboard = KeyboardTracker(self.confidence_req)
        self.hand_tracker = HandTracker(
            self.threshold,
            self.fingertip_history_size,
            self.keyboard,
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
            cv.imshow("Hand Result", self.hand_tracker.current_frame)
        else:
            cv.imshow("Hand Result", fallback_frame)

    def handle_keypress(self):
        key = cv.waitKey(1)

        if key & 0xFF == ord("q"):
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


def main():
    app = ClearboardApp()
    app.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
