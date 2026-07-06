import cv2 as cv
import mediapipe as mp
import numpy as np
import textwrap
from time import sleep, time
from textblob import TextBlob
from ultralytics import YOLO

from config import (
    CAMERA_ID,
    CONFIDENCE_REQ,
    DETECTED_KEYBOARD,
    FINGERTIP_HISTORY_SIZE,
    KEY_HISTORY_SIZE,
    KEYBOARD_POINT_SMOOTHING_ALPHA,
    HAND_TASK_PATH,
    KEYBOARD_MODEL_PATH,
    MANUAL_CORNER_LABELS,
    MANUAL_CORNER_SHORT_LABELS,
    MIN_PRESS_TRAVEL,
    PRESS_COOLDOWN_MS,
    SMOOTHING_ALPHA,
    THRESHOLD,
)
from hand_tracker import HandTracker
from keyboard_tracker import KeyboardTracker


class ClearboardApp:
    def __init__(self):
        self.window_name = "Hand Result"
        self.text_window_name = "Typed Text"
        self.hand_task_path = HAND_TASK_PATH
        self.keyboard_model_path = KEYBOARD_MODEL_PATH
        self.camera_id = CAMERA_ID
        self.confidence_req = CONFIDENCE_REQ
        self.threshold = THRESHOLD
        self.fingertip_history_size = FINGERTIP_HISTORY_SIZE
        self.key_history_size = KEY_HISTORY_SIZE
        self.min_press_travel = MIN_PRESS_TRAVEL
        self.press_cooldown_ms = PRESS_COOLDOWN_MS
        self.smoothing_alpha = SMOOTHING_ALPHA
        self.keyboard_point_smoothing_alpha = KEYBOARD_POINT_SMOOTHING_ALPHA

        self.cam = None
        self.keyboard_model = None
        self.manual_corner_points = []
        self.keyboard = KeyboardTracker(self.confidence_req, DETECTED_KEYBOARD)
        self.hand_tracker = HandTracker(
            self.threshold,
            self.fingertip_history_size,
            self.keyboard,
            self.press_cooldown_ms,
            self.min_press_travel,
            self.smoothing_alpha,
            self.keyboard_point_smoothing_alpha,
            self.key_history_size
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
        cv.namedWindow(self.text_window_name)
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

    def frame_loop(self, landmarker):
        while True:
            frame_exists, frame = self.read_frame()
            if not frame_exists:
                print("Camera Frame not Found.")
                break

            # frame = self.process_keyboard_detection(frame)
            self.process_hand_detection(frame, landmarker)
            self.show_frame(frame)
            self.show_text_window(self.hand_tracker.text)

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
            corrected_text = self.correctText(self.hand_tracker.text)
            self.show_text_window(corrected_text, final=True)
            cv.waitKey(0)
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
            corrected_text = str(textBlob.correct())
            print(corrected_text)
            return corrected_text
        else:
            print("No text detected")
            return ""

    def show_text_window(self, text, final=False):
        frame = self.create_text_window_frame(text, final)
        cv.imshow(self.text_window_name, frame)

    def create_text_window_frame(self, text, final=False):
        frame = np.full((360, 720, 3), 245, dtype=np.uint8)
        title = "Final Corrected Message" if final else "Typed Text"
        body = text if text else ""

        cv.putText(
            frame,
            title,
            (24, 42),
            cv.FONT_HERSHEY_SIMPLEX,
            0.85,
            (40, 40, 40),
            2,
            cv.LINE_AA,
        )

        if not body:
            body = "No text detected" if final else "Waiting for keypresses..."

        y = 88
        for line in textwrap.wrap(body, width=58):
            cv.putText(
                frame,
                line,
                (24, y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                (25, 25, 25),
                1,
                cv.LINE_AA,
            )
            y += 32

            if y > 300:
                cv.putText(
                    frame,
                    "...",
                    (24, y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (25, 25, 25),
                    1,
                    cv.LINE_AA,
                )
                break

        if final:
            cv.putText(
                frame,
                "Press any key to close",
                (24, 336),
                cv.FONT_HERSHEY_SIMPLEX,
                0.55,
                (90, 90, 90),
                1,
                cv.LINE_AA,
            )

        return frame
