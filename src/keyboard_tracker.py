import cv2 as cv
import numpy as np

"""
    - This class's purpose is to keep track of keyboard detection and annotation
    - Corner clicking and calibration
        - Ordering Corners, calculating homography etc.
    - Also in charge of mapping key presses to the location on keyboard
"""
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
