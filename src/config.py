from keyboardLayouts import LAPTOP_KEYS


DETECTED_KEYBOARD = LAPTOP_KEYS

HAND_TASK_PATH = "src/hand_landmarker.task"
KEYBOARD_MODEL_PATH = "src/best.pt"
CAMERA_ID = 0
CONFIDENCE_REQ = 0.65
THRESHOLD = 5 * pow(10, -5)
FINGERTIP_HISTORY_SIZE = 3
KEY_HISTORY_SIZE = 8
PRESS_COOLDOWN_MS = 150
SMOOTHING_ALPHA = 0.5
MANUAL_CORNER_LABELS = (
    "top-left",
    "top-right",
    "bottom-right",
    "bottom-left",
)
MANUAL_CORNER_SHORT_LABELS = ("TL", "TR", "BR", "BL")
