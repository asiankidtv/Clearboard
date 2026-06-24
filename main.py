import cv2 as cv
import mediapipe as mp
import numpy
from time import time

HAND_TASK_PATH = "./hand_landmarker.task"
CAMERA_ID = 0
CURRENT_FRAME = None

def annotate_hands(rgbImage, detectionResult):
    # Objects that google uses for this annotation.
    mp_hands = mp.tasks.vision.HandLandmarksConnections
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles

    hand_landmarks_list = detectionResult.hand_landmarks
    handedness_list = detectionResult.handedness

    # Annotates on every hand it can get in scence.
    for i in range(len(handedness_list)):
        handLandmarks = hand_landmarks_list[i]

        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            rgbImage,
            handLandmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    
    return rgbImage


# Create a hand landmarker instance with the live stream mode:
def visualizeResult(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
    global CURRENT_FRAME


    DrawnFrame = output_image.numpy_view().copy() # Gets an rgb image to annotate.
    DrawnFrame = annotate_hands(DrawnFrame, result)

    DrawnFrame = cv.cvtColor(DrawnFrame, cv.COLOR_RGB2BGR)
    

    CURRENT_FRAME = DrawnFrame

def main():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    cam = cv.VideoCapture(CAMERA_ID, cv.CAP_AVFOUNDATION)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_TASK_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=visualizeResult,
        num_hands=2,
    )
    
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            frameExists, frame = cam.read()
            if not frameExists:
                print("Camera Frame not Found.")
                break
                
            # Attempts Annotation
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            landmarker.detect_async(mp_image, int(time() * 1000))

            # If there is an annotated Frame display that, instead display the latest unannotated frame.
            if CURRENT_FRAME is not None:
                cv.imshow("Hand Result", CURRENT_FRAME)
            else:
                cv.imshow("Hand Result", frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    
    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()