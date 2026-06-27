import cv2 as cv
import mediapipe as mp
from time import sleep, time
from ultralytics import YOLO

# About 35 ms between each frame
# Up in the y-axis is negative, down is positive

# Global Variables to change for the funsies whenever
HAND_TASK_PATH = "./hand_landmarker.task"
CAMERA_ID = 0
CONFIDENCE_REQ = 0.5
CURRENT_FRAME = None
THRESHOLD = 1 * pow(10, -7)

# Code Variables.
CORNERS = []
FINGERTIP_POS = {
    "Left": {},
    "Right": {},
}
TIMESTAMPS = {
    "Left": {},
    "Right": {},
}

def annotateHands(rgbImage, detectionResult):
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
    global FINGERTIP_POS

    # ---- Updates Locations and timestamps for each fingertip
    if result.hand_landmarks:
        for i, hand in enumerate(result.handedness):
            handedness = hand[0].category_name
            for lmId, landmark in enumerate(result.hand_world_landmarks[i]):
                # Only gathers fingertip data
                if lmId % 4 != 0 or lmId == 0:
                    continue
                
                # If a location list for the fingertip or timestamp has not been created yet, add one.
                if FINGERTIP_POS[handedness].get(lmId, -1) == -1:
                    FINGERTIP_POS[handedness][lmId] = []
                    TIMESTAMPS[handedness][lmId] = []

                if len(FINGERTIP_POS[handedness][lmId]) >= 15:
                    FINGERTIP_POS[handedness][lmId].pop(0)
                    FINGERTIP_POS[handedness][lmId].pop(0)

                FINGERTIP_POS[handedness][lmId].append(landmark.y)
                TIMESTAMPS[handedness][lmId].append(timestamp_ms)

                # ---- Key Press Logic, VERY SKETCHY PROB NEED TO CHANGE LATER ----
                if len(FINGERTIP_POS[handedness][lmId]) > 4 and len(TIMESTAMPS[handedness][lmId]) > 4:
                    firstVelocity = (FINGERTIP_POS[handedness][lmId][-2] - FINGERTIP_POS[handedness][lmId][-3]) / ((TIMESTAMPS[handedness][lmId][-2] - TIMESTAMPS[handedness][lmId][-3]) * 1000)
                    secondVelocity = (FINGERTIP_POS[handedness][lmId][-1] - FINGERTIP_POS[handedness][lmId][-2]) / ((timestamp_ms - TIMESTAMPS[handedness][lmId][-2]) * 1000)

                    if abs(secondVelocity - firstVelocity) > THRESHOLD and firstVelocity < 0 and secondVelocity > 0:
                        print(f"Key Press for {handedness} hand, tip {lmId} detected at {timestamp_ms}")
                # ---- End VERY SKETCHY Key press logic ----
                
    # ---- End Fingertip update logix ----

    DrawnFrame = output_image.numpy_view().copy() # Gets an rgb image to annotate.
    DrawnFrame = annotateHands(DrawnFrame, result)
    DrawnFrame = cv.cvtColor(DrawnFrame, cv.COLOR_RGB2BGR)

    CURRENT_FRAME = DrawnFrame

def main():
    global CORNERS

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    cam = cv.VideoCapture(CAMERA_ID, cv.CAP_AVFOUNDATION)
    model = YOLO("best.pt")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_TASK_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=visualizeResult,
        num_hands=2,
    )

    startTime = time()
    started = False
    while time() - startTime < 5:
        started, frame = cam.read()
        if not started:
            print("Warming Up, frame not found")
        else:
            break
        sleep(0.1)

    if not started:
        print("Camera could not start")
        cam.release()

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            frameExists, frame = cam.read()
            if not frameExists:
                print("Camera Frame not Found.")
                break
                
            # Attempts Annotation for keyboard
            results = model.track(frame, persist=True, conf=CONFIDENCE_REQ, verbose=False)
            for result in results:
                if result.obb is not None:
                    for corners in result.obb.xyxyxyxy:
                        pts = corners.cpu().numpy().reshape((-1, 1, 2)).astype(int)
                        CORNERS = pts.tolist()
                        cv.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Annotation for hands
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            landmarker.detect_async(mp_image, int(time() * 1000))

            # If there is an annotated Frame display that, instead display the latest unannotated frame.
            if CURRENT_FRAME is not None:
                cv.imshow("Hand Result", CURRENT_FRAME)
            else:
                cv.imshow("Hand Result", frame)

            key = cv.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('c'):
                print("Keyboard Corners Locked")
                print(CORNERS)

    
    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)