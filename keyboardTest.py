from ultralytics import YOLO
import cv2 as cv
from time import sleep

CONFIDENCE_REQ = 0.4
CAMERA_ID = 0

def main():
    model = YOLO('best.pt')
    cam = cv.VideoCapture(CAMERA_ID, cv.CAP_AVFOUNDATION)
    sleep(3)

    while True:
        frameExists, frame = cam.read()
        if not frameExists:
            print("No Frame Found")
            break

        results = model.track(frame, persist=True, conf=CONFIDENCE_REQ)
        for result in results:
            if result.obb is not None:
                for corners in result.obb.xyxyxyxy:
                    pts = corners.cpu().numpy().reshape((-1, 1, 2)).astype(int)
                    
                    cv.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        cv.imshow("Detection", frame)

        if cv.waitKey(20) & 0xFF==ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()