import numpy as np
import cv2 as cv
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


model_path = hf_hub_download(
    repo_id="rujutashashikanjoshi/yolo12-drone-detection-0205-100m",
    filename="best.pt"
)

model = YOLO(model_path)

def is_drone(roi, conf_thresh = 0.1):
    if roi is None or roi.size == 0:
        return False
    
    results = model(roi, verbose= False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])  
            
            if conf < conf_thresh:
                continue
            label = model.names[cls_id]
            if label.lower() == "drone":
                return True
    return False

def main():
    source = 'verification/drone.mp4'
    cap = cv.VideoCapture(source)

    fgbg = cv.createBackgroundSubtractorMOG2(
        history=500,        
        varThreshold=16,    
        detectShadows=True  
    )

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    i = 0
    while True:
        
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)

        _, fgmask_bin = cv.threshold(fgmask, 200, 255, cv.THRESH_BINARY)

        fgmask_clean = cv.morphologyEx(fgmask_bin, cv.MORPH_OPEN, kernel, iterations=1)
        fgmask_clean = cv.dilate(fgmask_clean, kernel, iterations=2)

        contours, _ = cv.findContours(
            fgmask_clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        roi = None
        min_area = 100
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv.boundingRect(cnt)
            h_img, w_img = frame.shape[:2]
            t= 20
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)
            roi = frame[y1:y2, x1:x2]
            
            if i % 5 == 0 and is_drone(roi):
                cv.putText(
                    frame,
                    "DRONE",
                    (x, y - 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv.LINE_AA,
                )
            
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(
                frame,
                "Object",
                (x, y - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv.LINE_AA,
            )
            
        if roi is not None:
            cv.imshow("ROI", roi)
        cv.imshow("Original with detections", frame)
        cv.imshow("Foreground mask (raw)", fgmask)
        cv.imshow("Foreground mask (cleaned)", fgmask_clean)

        key = cv.waitKey(30) & 0xFF
        if key == 27:
            break
        i += 1
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()