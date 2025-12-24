import numpy as np
import cv2 as cv
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt


model_path = hf_hub_download(
    repo_id="rujutashashikanjoshi/yolo12-drone-detection-0205-100m",
    filename="best.pt"
)

model = YOLO(model_path)

def create_kalman_filter(dt=1.0):
    kf = cv.KalmanFilter(4, 2)

    # State: [x, y, vx, vy]
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # Measurement: [x, y]
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-1  
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    return kf

def is_drone(roi, conf_thresh = 0.3): # Threshold is low due to poor pretrained model used (model is trained on google images)
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
    W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fgbg = cv.createBackgroundSubtractorMOG2(history=500,        
                                             varThreshold=16,    
                                             detectShadows=True
                                             )

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    
    i = 0
    STATE = 'Verifying...'
    
    kf = create_kalman_filter()
    kf_initialized = False
    
    meas_x, meas_y = [], []
    filt_xs, filt_ys = [], []
    pred_xs, pred_ys = [], []


    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_title("Drone 2D Trajectory (Image Plane)")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)   # inverted y-axis
    fig.show()

    meas_line, = ax.plot([], [], 'g.', label="Measurements", alpha=0.5)
    filt_line, = ax.plot([], [], 'r-', label="Kalman filtered")
    pred_line, = ax.plot([], [], 'b--', label="Kalman prediction", alpha=0.5)

    ax.legend(loc="upper right")
    ax.set_aspect('equal', adjustable='box')
    
    while True:

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read video")

        detected_measurement = None

        fgmask = fgbg.apply(frame)
        _, fgmask_bin = cv.threshold(fgmask, 200, 255, cv.THRESH_BINARY)
        fgmask_clean = cv.morphologyEx(fgmask_bin, cv.MORPH_OPEN, kernel, iterations=1)
        fgmask_clean = cv.dilate(fgmask_clean, kernel, iterations=2)

        contours, _ = cv.findContours(
            fgmask_clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        if kf_initialized:
            prediction = kf.predict()
            pred_x = int(prediction[0, 0])
            pred_y = int(prediction[1, 0])

            pred_xs.append(pred_x)
            pred_ys.append(pred_y)
            cv.circle(frame, (pred_x, pred_y), 4, (255, 0, 0), -1)

        best_cnt = None
        best_dist = np.inf

        for cnt in contours:
            if cv.contourArea(cnt) < 100:
                continue

            x, y, w, h = cv.boundingRect(cnt)
            cx, cy = x + w / 2, y + h / 2

            if kf_initialized:
                dist = np.hypot(cx - pred_x, cy - pred_y)
                if dist < best_dist:
                    best_dist = dist
                    best_cnt = (x, y, w, h, cx, cy)
            else:
                best_cnt = (x, y, w, h, cx, cy)
                break

        if best_cnt is not None:
            x, y, w, h, cx, cy = best_cnt
            roi = frame[y:y+h, x:x+w]

            if i % 5 == 0 and STATE == 'Verifying...' and is_drone(roi):
                STATE = 'DRONE'

            if STATE == 'DRONE':
                measurement = np.array([[np.float32(cx)], [np.float32(cy)]])

                if not kf_initialized:
                    kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                    kf_initialized = True
                else:
                    if best_dist < 50:
                        kf.correct(measurement)
                        meas_x.append(cx)
                        meas_y.append(cy)

            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # --- Draw filtered state ---
        if kf_initialized:
            fx = int(kf.statePost[0, 0])
            fy = int(kf.statePost[1, 0])
            vx = float(kf.statePost[2, 0])
            vy = float(kf.statePost[3, 0])


            filt_xs.append(fx)
            filt_ys.append(fy)

            cv.circle(frame, (fx, fy), 4, (0, 0, 255), -1)

        # --- Visualization ---
        cv.imshow("Original with detections", frame)
        cv.imshow("Foreground mask (cleaned)", fgmask_clean)

        # --- Plot ---
        meas_line.set_data(meas_x, meas_y)
        filt_line.set_data(filt_xs, filt_ys)
        pred_line.set_data(pred_xs, pred_ys)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()


        key = cv.waitKey(30) & 0xFF
        if key == 27:
            break

        i += 1

    
    plt.ioff()
    plt.show()
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()