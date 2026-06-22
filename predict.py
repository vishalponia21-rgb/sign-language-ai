"""
predict.py — Standalone real-time ASL prediction (OpenCV window, no Streamlit).
Run: python3 predict.py
Press 'q' to quit, 's' to capture/save current letter.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import pickle
import numpy as np
import time

# ─────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────
print("🧠  Loading model…")
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
print("✅  model.pkl loaded")

# ─────────────────────────────────────────────
#  MediaPipe — VIDEO mode for real-time
# ─────────────────────────────────────────────
print("✋  Initialising MediaPipe…")
base_opts = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task")
opts = vision.HandLandmarkerOptions(
    base_options=base_opts,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(opts)
print("✅  MediaPipe ready\n")

# ─────────────────────────────────────────────
#  Camera
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌  Cannot open camera.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("📷  Camera started — show your hand!")
print("     Press 'q' to quit  |  's' to save screenshot\n")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

frame_idx   = 0
smooth_buf  = []
SMOOTH_N    = 3
fps_times   = []

try:
    while cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("❌  Frame read failed.")
            break

        frame = cv2.flip(frame, 1)          # mirror
        h, w  = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int(frame_idx * (1000.0 / 30.0))
        result = landmarker.detect_for_video(mp_img, ts_ms)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            # Draw skeleton
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 180, 220), 2)
            for x, y in pts:
                cv2.circle(frame, (x, y), 5,  (0, 255, 136), -1)
                cv2.circle(frame, (x, y), 8,  (255, 255, 255), 1)

            # Feature vector
            feats = []
            for lm in hand:
                feats.extend([round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)])

            pred = model.predict([feats])[0]
            conf = float(model.predict_proba([feats]).max() * 100)

            # Smoothing
            smooth_buf.append(pred)
            if len(smooth_buf) > SMOOTH_N:
                smooth_buf.pop(0)
            smoothed = pred if len(set(smooth_buf)) == 1 else "…"

            # Overlay
            colour = (0, 255, 136) if conf >= 75 else ((0, 180, 255) if conf >= 55 else (0, 80, 255))
            cv2.rectangle(frame, (0, 0), (340, 130), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, 0), (340, 130), colour, 2)
            cv2.putText(frame, f"{smoothed}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.2, colour, 5)
            cv2.putText(frame, f"Conf: {conf:.1f}%", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        else:
            cv2.putText(frame, "Show your hand...", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

        # FPS counter
        fps_times.append(time.time() - t0)
        if len(fps_times) > 30:
            fps_times.pop(0)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

        cv2.imshow("Sign Language AI — Press Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"  📸  Screenshot saved: {fname}")

        frame_idx += 1

finally:
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋  Prediction stopped. Goodbye!")