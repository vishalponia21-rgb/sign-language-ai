"""
collect_data.py — Collect custom hand sign samples for any label.

Usage:
  python3 collect_data.py

How it works:
  1. Choose a label (e.g. 'A' or 'hello')
  2. Press SPACE when your hand is in position to start recording
  3. Records SAMPLES_PER_LABEL frames automatically, saving features to CSV
  4. CSV format: label, x0, y0, z0, x1, y1, z1, … (63 feature cols + 1 label)

The output CSV can be used to retrain or fine-tune the model.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import csv
import os
import time

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
SAMPLES_PER_LABEL = 100
OUTPUT_DIR        = "custom_data"
OUTPUT_CSV        = os.path.join(OUTPUT_DIR, "custom_dataset.csv")
HAND_MODEL_PATH   = "hand_landmarker.task"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  MediaPipe setup
# ─────────────────────────────────────────────
base_opts = mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH)
opts = vision.HandLandmarkerOptions(
    base_options=base_opts,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
)
landmarker = vision.HandLandmarker.create_from_options(opts)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

# ─────────────────────────────────────────────
#  CSV header
# ─────────────────────────────────────────────
header_written = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0

# ─────────────────────────────────────────────
#  Camera
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌  Cannot open camera.")
    exit(1)

print("\n🤟  Custom Data Collector — Sign Language AI")
print("=" * 46)
print(f"   Output CSV : {OUTPUT_CSV}")
print(f"   Samples    : {SAMPLES_PER_LABEL} per label")
print("=" * 46)

while True:
    label = input("\nEnter label to collect (e.g. A / B / hello / q to quit): ").strip()
    if label.lower() == "q":
        break
    if not label:
        continue

    print(f"\n  Collecting '{label}' — position your hand, then press SPACE in the window.")
    print("  Press ESC to skip this label.\n")

    collecting   = False
    count        = 0
    frame_idx    = 0
    rows_to_save = []

    while count < SAMPLES_PER_LABEL:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int(frame_idx * (1000.0 / 30.0))
        result = landmarker.detect_for_video(mp_img, ts_ms)

        overlay = frame.copy()

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            pts  = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for a, b in HAND_CONNECTIONS:
                cv2.line(overlay, pts[a], pts[b], (0, 180, 220), 2)
            for x, y in pts:
                cv2.circle(overlay, (x, y), 5, (0, 255, 136), -1)

            if collecting:
                feats = []
                for lm in hand:
                    feats.extend([round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)])
                rows_to_save.append([label] + feats)
                count += 1

                # Progress bar
                bar_w = int((count / SAMPLES_PER_LABEL) * (w - 40))
                cv2.rectangle(overlay, (20, h - 40), (20 + bar_w, h - 20), (0, 255, 136), -1)
                cv2.rectangle(overlay, (20, h - 40), (w - 20, h - 20), (100, 100, 100), 2)

        status = f"RECORDING '{label}'  {count}/{SAMPLES_PER_LABEL}" if collecting else f"Ready — Press SPACE to start  |  Label: {label}"
        col = (0, 255, 136) if collecting else (0, 200, 255)
        cv2.putText(overlay, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        cv2.imshow("Data Collector — Sign Language AI", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            collecting = True
        elif key == 27:          # ESC
            print("  ⏭  Skipped.")
            break

        frame_idx += 1

    if rows_to_save:
        with open(OUTPUT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            if not header_written:
                cols = ["label"] + [f"{c}{i}" for i in range(21) for c in ["x","y","z"]]
                writer.writerow(cols)
                header_written = True
            writer.writerows(rows_to_save)
        print(f"  ✅  Saved {len(rows_to_save)} samples for '{label}' → {OUTPUT_CSV}")

cv2.destroyAllWindows()
landmarker.close()
cap.release()
print("\n✅  Data collection complete.")
print(f"   CSV saved to: {OUTPUT_CSV}")