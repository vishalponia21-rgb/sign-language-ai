import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import csv
import os
import signal
import sys

MODEL_PATH = "hand_landmarker.task"

# Dataset folder
os.makedirs("dataset", exist_ok=True)
CSV_FILE = "dataset/hand_data.csv"

# Create CSV if not exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header += [f'p{i}_x', f'p{i}_y', f'p{i}_z']
        writer.writerow(header)
    print("CSV file created!")

cap = cv2.VideoCapture(0)

base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)

current_label = None
count = 0
SAMPLES_PER_LABEL = 100

# Graceful shutdown on signals (SIGINT / SIGTERM)
terminate = False
def _signal_handler(signum, frame):
    global terminate
    print("\nReceived termination signal, exiting gracefully...")
    terminate = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

print("\n=== Data Collection — 26 Alphabets ===")
print("Press any letter key (A-Z) to start collecting")
print("Hold your hand sign for 100 samples")
print("Press 'q' to quit\n")

# Check existing data
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'r') as f:
        rows = list(csv.reader(f))
    existing = {}
    for row in rows[1:]:
        if row:
            existing[row[0]] = existing.get(row[0], 0) + 1
    if existing:
        print("Already collected:")
        for label, count_ex in sorted(existing.items()):
            print(f"  {label}: {count_ex} samples ✅")
    
    remaining = [chr(i) for i in range(65, 91) if chr(i) not in existing]
    print(f"\nRemaining: {' '.join(remaining)}\n")

frame_idx = 0
try:
    while cap.isOpened() and not terminate:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * (1000.0 / 30.0))
        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            h, w, _ = frame.shape
            hand_landmarks = result.hand_landmarks[0]

            # Draw points
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Save data if label selected
            if current_label and count < SAMPLES_PER_LABEL:
                row = [current_label]
                for lm in hand_landmarks:
                    row += [round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)]

                with open(CSV_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                count += 1
                cv2.putText(frame, f"Collecting {current_label}: {count}/100", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if count >= SAMPLES_PER_LABEL:
                    print(f"✅ '{current_label}' — 100 samples saved!")
                    current_label = None
                    count = 0

            elif current_label is None:
                cv2.putText(frame, "Press a letter key to collect", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        else:
            cv2.putText(frame, "Show your hand...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show current label big
        if current_label:
            cv2.putText(frame, current_label, (frame.shape[1]//2 - 30, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)

        cv2.imshow("Data Collection - A to Z", frame)

        key = cv2.waitKey(1) & 0xFF
        if terminate:
            break
        if key == ord('q'):
            break
        elif 97 <= key <= 122:  # a-z
            current_label = chr(key).upper()
            count = 0
            print(f"Collecting '{current_label}' — show your hand sign!")
        elif 65 <= key <= 90:  # A-Z
            current_label = chr(key)
            count = 0
            print(f"Collecting '{current_label}' — show your hand sign!")

        frame_idx += 1
finally:
    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection complete!")