import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv

# Setup MediaPipe
MODEL_PATH = "hand_landmarker.task"
base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

DATASET_PATH = "asl_dataset"
SAMPLES_PER_CLASS = 500  # 500 images per letter

data = []
labels = []

# Get all letter folders — skip 'nothing', 'space', 'del'
valid_labels = [chr(i) for i in range(65, 91)]  # A-Z
folders = [f for f in os.listdir(DATASET_PATH) if f.upper() in valid_labels]
folders.sort()

print(f"Found {len(folders)} letter folders")
print(f"Processing {SAMPLES_PER_CLASS} images per letter...\n")

for folder in folders:
    folder_path = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    images = os.listdir(folder_path)[:SAMPLES_PER_CLASS]
    count = 0

    for img_file in images:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = hand_landmarker.detect(mp_image)

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            row = []
            for lm in hand_landmarks:
                row += [round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)]
            data.append(row)
            labels.append(folder.upper())
            count += 1

    print(f"  {folder.upper()}: {count} samples processed")

hand_landmarker.close()

print(f"\nTotal samples: {len(data)}")
print("Training model...")

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model trained!")
print(f"🎯 Accuracy: {accuracy * 100:.2f}%")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved — model.pkl")