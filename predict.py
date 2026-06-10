import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import pickle
import numpy as np

# Load the trained model
print("Loading model...")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model loaded successfully!")

# Setup MediaPipe
MODEL_PATH = "hand_landmarker.task"
base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)

# Start camera
cap = cv2.VideoCapture(0)
print("Camera started! Show your hand...")
print("Press 'q' to quit")

frame_idx = 0
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Camera not found!")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * (1000.0 / 30.0))
        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            h, w, _ = frame.shape
            hand_landmarks = result.hand_landmarks[0]

            # Draw 21 points
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Prepare data for prediction
            row = []
            for lm in hand_landmarks:
                row += [round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)]

            # Predict the sign
            prediction = model.predict([row])[0]
            confidence = model.predict_proba([row]).max() * 100

            # Show prediction on screen
            cv2.putText(frame, f"Sign: {prediction}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        else:
            cv2.putText(frame, "Show your hand...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Sign Language AI - Live Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1
finally:
    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Prediction stopped!")