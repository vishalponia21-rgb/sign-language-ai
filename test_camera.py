import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

cap = cv2.VideoCapture(0)

print("Camera chal raha hai! Haath dikhao...")
print("Band karne ke liye 'q' dabao")

base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)

frame_idx = 0
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Camera nahi mila!")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * (1000.0 / 30.0))
        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in result.hand_landmarks:
                # 21 points draw karo
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            cv2.putText(frame, "Haath detect hua! 21 points!", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Haath dikhao...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Sign Language AI - Day 2", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1
finally:
    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Day 2 Complete! 🎉")