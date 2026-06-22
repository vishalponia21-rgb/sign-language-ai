"""
test_camera.py — Verify camera access and MediaPipe hand detection.
Run: python3 test_camera.py
Press 'q' to quit.
"""

import cv2
import sys

print("\n🔍  Testing Camera Access…\n")

# Try indices 0-3
camera_index = None
for idx in range(4):
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✅  Camera found at index {idx}")
            print(f"    Resolution: {frame.shape[1]}×{frame.shape[0]}")
            camera_index = idx
            break
        cap.release()
    else:
        cap.release()

if camera_index is None:
    print("❌  No camera found! Check permissions in System Preferences → Privacy → Camera.")
    sys.exit(1)

# Re-open at found index
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n📷  Live preview — Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌  Cannot read frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # Overlay guide box
    cx, cy = w // 2, h // 2
    box_size = min(w, h) // 3
    cv2.rectangle(
        frame,
        (cx - box_size, cy - box_size),
        (cx + box_size, cy + box_size),
        (0, 255, 136), 2,
    )
    cv2.putText(frame, "Hold your hand inside this box",
                (cx - box_size, cy - box_size - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 136), 2)
    cv2.putText(frame, f"Camera OK — {w}x{h} | Press Q to quit",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Camera Test — Sign Language AI", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅  Camera test complete.")