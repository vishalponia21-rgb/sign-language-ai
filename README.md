# Sign Language Recognition (ASL) — Computer Vision Project

This project performs real-time hand sign recognition using computer vision and deep learning. It uses MediaPipe to extract 3D hand landmarks (keypoints) from a webcam feed, then feeds the landmarks into a trained TensorFlow/Keras model to classify the sign.

## Idea in Short
- Use the webcam to capture frames.
- Detect a hand and extract 21 landmarks per hand using MediaPipe Hands.
- Flatten the 3D coordinates (x, y, z) of the 21 landmarks into a 63-length feature vector.
- Run the 63 features through a trained model (`asl_keypoints_model.h5`) to get a class prediction (A–Z + special tokens like `del`, `nothing`, `space`).
- Overlay the predicted class on the video stream.

## How the Code Works (app.py)
Below is the high-level flow that `app.py` implements:

1. Imports libraries
   - `cv2` for camera I/O and drawing
   - `numpy` for array operations
   - `tensorflow` to load and run the trained model
   - `mediapipe` to detect hands and extract 3D landmarks

2. Load model and define labels
   - Loads `asl_keypoints_model.h5` via `tf.keras.models.load_model(...)`
   - Defines `class_names` with 29 classes: `A`..`Z`, plus `del`, `nothing`, `space`

3. Initialize MediaPipe Hands
   - `mp.solutions.hands.Hands(...)` configured for real-time (video) mode, `max_num_hands=1`

4. Capture video
   - `cv2.VideoCapture(0)` opens the default webcam

5. For each frame
   - Read a frame and flip horizontally so it acts like a mirror
   - Convert BGR→RGB (MediaPipe expects RGB)
   - Run MediaPipe: `results = hands.process(rgb)`

6. If a hand is detected
   - Draw landmarks and connections for visualization
   - Build the feature vector:
     - For each of 21 landmarks, append `[lm.x, lm.y, lm.z]`
     - Final shape: `1 x 63` (batch size 1, 63 features)
   - Predict: `pred = model.predict(keypoints)` then `np.argmax(pred)` to get the index
   - Overlay predicted class text with `cv2.putText`

7. Display and exit
   - Show the frame in a window titled `ASL Prediction`
   - Press `q` to quit
   - Cleanly release the webcam and close windows

This is exactly what you see in code:
- Hand landmarks are collected on lines where we iterate `for lm in hand_landmarks.landmark` and extend with `x, y, z`.
- The model expects shape `(1, 63)`; the code reshapes with `np.array(keypoints).reshape(1, 63)`.
- Prediction is done with `model.predict(...)` and mapping is done via `class_names[np.argmax(...)]`.

## Project Structure
- `app.py`: Real-time prediction script using webcam + MediaPipe + TensorFlow model.
- `sign_language.ipynb`: Notebook for experiments/training/evaluation.
- `Data/`: Example data folders (e.g., `A/`, `B/`, `s/`).
- Models: `asl_keypoints_model.h5`, `asl_model.h5`, `asl_model_V2.h5`, `asl_model_V3.h5`.
- Images: e.g., `asl-alphabet.png` for reference.

## Requirements
- Python 3.9+
- Suggested packages (install what your code uses):
  - `tensorflow` (or `tensorflow-gpu` if you have a compatible GPU)
  - `opencv-python`
  - `numpy`
  - `mediapipe`
  - (Optional) `pandas`, `matplotlib`, `scikit-learn` for experiments in the notebook

If you want a frozen list later, you can generate `requirements.txt` after installing:
```bash
pip freeze > requirements.txt
```

## Setup (Windows PowerShell)
1) Navigate to the project directory:
```bash
cd "D:\computor vision projects\sign arabic langauge"
```
2) Create and activate a virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
3) Install dependencies (example):
```bash
pip install --upgrade pip
pip install tensorflow opencv-python numpy mediapipe
```

## Run
- Real-time prediction with webcam:
```bash
python app.py
```
- Explore/train/evaluate in the notebook:
```bash
jupyter notebook sign_language.ipynb
```

## Notes on Models
- `asl_keypoints_model.h5` expects a 63-length input vector: 21 hand landmarks × (x, y, z).
- If you switch to a different model file, make sure `app.py` points to it and that the input shape and preprocessing match.
- The label order in `class_names` must match the model’s training label order.

## Limitations and Tips
- Lighting and background affect detection quality.
- Keep the hand within the frame and avoid extreme angles.
- FPS depends on your CPU/GPU and model size.
- For better results: try larger training datasets, more robust augmentation, or sequence-based models (e.g., LSTM/Temporal models) for dynamic signs.

## Future Improvements
- Support multiple hands and/or dynamic signs (sequences over time)
- Add smoothing over several frames to stabilize predictions
- Provide a simple GUI and on-screen instructions
- Export a `requirements.txt` and a small demo dataset

## Model Concept and Architecture

- The core idea: classify a single hand sign using only 3D hand landmarks (21 points × x/y/z = 63 features) instead of raw images. This reduces input size, speeds up training/inference, and focuses on pose rather than background/lighting.
- Typical architecture used here (for `asl_keypoints_model.h5`):
  - Input: 63-dim vector of normalized landmark coordinates
  - Several Dense (fully-connected) layers with ReLU activation
  - Dropout for regularization
  - Output: Softmax layer over 29 classes (`A`–`Z`, `del`, `nothing`, `space`)
- Loss: Categorical cross-entropy; Optimizer: Adam (commonly used)

## Data and Preprocessing

- Dataset: ASL Alphabet (static hand signs) downloaded from Kaggle.
- Preprocessing pipeline:
  - Detect a hand with MediaPipe Hands for each image/frame.
  - Extract 21 landmarks → concatenate (x, y, z) into a 63-length vector.
  - Optional normalization (e.g., relative to wrist or bounding box) to reduce scale/translation variance.
  - Encode labels to match `class_names` order.

## Training Workflow (Notebook)

- Split data into training/validation (e.g., 80/20).
- Train the MLP/Dense network on landmark vectors.
- Monitor validation accuracy/loss; adjust epochs, learning rate, dropout if needed.
- Save the trained model as `asl_keypoints_model.h5` for use in `app.py`.

## Evaluation and Tips

- Evaluate on a held-out validation split and on real webcam data.
- Failure modes:
  - Poor lighting or partial hand in frame → landmarks less stable
  - Unseen camera angles or occlusions
  - Class confusion for visually similar signs
- Improve by:
  - Collecting more diverse samples per class
  - Data augmentation (rotate/scale/jitter landmarks)
  - Adding temporal smoothing across consecutive frames

## How to Retrain with Your Own Data

1) Collect images or short clips for each target class (e.g., `Data/A`, `Data/B`, ...).
2) In the notebook, run the preprocessing cells to extract landmarks and build `X` (features) and `y` (labels).
3) Train the model (`model.fit(...)`).
4) Save: `model.save("asl_keypoints_model.h5")`.
5) Ensure `class_names` in `app.py` matches the label order used during training.

## Inference Pipeline (Real-time)

- Capture a frame from webcam → detect hand → extract 21 landmarks → build 63-dim vector → run through the model → map argmax to human-readable label (`class_names`) → overlay text on the frame.
