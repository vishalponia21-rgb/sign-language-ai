# 🤟 Sign Language AI — v4.0

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.11-FF6F00?style=for-the-badge&logo=google&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-99.73%25-00FF88?style=for-the-badge)
![Samples](https://img.shields.io/badge/Trained%20On-126K%20Samples-9D4EDD?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**Real-time American Sign Language (ASL) alphabet recognition from webcam input.**
*Helping deaf and hard-of-hearing communities communicate digitally — no special hardware required.*

[Features](#-features) · [Architecture](#-how-it-works) · [Model](#-model-details) · [Dataset](#-dataset) · [Setup](#-quick-start) · [Results](#-results)

</div>

---

## 📸 What This Project Does

Sign Language AI is a **production-grade, real-time ASL recognition system** that:

1. **Captures** a webcam frame via browser
2. **Detects** your hand using Google's MediaPipe AI
3. **Extracts** 88 mathematical features from 21 hand landmarks
4. **Predicts** which ASL letter (A–Z) you are showing using a trained Random Forest classifier
5. **Displays** the detected letter with confidence score in a premium UI
6. **Builds words** letter by letter with a word builder (add, backspace, space, speak aloud)

When someone opens the app, they see a stunning landing page → click Launch → show a hand sign → get instant predictions with 99.73% accuracy.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 🎯 **99.73% Test Accuracy** | Random Forest trained on 126,718 samples |
| ⚡ **< 50ms Latency** | Full pipeline per frame on CPU — no GPU needed |
| ✋ **21 Hand Landmarks** | MediaPipe detects 3-D hand keypoints in real-time |
| 📐 **88 Feature Dimensions** | Normalized coords + pairwise distances |
| 🔤 **A–Z Detection** | All 26 English alphabet letters supported |
| 🔡 **Word Builder** | Add letters → Space → Backspace → Clear → Copy |
| 🔊 **Text-to-Speech** | Browser-native TTS reads your word aloud |
| 📜 **Detection History** | Last 10 detected letters shown as circular badges |
| 🎚 **Adjustable Settings** | Confidence & smoothing sliders in sidebar |
| 🤖 **Auto-Type Mode** | Hold a sign steady → letter added automatically |
| 📚 **Learn Signs Tab** | All 26 signs with hand-shape descriptions |
| 🏠 **Landing Page** | Premium animated splash screen before app |
| 🌑 **Ultra Premium Dark UI** | Neon green on navy blue with glassmorphism |
| 🦴 **Skeleton Overlay** | Live 21-point hand skeleton drawn on camera feed |

---

## 🔬 How It Works

```
Webcam Frame (browser)
       │
       ▼
┌──────────────────────────────────────┐
│  1. IMAGE DECODE                     │
│  JPEG bytes → NumPy BGR array       │
│  Upscale to ≥640px if too small     │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  2. HAND DETECTION                   │
│  MediaPipe HandLandmarker            │
│  → Detects hand in frame            │
│  → Outputs 21 keypoints (x, y, z)   │
│  → Confidence ≥ 0.20                │
└──────────────────┬───────────────────┘
                   │  21 × (x, y, z) coordinates
                   ▼
┌──────────────────────────────────────┐
│  3. FEATURE ENGINEERING              │
│  a) Translate: wrist → (0,0,0)      │
│  b) Scale: divide by wrist→midMCP   │
│  c) Flatten: 21×3 = 63 features     │
│  d) Distances: 25 landmark pairs    │
│  Total: 63 + 25 = 88 features       │
└──────────────────┬───────────────────┘
                   │  88-dim float32 vector
                   ▼
┌──────────────────────────────────────┐
│  4. CLASSIFICATION                   │
│  RandomForestClassifier              │
│  → 500 decision trees                │
│  → predict() → letter (A-Z)         │
│  → predict_proba() → confidence %   │
│  Model: model.pkl (538 MB)          │
└──────────────────┬───────────────────┘
                   │  letter + confidence
                   ▼
┌──────────────────────────────────────┐
│  5. POST-PROCESSING                  │
│  Confidence gate: ≥ threshold %     │
│  Smoothing: N identical frames       │
│  → Commit to detection history      │
└──────────────────┬───────────────────┘
                   │
                   ▼
        🔤 Letter displayed in UI
        📊 Confidence bar updated
        🦴 Skeleton overlay drawn
```

---

## 🤖 Model Details

### Model Name: **`model.pkl`**
### Model Type: **Random Forest Classifier**

| Property | Value |
|----------|-------|
| **Algorithm** | `sklearn.ensemble.RandomForestClassifier` |
| **File Name** | `model.pkl` |
| **File Size** | ~538 MB |
| **Serialization** | Python `pickle` (protocol = HIGHEST_PROTOCOL) |
| **Number of Trees** | **500** decision trees |
| **Max Depth** | None (fully grown) |
| **Min Samples Split** | 2 |
| **Min Samples Leaf** | 1 |
| **Max Features** | `"sqrt"` (standard RF feature sampling) |
| **Bootstrap** | True |
| **Class Weight** | `"balanced"` (handles any class imbalance) |
| **Random State** | 42 (reproducible) |
| **n_jobs** | -1 (uses all CPU cores for training) |
| **Classes** | 26 (A–Z) |
| **Input Features** | 88 float32 values |
| **Output** | Single ASL letter (A–Z) + probability vector |

### Why Random Forest?
- No GPU required — runs in < 10ms on any CPU
- Naturally handles 88-dimensional feature spaces
- Resistant to overfitting with 500 trees
- `predict_proba()` gives calibrated confidence scores
- Interpretable feature importances

---

## 📊 Feature Engineering

The model does NOT use raw pixel data. Instead, it uses **mathematical features extracted from hand landmarks**:

### Step 1 — Wrist Centering
```
All 21 landmarks are translated so that the wrist (landmark 0) is at (0, 0, 0).
This makes features position-independent (hand can be anywhere in frame).
```

### Step 2 — Scale Normalization
```
All coordinates divided by the distance from wrist to Middle MCP (landmark 9).
This makes features size-independent (hand can be near or far from camera).
```

### Step 3 — Coordinate Flattening (63 features)
```
21 landmarks × (x, y, z) = 63 normalized float values
```

### Step 4 — Pairwise Distances (25 features)
```
Euclidean distances between 25 key landmark pairs:
- 5 fingertip-to-wrist distances (thumb, index, middle, ring, pinky)
- 4 thumb-to-other-fingertip distances
- 3 adjacent fingertip-to-fingertip distances
- 2 cross-hand distances
- 5 fingertip-to-own-MCP distances (curl detection)
- 1 palm width (index MCP to pinky MCP)
- 5 adjacent MCP distances
Total: 63 + 25 = 88 features
```

### Why These Features?
- **Position invariant** — works anywhere in frame
- **Scale invariant** — works at any distance from camera
- **Distance features** — capture finger curl, spread, and shape
- **Much better** than raw pixel data or raw coordinates

---

## 📊 Results

### Test Set Performance (15% hold-out)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **99.73%** |
| Test Samples | 19,008 |
| Train Samples | 107,710 |
| Macro Avg Precision | 1.00 |
| Macro Avg Recall | 1.00 |
| Macro Avg F1-Score | 1.00 |

### Per-Letter Accuracy

| Letter | Accuracy | Letter | Accuracy | Letter | Accuracy |
|--------|----------|--------|----------|--------|----------|
| A | 100.0% | J | 99.8% | S | 99.4% |
| B | 100.0% | K | 100.0% | T | 99.7% |
| C | 100.0% | L | 99.7% | U | 99.7% |
| D | 99.6%  | M | 99.3% | V | 99.7% |
| E | 100.0% | N | 98.5% | W | 99.5% |
| F | 100.0% | O | 99.7% | X | 99.1% |
| G | 99.9%  | P | 99.4% | Y | 100.0% |
| H | 99.7%  | Q | 100.0% | Z | 99.5% |
| I | 100.0% | R | 100.0% | — | — |

> **N** (98.5%) is the weakest letter — it looks similar to M and is intentionally harder.

---

## 🗂 Dataset

### Primary Dataset — Kaggle ASL Alphabet
- **Name**: [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) by Grassknoted
- **Total Images Available**: ~87,000 (29 classes)
- **Classes Used**: A–Z only (26 classes)
- **Images Used**: 3,000 per letter × 26 = **78,000 images**
- **Format**: 200×200 pixel JPG
- **Size on disk**: ~1 GB

### Data Augmentation
- **Horizontal Flip**: Each image is flipped left-right to simulate mirror hands
- **Effect**: 78,000 × 2 = **156,000 candidate samples**
- **After MediaPipe filtering** (some images have no detectable hand): **126,718 final samples**

### Train/Test Split
- **Training set**: 107,710 samples (85%)
- **Test set**: 19,008 samples (15%)
- **Stratified**: Equal proportion of each class in both sets
- **Random state**: 42

---

## 📁 Project Structure

```
sign-language-ai/
│
├── app.py                    ← Main Streamlit app (v4.0 Ultra Premium UI)
├── main.py                   ← Orchestration entry point
├── train_asl.py              ← Model training pipeline (v3.0, improved)
├── predict.py                ← Standalone OpenCV prediction (no Streamlit)
├── test_camera.py            ← Camera diagnostic utility
├── collect_data.py           ← Collect custom sign samples
├── generate_complete_dataset.py  ← Dataset generation helper
│
├── model.pkl                 ← ✅ Trained Random Forest (~538 MB)
├── hand_landmarker.task      ← MediaPipe hand model (7.4 MB)
│
├── X_keypoints.npy           ← Cached feature matrix (optional)
├── y_labels.npy              ← Cached labels (optional)
│
├── asl_dataset/              ← Kaggle ASL images (A-Z folders, ~1 GB)
│   ├── A/  (3000 images)
│   ├── B/  (3000 images)
│   └── …   (A–Z, 3000 each)
│
├── dataset/                  ← Custom ISL samples (2600 images)
├── asl_keypoints_model.h5    ← Earlier Keras model (not used in v4)
├── asl_keypoints_model_V2.h5 ← Earlier Keras model V2 (not used in v4)
│
├── requirements.txt          ← Python dependencies
├── .gitignore
└── README.md
```

---

## 🛠 Libraries & Dependencies

### Core ML & Computer Vision

| Library | Version | Purpose |
|---------|---------|---------|
| **mediapipe** | 0.10.11 | Hand landmark detection (21 keypoints) using `HandLandmarker` task |
| **scikit-learn** | ≥ 1.3.0 | `RandomForestClassifier` for ASL prediction |
| **opencv-python-headless** | 4.8.1.78 | Image decode, resize, BGR↔RGB conversion, skeleton drawing |
| **numpy** | 1.26.4 | Numerical operations, feature arrays, matrix math |

### Web UI

| Library | Version | Purpose |
|---------|---------|---------|
| **streamlit** | ≥ 1.28.0 | Web app framework, camera input, session state, sidebar |
| **streamlit.components.v1** | (built-in) | Embedding raw HTML/JS (landing page, TTS) |

### Data & Utilities

| Library | Version | Purpose |
|---------|---------|---------|
| **pandas** | ≥ 2.0.0 | Data manipulation during training |
| **Pillow** | ≥ 10.0.0 | Image processing support |
| **pickle** | (stdlib) | Model serialization/deserialization |
| **base64** | (stdlib) | Encoding hand image for CSS background |
| **collections.deque** | (stdlib) | Circular buffer for detection history |

### MediaPipe Internals (auto-installed)
- **TensorFlow Lite** — powers the HandLandmarker model
- **XNNPACK** — CPU acceleration delegate
- **OpenGL** — GPU context (used even on CPU mode on macOS/Metal)

---

## 🔧 Training Pipeline (train_asl.py)

```python
# 1. Initialize MediaPipe HandLandmarker
#    - min_hand_detection_confidence = 0.30
#    - running_mode = IMAGE (batch processing)

# 2. For each letter folder (A-Z):
#    - Load up to 3000 images
#    - Resize to 400px max dimension (speed vs accuracy balance)
#    - Convert BGR → RGB
#    - Run MediaPipe → get 21 landmarks
#    - Extract 88 features (normalize + distances)
#    - AUGMENT: horizontal flip → extract 88 more features
#    - Skip images where no hand detected

# 3. Build dataset: 126,718 samples × 88 features

# 4. Train/Test split: 85% / 15%, stratified, random_state=42

# 5. Train RandomForestClassifier:
#    n_estimators=500, class_weight='balanced', n_jobs=-1

# 6. Evaluate on test set → 99.73% accuracy

# 7. Save as model.pkl using pickle.HIGHEST_PROTOCOL
```

**Training time**: ~34 minutes (MediaPipe processing) + ~75 seconds (Random Forest)
**Total**: ~35–40 minutes on Apple M2

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Webcam connected
- ~2 GB free disk space
- `asl_dataset/` folder (Kaggle download)
- `hand_landmarker.task` file

### 1 · Clone & enter directory
```bash
git clone https://github.com/vishalponia21-rgb/sign-language-ai.git
cd sign-language-ai
```

### 2 · Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### 3 · Install dependencies
```bash
pip install -r requirements.txt
```

### 4 · Get `hand_landmarker.task`
```bash
curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" \
     -o hand_landmarker.task
```

### 5 · Train the model (first time only, ~40 min)
```bash
python train_asl.py
```

### 6 · Run the app
```bash
python main.py
# OR directly:
streamlit run app.py --theme.base dark --theme.primaryColor "#00FF88"
```

Open **http://localhost:8501** in your browser.

---

## 🎛 App Controls

| Control | Location | Description |
|---------|----------|-------------|
| Min Confidence % | Sidebar slider | Only commit predictions above this level (default 40%) |
| Smoothing Frames | Sidebar slider | Same letter required N times before showing (default 2) |
| Auto-Type Mode | Sidebar toggle | Auto-add detected letter without clicking Add |
| Hold Frames | Sidebar slider (auto-type) | Frames to hold before auto-adding next letter |
| ➕ Add Letter | Main panel | Manually add detected letter to word |
| ⌫ Delete | Main panel | Remove last character |
| ⎵ Space | Main panel | Add space between words |
| 🗑 Clear | Main panel | Clear entire word |
| 🔊 Speak | Main panel | Read word aloud using browser TTS |
| ⬅️ Back to Home | Sidebar | Return to landing page |

---

## 🧪 Other Scripts

### Standalone prediction (OpenCV window)
```bash
python predict.py
# Press Q to quit
```

### Test your camera
```bash
python test_camera.py
```

### Collect custom sign data
```bash
python collect_data.py
# Enter label → press SPACE → records 100 frames
```

---

## ⚡ Performance Benchmarks

| Operation | Time |
|-----------|------|
| Model loading (first time) | ~3–5 seconds |
| Model loading (cached) | < 100ms |
| Frame decode + resize | ~3–5 ms |
| MediaPipe hand detection | ~20–35 ms |
| Feature extraction | < 1 ms |
| Random Forest predict | ~5–10 ms |
| **Total per frame** | **< 50 ms** |
| **Effective FPS** | **~20–30 FPS** |

---

## 🌍 Social Impact

- **18 lakh+** deaf and hard-of-hearing individuals in India alone
- **Globally**: 430 million people require rehabilitation for hearing loss (WHO)
- No special hardware required — any laptop with a webcam works
- Real-time text output bridges communication gap
- Foundation for: full sentence recognition, ISL support, mobile apps

---

## 🔮 Possible Extensions

- [ ] **ISL (Indian Sign Language)** support with custom dataset
- [ ] **Full sentence recognition** with LSTM/Transformer
- [ ] **Mobile app** — React Native / Flutter
- [ ] **Real-time translation API** — Hindi, Tamil, etc.
- [ ] **Two-hand detection** — `num_hands=2`
- [ ] **Dynamic gesture recognition** — LSTM for J, Z motion letters
- [ ] **User authentication** + conversation history
- [ ] **Cloud deployment** — Hugging Face Spaces / Streamlit Cloud

---

## 🙋 Challenges & Solutions

| Challenge | Solution |
|-----------|---------|
| Low confidence on similar signs (M/N, U/V) | 6× more training data + pairwise distance features |
| Raw coordinates change with hand distance | Wrist-centered + scale normalization |
| Flickering predictions | N-frame smoothing buffer |
| OpenCV on Streamlit Cloud | Use `opencv-python-headless` |
| model.pkl too large for GitHub | Listed in `.gitignore`; train locally with `train_asl.py` |
| HTML in st.markdown() showing as text | Use `components.html()` for complex HTML blocks |
| Slow training (78K images) | MediaPipe batch processing + multicore Random Forest |

---

## 💡 Deployment Options

| Platform | Webcam | Cost | Notes |
|----------|--------|------|-------|
| **Local (recommended)** | ✅ Full | Free | Best experience |
| Streamlit Cloud | ⚠️ Upload only | Free | Camera doesn't stream |
| Hugging Face Spaces | ⚠️ Upload only | Free | Same limitation |
| AWS / GCP | ✅ With setup | Paid | Needs video streaming setup |

---

## 👨‍💻 Author

**Vishal Ponia**
Computer Vision & Machine Learning Engineer

[![GitHub](https://img.shields.io/badge/GitHub-vishalponia21--rgb-181717?style=flat&logo=github)](https://github.com/vishalponia21-rgb)

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

**Sign Language AI v4.0**

Built with ❤️ for the deaf community

MediaPipe × Random Forest × Streamlit × 126K Samples × 99.73% Accuracy

*Making communication accessible — one hand sign at a time.* 🤟

</div>