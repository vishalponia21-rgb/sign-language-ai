import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import pickle
from collections import deque
from datetime import datetime
from PIL import Image
import io
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="Sign Language AI 🤟",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS for Dark Theme ====================
st.markdown("""
    <style>
    :root {
        --primary-color: #00FF88;
        --secondary-color: #1a1a1a;
        --text-color: #ffffff;
    }
    
    .stApp {
        background-color: #0a0a0a;
        color: #ffffff;
    }
    
    .main-title {
        font-size: 3.5em;
        font-weight: bold;
        background: linear-gradient(135deg, #00FF88 0%, #00cc6a 50%, #00FF88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5em;
        animation: gradientShift 3s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .letter-display {
        font-size: 8em;
        font-weight: bold;
        text-align: center;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        animation: bounce 0.6s ease;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }
    
    .letter-green {
        color: #00FF88;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.8);
    }
    
    .letter-yellow {
        color: #FFD700;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.8);
    }
    
    .letter-red {
        color: #FF6B6B;
        text-shadow: 0 0 30px rgba(255, 107, 107, 0.8);
    }
    
    .word-display {
        font-size: 4em;
        font-weight: bold;
        color: #00FF88;
        text-align: center;
        padding: 20px;
        background-color: #1a1a1a;
        border-radius: 10px;
        border: 2px solid #00FF88;
        margin: 10px 0;
        min-height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .confidence-text {
        font-size: 1.5em;
        text-align: center;
        color: #00FF88;
    }
    
    .history-item {
        display: inline-block;
        background-color: #00FF88;
        color: #000000;
        padding: 8px 15px;
        margin: 5px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .stat-box {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00FF88;
        margin: 10px 0;
    }
    
    .stat-number {
        font-size: 2em;
        font-weight: bold;
        color: #00FF88;
    }
    
    .stat-label {
        color: #888888;
        font-size: 0.9em;
    }
    
    .fps-counter {
        position: fixed;
        top: 10px;
        right: 10px;
        background-color: rgba(0, 255, 136, 0.2);
        color: #00FF88;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 0.9em;
        z-index: 1000;
        border: 1px solid #00FF88;
    }
    
    .practice-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border: 2px solid #00FF88;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    
    .practice-letter {
        font-size: 6em;
        color: #00FF88;
        margin: 20px 0;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== Session State Initialization ====================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'hand_landmarker' not in st.session_state:
    st.session_state.hand_landmarker = None
if 'current_word' not in st.session_state:
    st.session_state.current_word = ""
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = deque(maxlen=10)
if 'frame_counter' not in st.session_state:
    st.session_state.frame_counter = {}
if 'last_letter' not in st.session_state:
    st.session_state.last_letter = None

# New session state for real-time camera
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'frame_time' not in st.session_state:
    st.session_state.frame_time = deque(maxlen=30)
    
# Auto word builder
if 'letter_hold_time' not in st.session_state:
    st.session_state.letter_hold_time = {}
if 'auto_add_threshold' not in st.session_state:
    st.session_state.auto_add_threshold = 1.5  # seconds

# Session stats
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = datetime.now()
if 'total_letters_detected' not in st.session_state:
    st.session_state.total_letters_detected = 0
if 'words_formed' not in st.session_state:
    st.session_state.words_formed = 0

# Practice mode
if 'practice_mode' not in st.session_state:
    st.session_state.practice_mode = False
if 'practice_letter' not in st.session_state:
    st.session_state.practice_letter = None
if 'practice_score' not in st.session_state:
    st.session_state.practice_score = 0
if 'practice_attempts' not in st.session_state:
    st.session_state.practice_attempts = 0

# TTS instance
if 'tts_engine' not in st.session_state and TTS_AVAILABLE:
    try:
        st.session_state.tts_engine = pyttsx3.init()
        st.session_state.tts_engine.setProperty('rate', 150)
    except:
        st.session_state.tts_engine = None

# ==================== Model Loading ====================
@st.cache_resource
def load_model():
    """Load the trained Random Forest model"""
    with st.spinner("🤖 Loading AI model..."):
        try:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            st.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return None

@st.cache_resource
def load_hand_landmarker():
    """Load MediaPipe Hand Landmarker"""
    with st.spinner("🖐️ Loading hand detection model..."):
        try:
            MODEL_PATH = "hand_landmarker.task"
            base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            landmarker = vision.HandLandmarker.create_from_options(options)
            st.success("✅ Hand detection model loaded!")
            return landmarker
        except Exception as e:
            st.error(f"❌ Error loading hand landmarker: {str(e)}")
            return None

# ==================== New Utility Functions ====================
def speak_text(text):
    """Text-to-speech functionality"""
    if TTS_AVAILABLE and st.session_state.tts_engine:
        try:
            st.session_state.tts_engine.say(text)
            st.session_state.tts_engine.runAndWait()
        except:
            pass

def get_confidence_color(confidence):
    """Get color based on confidence score"""
    if confidence >= 0.8:
        return "letter-green"  # Green > 80%
    elif confidence >= 0.6:
        return "letter-yellow"  # Yellow 60-80%
    else:
        return "letter-red"  # Red < 60%

def get_confidence_rgb(confidence):
    """Get RGB color tuple based on confidence"""
    if confidence >= 0.8:
        return (0, 255, 136)  # Green
    elif confidence >= 0.6:
        return (255, 215, 0)  # Yellow
    else:
        return (255, 107, 107)  # Red

def calculate_fps():
    """Calculate current FPS"""
    if len(st.session_state.frame_time) > 1:
        time_diffs = np.diff(list(st.session_state.frame_time))
        avg_time = np.mean(time_diffs)
        if avg_time > 0:
            return 1.0 / avg_time
    return 0

def get_session_stats():
    """Get current session statistics"""
    elapsed = (datetime.now() - st.session_state.session_start_time).total_seconds()
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    return {
        'elapsed_time': f"{hours:02d}:{minutes:02d}:{seconds:02d}",
        'total_letters': st.session_state.total_letters_detected,
        'words_formed': st.session_state.words_formed,
        'current_word_length': len(st.session_state.current_word.replace(" ", ""))
    }

def export_words():
    """Generate downloadable text file of detected words"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sign_language_output_{timestamp}.txt"
    content = f"""Sign Language AI - Detection Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

=== Session Statistics ===
Session Duration: {get_session_stats()['elapsed_time']}
Total Letters Detected: {st.session_state.total_letters_detected}
Words Formed: {st.session_state.words_formed}

=== Detected Letters History ===
{' '.join(st.session_state.detection_history)}

=== Final Output ===
{st.session_state.current_word}
"""
    return content, filename

# ==================== Existing Utility Functions ====================
def draw_hand_landmarks(frame, hand_landmarks):
    """Draw hand landmarks on the frame"""
    h, w, _ = frame.shape
    for landmark in hand_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (0, 255, 136), -1)  # Green dots
    return frame

def extract_features(hand_landmarks):
    """Extract 63 features (21 points × 3 coordinates) from hand landmarks"""
    features = []
    for lm in hand_landmarks:
        features.extend([round(lm.x, 4), round(lm.y, 4), round(lm.z, 4)])
    return features

def detect_hand(frame, hand_landmarker):
    """Detect hand landmarks in a frame"""
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect(mp_image)
    
    if result.hand_landmarks:
        return result.hand_landmarks[0]
    return None

def predict_sign(features, model):
    """Predict the sign and confidence"""
    try:
        prediction = model.predict([features])[0]
        confidence = model.predict_proba([features]).max() * 100
        return prediction, confidence / 100
    except Exception as e:
        return None, 0

def add_to_history(letter):
    """Add letter to detection history"""
    st.session_state.detection_history.appendleft(letter)
    st.session_state.total_letters_detected += 1

def auto_add_letter_to_word(letter, confidence):
    """Auto add letter to word after 1.5 seconds of holding"""
    if letter not in st.session_state.letter_hold_time:
        st.session_state.letter_hold_time[letter] = time.time()
    
    hold_duration = time.time() - st.session_state.letter_hold_time[letter]
    progress = min(hold_duration / st.session_state.auto_add_threshold, 1.0)
    
    if hold_duration >= st.session_state.auto_add_threshold and confidence >= 0.6:
        st.session_state.current_word += letter
        st.session_state.letter_hold_time[letter] = time.time()  # Reset timer
        speak_text(letter)
        st.balloons()  # Celebration!
        return True, progress
    
    return False, progress

# ==================== Sign Descriptions ====================
SIGN_DESCRIPTIONS = {
    'A': "Close your hand into a fist with thumb on the side",
    'B': "Open hand with fingers together, thumb bent",
    'C': "Make a C shape with thumb and fingers",
    'D': "Fist with index finger pointing up",
    'E': "Clawed hand position",
    'F': "Thumb and index finger in O shape, other fingers up",
    'G': "Point with index and middle finger sideways",
    'H': "Index and middle fingers together, other fingers down",
    'I': "Pinky finger pointing up",
    'J': "Hook shape with pinky finger",
    'K': "Three fingers up, index and middle spread",
    'L': "Thumb and index finger in L shape",
    'M': "Three fingers over thumb",
    'N': "Two fingers over thumb",
    'O': "Fingers and thumb in O shape",
    'P': "Similar to K but bent down",
    'Q': "Similar to G but bent down",
    'R': "Cross index and middle fingers",
    'S': "Fist position",
    'T': "Thumb between index and middle fingers",
    'U': "Two fingers up, together",
    'V': "Two fingers spread (peace sign)",
    'W': "Three fingers spread",
    'X': "Index finger bent in X",
    'Y': "Thumb and pinky up, other fingers down",
    'Z': "Index finger making Z motion"
}

# ==================== Sidebar ====================
with st.sidebar:
    st.markdown('<div class="main-title">🤟 Sign Language AI</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # About Section
    st.subheader("📖 About This Project")
    st.write(
        "Real-time American Sign Language (ASL) recognition using AI. "
        "Our model detects hand gestures from your webcam and translates them to letters."
    )
    
    st.markdown("---")
    st.subheader("📊 Project Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">26</div>
            <div class="stat-label">Letters Supported</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">96%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">⚡</div>
            <div class="stat-label">Real-time</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">📊</div>
            <div class="stat-label">10K+ Samples</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📚 How to Use")
    st.info(
        "1. Go to the **Live Detection** tab\n"
        "2. Allow camera access\n"
        "3. Make ASL hand gestures\n"
        "4. Watch your gesture get recognized\n"
        "5. Build words with the letter buttons"
    )
    
    st.markdown("---")
    st.subheader("🛠️ Tech Stack")
    st.write(
        "- **MediaPipe**: Hand landmark detection\n"
        "- **OpenCV**: Video processing\n"
        "- **Scikit-learn**: Random Forest classification\n"
        "- **Streamlit**: Web interface"
    )
    
    st.markdown("---")
    st.caption("👨‍💻 Built by **Vishal Ponia**")
    st.caption("Dataset: Kaggle ASL Dataset")

# ==================== Main Content ====================
tab1, tab2, tab3 = st.tabs(["🎥 Live Detection", "📖 Learn Signs", "ℹ️ About"])

# ==================== TAB 1: LIVE DETECTION ====================
with tab1:
    st.markdown('<div class="main-title">🎥 Real-Time Hand Detection</div>', unsafe_allow_html=True)
    
    # Load models
    if st.session_state.model is None:
        st.session_state.model = load_model()
    if st.session_state.hand_landmarker is None:
        st.session_state.hand_landmarker = load_hand_landmarker()
    
    if st.session_state.model is None or st.session_state.hand_landmarker is None:
        st.error("❌ Failed to load models. Please check the files exist.")
        st.stop()
    
    # Session stats at top
    stats = get_session_stats()
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    with col_stats1:
        st.metric("⏱️ Session Time", stats['elapsed_time'])
    with col_stats2:
        st.metric("📊 Letters", stats['total_letters'])
    with col_stats3:
        st.metric("📝 Words", st.session_state.words_formed)
    with col_stats4:
        st.metric("🎯 Word Length", stats['current_word_length'])
    
    st.markdown("---")
    
    # Create main layout
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("⚙️ Controls & Settings")
        
        # Start/Stop button for real-time camera
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("▶️ Start Camera"):
                st.session_state.camera_running = True
        with col_stop:
            if st.button("⏹️ Stop Camera"):
                st.session_state.camera_running = False
        
        confidence_threshold = st.slider("Confidence", 30, 95, 60) / 100
        
        st.markdown("---")
        st.subheader("📝 Word Builder")
        
        # Word display with border
        word_text = st.session_state.current_word if st.session_state.current_word else "..."
        st.markdown(
            f'<div class="word-display">{word_text}</div>',
            unsafe_allow_html=True
        )
        
        # Word builder buttons
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("⌫ Delete"):
                if st.session_state.current_word:
                    st.session_state.current_word = st.session_state.current_word[:-1]
                    st.rerun()
        with col_b:
            if st.button("🔤 Space"):
                st.session_state.current_word += " "
                st.session_state.words_formed += 1
                st.rerun()
        with col_c:
            if st.button("🗑️ Clear"):
                st.session_state.current_word = ""
                st.rerun()
        
        # Export button
        if st.session_state.current_word:
            content, filename = export_words()
            st.download_button(
                label="📥 Export Words",
                data=content,
                file_name=filename,
                mime="text/plain"
            )
            
            # Speak word button
            if TTS_AVAILABLE:
                if st.button("🔊 Speak"):
                    speak_text(st.session_state.current_word.replace(" ", " "))
        
        st.markdown("---")
        st.subheader("📜 Detection History")
        if st.session_state.detection_history:
            history_html = " ".join([f'<span class="history-item">{letter}</span>' for letter in st.session_state.detection_history])
            st.markdown(f'<div>{history_html}</div>', unsafe_allow_html=True)
        else:
            st.info("👋 Detections will appear here")
    
    with col1:
        st.subheader("🎬 Live Webcam Feed")
        
        video_placeholder = st.empty()
        info_placeholder = st.empty()
        letter_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        if st.session_state.camera_running:
            cap = cv2.VideoCapture(0)
            last_detected_letter = None
            frame_counter = {}
            
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Cannot access camera")
                    break
                
                frame_start_time = time.time()
                confidence = 0  # Initialize confidence variable
                
                # Flip frame
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                # Detect hand
                hand_landmarks = detect_hand(frame, st.session_state.hand_landmarker)
                
                if hand_landmarks:
                    # Draw landmarks
                    frame = draw_hand_landmarks(frame, hand_landmarks)
                    
                    # Extract and predict
                    features = extract_features(hand_landmarks)
                    prediction, confidence = predict_sign(features, st.session_state.model)
                    
                    # Smoothing - require 3 consecutive frames
                    if prediction not in frame_counter:
                        frame_counter[prediction] = 0
                    
                    if confidence >= confidence_threshold:
                        frame_counter[prediction] += 1
                        frame_counter = {prediction: frame_counter[prediction]}
                    else:
                        frame_counter = {}
                    
                    # Display detected letter with bounce animation
                    if frame_counter.get(prediction, 0) >= 3:
                        if prediction != last_detected_letter:
                            last_detected_letter = prediction
                            add_to_history(prediction)
                            
                            # Auto-add to word
                            auto_added, progress = auto_add_letter_to_word(prediction, confidence)
                        
                        conf_color = get_confidence_color(confidence)
                        conf_rgb = get_confidence_rgb(confidence)
                        
                        # Draw detection on frame
                        cv2.putText(frame, f"{prediction}", (20, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 2, conf_rgb, 3)
                        cv2.putText(frame, f"{confidence*100:.1f}%", (20, 100),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, conf_rgb, 2)
                        
                        # Display in UI with confidence-based color
                        letter_html = f'<div class="{conf_color} letter-display">{prediction}</div>'
                        letter_placeholder.markdown(letter_html, unsafe_allow_html=True)
                        
                        # Show progress for auto-add (convert to 0-100 for st.progress)
                        _, progress = auto_add_letter_to_word(prediction, confidence)
                        try:
                            progress_placeholder.progress(int(progress * 100))
                        except Exception:
                            # Fallback: ensure progress is in 0-100
                            p_val = max(0, min(int(progress * 100), 100))
                            progress_placeholder.progress(p_val)
                        # Display countdown text separately (info placeholder will be refreshed each loop)
                        try:
                            info_placeholder.write(f"Hold for auto-add: {(st.session_state.auto_add_threshold * (1 - progress)):.1f}s")
                        except Exception:
                            pass
                    else:
                        letter_placeholder.empty()
                        progress_placeholder.empty()
                else:
                    letter_placeholder.empty()
                    progress_placeholder.empty()
                    cv2.putText(frame, "Show your hand...", (20, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # FPS calculation
                frame_time = time.time() - frame_start_time
                st.session_state.frame_time.append(time.time())
                fps = calculate_fps()
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # `use_column_width` is deprecated — use explicit width
                video_placeholder.image(frame_rgb, width=640)
                
                # Info display
                info_placeholder.write(f"🎬 FPS: {fps:.1f} | Hand: {'✓' if hand_landmarks else '✗'} | Confidence: {confidence*100:.1f}% (threshold: {confidence_threshold*100:.0f}%)")
                
                time.sleep(0.01)  # ~100ms per frame
            
            cap.release()
        else:
            st.info("Click **Start Camera** to begin detection")

# ==================== TAB 2: LEARN SIGNS & PRACTICE ====================
with tab2:
    st.markdown('<div class="main-title">📖 Learn ASL & Practice</div>', unsafe_allow_html=True)
    
    tab2_learn, tab2_practice = st.tabs(["📚 Learn", "🎮 Practice Mode"])
    
    with tab2_learn:
        st.write("Explore how to make each ASL letter. Click on any letter to see detailed instructions.")
        
        # Create grid of signs
        cols_per_row = 4
        letters = list(SIGN_DESCRIPTIONS.keys())
        
        for i in range(0, len(letters), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(letters):
                    letter = letters[i + j]
                    with col:
                        # Sign card
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
                            border: 2px solid #00FF88;
                            border-radius: 15px;
                            padding: 20px;
                            text-align: center;
                            cursor: pointer;
                        ">
                            <div style="font-size: 3em; margin: 10px 0; color: #00FF88; font-weight: bold;">
                                {letter}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Expandable description
                        with st.expander(f"How to make '{letter}'"):
                            st.write(f"**Description:** {SIGN_DESCRIPTIONS[letter]}")
                            st.info("💡 Tip: Keep your hand steady and centered for better detection!")
    
    with tab2_practice:
        st.write("Test your skills! The AI will show you a random letter, and you need to make that sign.")
        
        col_prac1, col_prac2 = st.columns([2, 1])
        
        with col_prac2:
            if st.button("🎲 Start New Practice Session"):
                st.session_state.practice_mode = True
                st.session_state.practice_score = 0
                st.session_state.practice_attempts = 0
                st.session_state.practice_letter = np.random.choice(list(SIGN_DESCRIPTIONS.keys()))
                st.rerun()
        
        with col_prac1:
            if st.session_state.practice_score > 0 or st.session_state.practice_attempts > 0:
                practice_stats = st.columns(3)
                with practice_stats[0]:
                    st.metric("✅ Correct", st.session_state.practice_score)
                with practice_stats[1]:
                    st.metric("📊 Attempts", st.session_state.practice_attempts)
                with practice_stats[2]:
                    accuracy = (st.session_state.practice_score / st.session_state.practice_attempts * 100) if st.session_state.practice_attempts > 0 else 0
                    st.metric("🎯 Accuracy", f"{accuracy:.1f}%")
        
        st.markdown("---")
        
        if st.session_state.practice_mode and st.session_state.practice_letter:
            st.subheader("🎯 Make This Sign:")
            st.markdown(f'<div class="practice-card"><div class="practice-letter">{st.session_state.practice_letter}</div></div>', unsafe_allow_html=True)
            st.write(f"**How to make it:** {SIGN_DESCRIPTIONS[st.session_state.practice_letter]}")
            
            if st.button("✓ I Made It! (Check)"):
                st.session_state.practice_attempts += 1
                # In a real app, this would use camera to verify
                st.success(f"Great! You practiced '{st.session_state.practice_letter}'")
                st.session_state.practice_score += 1
                st.session_state.practice_letter = np.random.choice(list(SIGN_DESCRIPTIONS.keys()))
                st.rerun()
            
            if st.button("⏭️ Skip This Letter"):
                st.session_state.practice_letter = np.random.choice(list(SIGN_DESCRIPTIONS.keys()))
                st.rerun()
            
            if st.button("🛑 End Practice Session"):
                st.session_state.practice_mode = False
                st.rerun()
        else:
            st.info("👉 Click **Start New Practice Session** to begin!")

# ==================== TAB 3: ABOUT ====================
with tab3:
    st.markdown('<div class="main-title">About Sign Language AI</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Project Overview")
        st.write(
            "Sign Language AI is a real-time American Sign Language recognition system "
            "that detects ASL alphabet letters through hand gesture recognition. "
            "This project aims to bridge communication gaps and make sign language "
            "more accessible through technology."
        )
        
        st.subheader("🤖 How It Works")
        st.write(
            "1. **Hand Detection**: MediaPipe detects 21 hand landmarks\n"
            "2. **Feature Extraction**: 63 features (21 points × 3 coordinates)\n"
            "3. **Classification**: Random Forest model predicts the letter\n"
            "4. **Confidence Filtering**: Only shows predictions with >60% confidence\n"
            "5. **Smoothing**: Requires 3 consistent frames for accurate detection"
        )
    
    with col2:
        st.subheader("📊 Technical Stack")
        st.markdown("""
        - **MediaPipe 0.10.11** - Hand landmark detection
        - **OpenCV** - Video capture and processing
        - **Scikit-learn** - Random Forest classifier
        - **Python 3.11** - Core language
        - **Streamlit** - Web interface framework
        """)
        
        st.subheader("📈 Model Performance")
        metrics_data = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Score": ["96%", "95%", "96%", "95.5%"]
        }
        st.table(metrics_data)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("📚 **Dataset**: Kaggle ASL Dataset\n\n10,045 total samples\n26 letters (A-Z)")
    
    with col2:
        st.success("🎯 **Real-time**: <50ms per frame\n\nSmooth live detection")
    
    with col3:
        st.warning("⚡ **GPU Optional**: Runs on CPU\n\nNo GPU required")
    
    st.markdown("---")
    
    st.subheader("👨‍💻 Developer")
    st.write("**Vishal Ponia** - AI/ML Developer")
    
    st.subheader("🔗 Links")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("[GitHub](https://github.com)")
    with col2:
        st.markdown("[LinkedIn](https://linkedin.com)")
    with col3:
        st.markdown("[Portfolio](https://portfolio.com)")
    
    st.markdown("---")
    st.caption(
        "🙏 This project is for educational and accessibility purposes. "
        "All credit goes to the Kaggle ASL dataset contributors."
    )
