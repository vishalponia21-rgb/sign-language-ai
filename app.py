"""
app.py — Sign Language AI v4.0 | ULTRA PREMIUM UI
99.73% accuracy · MediaPipe × Random Forest · 126K samples
"""

import streamlit as st
import cv2, numpy as np, pickle, base64, os
from collections import deque
import mediapipe as mp
from mediapipe.tasks.python import vision
import streamlit.components.v1 as components

# ══════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sign Language AI 🤟",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════
#  CONSTANTS & FEATURE CONFIG
# ══════════════════════════════════════════════════════
WRIST=0; THUMB_TIP=4; INDEX_TIP=8; MIDDLE_TIP=12; RING_TIP=16; PINKY_TIP=20
INDEX_MCP=5; MIDDLE_MCP=9; RING_MCP=13; PINKY_MCP=17; THUMB_MCP=2

DISTANCE_PAIRS=[
    (WRIST,THUMB_TIP),(WRIST,INDEX_TIP),(WRIST,MIDDLE_TIP),(WRIST,RING_TIP),(WRIST,PINKY_TIP),
    (THUMB_TIP,INDEX_TIP),(THUMB_TIP,MIDDLE_TIP),(THUMB_TIP,RING_TIP),(THUMB_TIP,PINKY_TIP),
    (INDEX_TIP,MIDDLE_TIP),(MIDDLE_TIP,RING_TIP),(RING_TIP,PINKY_TIP),
    (INDEX_TIP,PINKY_MCP),(PINKY_TIP,INDEX_MCP),
    (THUMB_TIP,THUMB_MCP),(INDEX_TIP,INDEX_MCP),(MIDDLE_TIP,MIDDLE_MCP),(RING_TIP,RING_MCP),(PINKY_TIP,PINKY_MCP),
    (INDEX_MCP,PINKY_MCP),(INDEX_TIP,RING_TIP),(THUMB_TIP,MIDDLE_TIP),
    (INDEX_MCP,MIDDLE_MCP),(MIDDLE_MCP,RING_MCP),(RING_MCP,PINKY_MCP),
]
HAND_CONNECTIONS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
                  (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
SIGN_DESCRIPTIONS={
    "A":"Closed fist; thumb rests alongside index finger, pointing up.",
    "B":"Open flat hand; four fingers together pointing up, thumb bent across palm.",
    "C":"Curve all fingers & thumb into a 'C' shape, like holding a can.",
    "D":"Index finger points up; middle, ring, pinky curve to touch thumb.",
    "E":"All four fingertips curl down to touch the thumb.",
    "F":"Index finger and thumb form a circle; other three fingers up.",
    "G":"Index finger and thumb point sideways like a gun pointing right.",
    "H":"Index and middle fingers extended horizontally side by side.",
    "I":"Pinky finger up; all other fingers and thumb closed in a fist.",
    "J":"Pinky up, trace a 'J' downward — hold end position for detection.",
    "K":"Index and middle fingers in a V; thumb between them pointing up.",
    "L":"Index finger points straight up; thumb extends out forming an 'L'.",
    "M":"Three fingers fold over the thumb (like a triple knock).",
    "N":"Two fingers fold over the thumb (like a double knock).",
    "O":"All fingertips curve to meet the thumb tip, forming a circle.",
    "P":"Like 'K' rotated — index and middle point downward.",
    "Q":"Like 'G' pointing downward — index and thumb pinch downward.",
    "R":"Index and middle fingers crossed over each other.",
    "S":"Closed fist with thumb resting across the front of fingers.",
    "T":"Thumb tucked between index and middle fingers inside fist.",
    "U":"Index and middle fingers together, pointing straight up.",
    "V":"Index and middle fingers spread apart (peace/victory sign).",
    "W":"Index, middle, and ring fingers spread open upward like a 'W'.",
    "X":"Index finger bent into a hook (like a beckoning gesture).",
    "Y":"Thumb and pinky extended outward; other three fingers closed.",
    "Z":"Index finger extended, trace a 'Z' in air — hold end position.",
}

# ══════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════
_defaults={"app_started":False,"current_word":"","detection_history":deque(maxlen=10),
           "frame_buffer":[],"_last_prediction":None,"_last_confidence":0.0,"auto_cooldown_ctr":0}
for k,v in _defaults.items():
    if k not in st.session_state: st.session_state[k]=v

# ══════════════════════════════════════════════════════
#  MODEL LOADERS
# ══════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🧠  Loading AI model…")
def load_model():
    with open("model.pkl","rb") as f: return pickle.load(f)

@st.cache_resource(show_spinner="✋  Loading MediaPipe…")
def load_landmarker():
    opts=vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=0.20, min_hand_presence_confidence=0.20, min_tracking_confidence=0.20)
    return vision.HandLandmarker.create_from_options(opts)

model=load_model(); landmarker=load_landmarker()

# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════
def extract_features(lms):
    pts=np.array([[l.x,l.y,l.z] for l in lms],dtype=np.float32)
    pts-=pts[WRIST]
    hs=np.linalg.norm(pts[MIDDLE_MCP])
    if hs>1e-6: pts/=hs
    feats=pts.flatten().tolist()
    for a,b in DISTANCE_PAIRS: feats.append(round(float(np.linalg.norm(pts[a]-pts[b])),5))
    return feats

def smooth_prediction(pred,n=3):
    buf=st.session_state.frame_buffer; buf.append(pred)
    if len(buf)>n: buf.pop(0)
    return buf[0] if len(buf)==n and len(set(buf))==1 else None

def draw_skeleton(frame,lms):
    h,w=frame.shape[:2]; pts=[(int(l.x*w),int(l.y*h)) for l in lms]
    for a,b in HAND_CONNECTIONS: cv2.line(frame,pts[a],pts[b],(0,180,220),2)
    for i,(x,y) in enumerate(pts):
        cv2.circle(frame,(x,y),5,(0,255,136) if i==0 else (0,200,255),-1)
        cv2.circle(frame,(x,y),7,(255,255,255),1)
    return frame

def conf_color(c): return "#00FF88" if c>=80 else "#FF9500" if c>=60 else "#FF6B35"

def img_to_b64(path):
    if os.path.exists(path):
        with open(path,"rb") as f: return base64.b64encode(f.read()).decode()
    return ""

hand_b64=img_to_b64("67d807d703544680ff4f3b15__asl-alphabet.png")

# ══════════════════════════════════════════════════════
#  MEGA CSS
# ══════════════════════════════════════════════════════
MEGA_CSS="""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;600;700;800;900&display=swap');

/* ── CSS Variables ── */
:root {
  --green:   #00FF88;
  --blue:    #0099FF;
  --purple:  #9D4EDD;
  --orange:  #FF6B35;
  --mint:    #00D9FF;
  --navy:    #0A0E27;
  --charcoal:#1A1F3A;
  --dark:    #080B1A;
  --white:   #FFFFFF;
  --gray:    #8B92B3;
  --border:  rgba(0,255,136,0.18);
  --glass:   rgba(26,31,58,0.6);
  --glow:    0 0 30px rgba(0,255,136,0.25);
}

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  color: var(--white) !important;
}

/* ── App Background ── */
.stApp {
  background: linear-gradient(135deg, #0A0E27 0%, #0D0A20 40%, #0A1428 100%) !important;
  min-height: 100vh;
  position: relative;
}

/* SVG hand pattern overlay */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400' viewBox='0 0 200 200'%3E%3Cg opacity='0.04' fill='%2300FF88'%3E%3Cellipse cx='100' cy='160' rx='18' ry='25'/%3E%3Crect x='90' y='100' width='8' height='55' rx='4'/%3E%3Crect x='100' y='90' width='8' height='60' rx='4'/%3E%3Crect x='110' y='93' width='8' height='57' rx='4'/%3E%3Crect x='120' y='97' width='8' height='50' rx='4'/%3E%3Crect x='78' y='107' width='8' height='40' rx='4' transform='rotate(-15 78 107)'/%3E%3C/g%3E%3C/svg%3E");
  background-size: 280px 280px;
  pointer-events: none;
  z-index: 0;
  opacity: 0.6;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--charcoal); border-radius: 10px; }
::-webkit-scrollbar-thumb { background: var(--green); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--mint); }

/* ══════════════════════════════════
   STICKY NAVBAR
══════════════════════════════════ */
.navbar {
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 9999;
  height: 72px;
  background: rgba(10,14,39,0.92);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 2rem;
  animation: slideDown 0.5s ease;
}
@keyframes slideDown { from{opacity:0;transform:translateY(-100%)} to{opacity:1;transform:translateY(0)} }
.nav-logo {
  display: flex; align-items: center; gap: 0.7rem;
  font-family: 'Outfit', sans-serif;
  font-size: 1.3rem; font-weight: 800;
  color: var(--green) !important;
  text-shadow: 0 0 20px rgba(0,255,136,0.4);
  letter-spacing: -0.5px;
}
.nav-logo-icon { font-size: 1.8rem; }
.nav-right { display:flex; align-items:center; gap:1.2rem; }
.nav-badge {
  background: rgba(0,255,136,0.1);
  border: 1px solid rgba(0,255,136,0.3);
  border-radius: 100px;
  padding: 0.3rem 0.9rem;
  font-size: 0.72rem; font-weight: 700;
  color: var(--green); letter-spacing: 1px;
  text-transform: uppercase;
}
.nav-stat {
  font-size: 0.78rem; color: var(--gray);
  display: flex; align-items: center; gap: 0.4rem;
}
.nav-stat strong { color: var(--white); }

/* Compensate for navbar height */
.block-container { padding-top: 5rem !important; }

/* ══════════════════════════════════
   SIDEBAR
══════════════════════════════════ */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #12162E 0%, #0E1128 100%) !important;
  border-right: 1px solid var(--border) !important;
  top: 72px !important;
}
[data-testid="stSidebar"] > div { padding-top: 1.5rem !important; }

.sb-logo {
  font-family: 'Outfit', sans-serif;
  font-size: 1.2rem; font-weight: 800;
  background: linear-gradient(90deg, var(--green), var(--mint));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; margin-bottom: 2px;
}
.sb-ver { font-size: 0.7rem; color: var(--gray); letter-spacing: 1px; }
.sb-divider { height:1px; background: var(--border); margin: 1rem 0; }
.sb-section-title {
  font-size: 0.65rem; font-weight: 700; letter-spacing: 2.5px;
  text-transform: uppercase; color: var(--gray); margin-bottom: 0.8rem;
}
.sb-stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin-bottom:0.5rem; }
.sb-stat-box {
  background: rgba(0,255,136,0.05);
  border: 1px solid rgba(0,255,136,0.12);
  border-radius: 12px; padding: 0.7rem 0.4rem; text-align: center;
  transition: all 0.2s ease;
}
.sb-stat-box:hover { background: rgba(0,255,136,0.1); border-color: rgba(0,255,136,0.3); }
.sb-stat-num { font-family:'Outfit',sans-serif; font-size:1.15rem; font-weight:800; color:var(--green); }
.sb-stat-lbl { font-size:0.62rem; color:var(--gray); margin-top:2px; letter-spacing:0.5px; }

/* Slider styling */
[data-testid="stSlider"] > div > div > div { background: var(--green) !important; }
[data-testid="stSlider"] > div > div { background: rgba(255,255,255,0.08) !important; }

/* Toggle */
[data-testid="stToggle"] > label > div[data-checked="true"] { background-color: var(--green) !important; }

/* ══════════════════════════════════
   TABS
══════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(26,31,58,0.5) !important;
  border-radius: 16px !important;
  padding: 5px !important;
  border: 1px solid var(--border) !important;
  gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 12px !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  color: var(--gray) !important;
  transition: all 0.25s ease !important;
  padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,217,255,0.08)) !important;
  color: var(--green) !important;
  box-shadow: 0 0 15px rgba(0,255,136,0.1) !important;
}

/* ══════════════════════════════════
   DETECTION PANEL
══════════════════════════════════ */
.cam-container {
  border: 2px solid var(--green);
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 0 40px rgba(0,255,136,0.25), 0 0 80px rgba(0,255,136,0.08);
  background: #000;
  position: relative;
  transition: box-shadow 0.3s ease;
}
.cam-container:hover { box-shadow: 0 0 60px rgba(0,255,136,0.35), 0 0 120px rgba(0,255,136,0.12); }

/* Letter Display */
.letter-hero {
  background: linear-gradient(135deg, rgba(0,255,136,0.05) 0%, rgba(0,153,255,0.04) 100%);
  border: 1.5px solid var(--border);
  border-radius: 24px;
  padding: 1.5rem 1rem;
  text-align: center;
  min-height: 180px;
  display: flex; flex-direction: column;
  justify-content: center; align-items: center;
  position: relative; overflow: hidden;
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 32px rgba(0,255,136,0.08);
}
.letter-hero::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(circle at 50% 30%, rgba(0,255,136,0.06), transparent 65%);
}
.letter-main {
  font-family: 'Outfit', sans-serif;
  font-size: clamp(4rem,10vw,7rem);
  font-weight: 900; line-height: 1; color: var(--green);
  text-shadow: 0 0 30px rgba(0,255,136,0.7), 0 0 60px rgba(0,255,136,0.4), 0 0 100px rgba(0,255,136,0.2);
  animation: letterPop 0.35s cubic-bezier(0.68,-0.55,0.265,1.55) both;
  position: relative; z-index: 1;
}
@keyframes letterPop { from{transform:scale(0.3);opacity:0} to{transform:scale(1);opacity:1} }
.letter-dim { opacity: 0.25; animation: none !important; }
.no-detect { color:var(--gray); font-size:0.88rem; line-height:1.7; position:relative;z-index:1; }
.no-detect-icon { font-size:2rem; margin-bottom:0.5rem; display:block; }

/* Confidence bar */
.conf-wrap { width:100%; margin-top:0.8rem; position:relative;z-index:1; }
.conf-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:5px; }
.conf-label { font-size:0.65rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--gray); }
.conf-pct { font-size:0.85rem; font-weight:800; }
.conf-track { background:rgba(255,255,255,0.06); border-radius:10px; height:7px; overflow:hidden; }
.conf-fill {
  height:100%; border-radius:10px;
  transition: width 0.5s ease, background-color 0.5s ease;
  box-shadow: 0 0 12px currentColor;
}

/* ══════════════════════════════════
   WORD BUILDER
══════════════════════════════════ */
.wb-panel {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 1.4rem;
  backdrop-filter: blur(12px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.wb-title {
  font-size: 0.65rem; font-weight: 800; letter-spacing: 3px;
  text-transform: uppercase; color: var(--green);
  margin-bottom: 0.8rem;
}
.word-display {
  background: rgba(0,0,0,0.4);
  border: 1.5px solid rgba(0,255,136,0.3);
  border-radius: 14px;
  padding: 1rem 1.2rem;
  min-height: 70px;
  font-family: 'Outfit', sans-serif;
  font-size: 1.6rem; font-weight: 800;
  color: var(--green);
  letter-spacing: 4px;
  display: flex; align-items: center; flex-wrap: wrap;
  word-break: break-all;
  margin-bottom: 0.8rem;
  position: relative;
}
.word-cursor {
  display: inline-block; width: 2.5px; height: 1em;
  background: var(--green);
  border-radius: 2px;
  margin-left: 4px;
  animation: blink 1s step-end infinite;
  vertical-align: middle;
  box-shadow: 0 0 8px var(--green);
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
.word-placeholder { color:#334155; font-size:0.85rem; font-weight:400; letter-spacing:0; font-style:italic; }

/* Buttons */
.stButton > button {
  background: rgba(26,31,58,0.85) !important;
  border: 1px solid rgba(0,255,136,0.35) !important;
  color: var(--white) !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  transition: all 0.25s ease !important;
  padding: 0.55rem 0.4rem !important;
  white-space: nowrap !important;
  backdrop-filter: blur(8px) !important;
}
.stButton > button:hover {
  transform: scale(1.04) translateY(-2px) !important;
  box-shadow: 0 6px 25px rgba(0,255,136,0.18) !important;
  border-color: var(--green) !important;
  background: rgba(0,255,136,0.08) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }

/* History badges */
.hist-wrap { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.hist-badge {
  width: 40px; height: 40px;
  background: var(--green);
  color: #000; font-weight: 900;
  font-size: 1rem;
  border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  font-family: 'Outfit', sans-serif;
  box-shadow: 0 0 12px rgba(0,255,136,0.4);
  animation: fadeInLeft 0.3s ease both;
  transition: transform 0.2s ease;
}
.hist-badge:hover { transform: scale(1.15); }
@keyframes fadeInLeft { from{opacity:0;transform:translateX(-12px)} to{opacity:1;transform:translateX(0)} }

/* Section headers */
.sec-label {
  font-size:0.63rem; font-weight:700; letter-spacing:2.5px;
  text-transform:uppercase; color:var(--gray); margin-bottom:0.4rem;
}

/* Metric cards */
.metric-grid {
  display:grid; grid-template-columns:repeat(4,1fr);
  gap:0.7rem; margin-bottom:1.2rem;
}
.metric-card {
  background: var(--glass);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 16px; padding:1rem 0.6rem; text-align:center;
  backdrop-filter:blur(10px);
  transition: all 0.25s ease;
  animation: fadeUp 0.6s ease both;
}
.metric-card:nth-child(1){animation-delay:.05s}
.metric-card:nth-child(2){animation-delay:.1s}
.metric-card:nth-child(3){animation-delay:.15s}
.metric-card:nth-child(4){animation-delay:.2s}
.metric-card:hover { border-color:rgba(0,255,136,0.25); transform:translateY(-3px); box-shadow: var(--glow); }
.metric-num { font-family:'Outfit',sans-serif; font-size:1.7rem; font-weight:900; color:var(--green); line-height:1; }
.metric-lbl { font-size:0.66rem; color:var(--gray); margin-top:4px; letter-spacing:0.5px; }

@keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }

/* ══════════════════════════════════
   LEARN SIGNS TAB
══════════════════════════════════ */
.sign-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:0.9rem; }
@media(max-width:900px){.sign-grid{grid-template-columns:repeat(2,1fr);}}

.sign-card {
  background: var(--glass);
  border: 1px solid rgba(0,255,136,0.14);
  border-radius: 18px; padding:1.4rem 0.9rem;
  text-align:center; cursor:default;
  backdrop-filter:blur(10px);
  transition: all 0.28s ease;
  position:relative; overflow:hidden;
  animation: fadeUp 0.5s ease both;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.sign-card::after {
  content: attr(data-letter);
  position:absolute; bottom:-15px; right:-5px;
  font-size:6rem; font-weight:900;
  color:rgba(0,255,136,0.03);
  font-family:'Outfit',sans-serif;
  pointer-events:none;
}
.sign-card:hover {
  border-color: var(--green);
  transform: translateY(-5px) scale(1.02);
  box-shadow: 0 0 25px rgba(0,255,136,0.2), 0 12px 40px rgba(0,0,0,0.4);
  background: rgba(0,255,136,0.06);
}
.sign-letter {
  font-family:'Outfit',sans-serif;
  font-size:3.2rem; font-weight:900;
  color:var(--green); line-height:1; margin-bottom:0.5rem;
  text-shadow: 0 0 20px rgba(0,255,136,0.5);
  transition: text-shadow 0.3s ease;
}
.sign-card:hover .sign-letter { text-shadow: 0 0 35px rgba(0,255,136,0.8); }
.sign-desc { font-size:0.73rem; color:var(--gray); line-height:1.55; }

/* ══════════════════════════════════
   ABOUT TAB
══════════════════════════════════ */
.about-card {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: 20px; padding:1.6rem;
  backdrop-filter:blur(10px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.2);
  margin-bottom:1rem;
  animation: fadeUp 0.6s ease both;
}
.about-card h3 {
  font-family:'Outfit',sans-serif;
  font-size:1.1rem; font-weight:700;
  color:var(--green); margin-bottom:0.8rem;
}
.pipeline-step {
  display:flex; gap:0.8rem; align-items:flex-start;
  margin-bottom:0.7rem; padding:0.6rem;
  border-radius:10px;
  background:rgba(0,255,136,0.03);
  border-left:2px solid rgba(0,255,136,0.3);
  transition: all 0.2s ease;
}
.pipeline-step:hover { background:rgba(0,255,136,0.07); border-left-color:var(--green); }
.step-num {
  width:28px; height:28px; min-width:28px;
  background:var(--green); color:#000;
  border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-weight:900; font-size:0.78rem;
}
.step-txt { font-size:0.85rem; color:#cbd5e1; line-height:1.5; }
.step-txt strong { color:var(--white); }

/* Tables */
table { border-collapse:collapse; width:100%; }
th,td { border:1px solid rgba(255,255,255,0.07); padding:9px 13px; font-size:0.84rem; }
th { background:rgba(0,255,136,0.08); color:var(--green); font-weight:700; letter-spacing:0.5px; }
tr:hover { background:rgba(255,255,255,0.02); }

/* Expander */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  background: var(--glass) !important;
  backdrop-filter: blur(10px) !important;
  margin-bottom: 0.5rem !important;
}

/* Text input */
[data-testid="stTextInput"] input {
  background: rgba(0,0,0,0.3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--white) !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--green) !important;
  box-shadow: 0 0 0 2px rgba(0,255,136,0.15) !important;
}

/* Camera widget */
[data-testid="stCameraInput"] > div { border-radius:18px !important; overflow:hidden; }
[data-testid="stCameraInput"] > div > div {
  border: 2px solid var(--green) !important;
  border-radius:18px !important;
  box-shadow: 0 0 30px rgba(0,255,136,0.2) !important;
}

/* Page section heading */
.page-title {
  font-family:'Outfit',sans-serif;
  font-size:1.8rem; font-weight:800;
  color:var(--white); margin-bottom:0.25rem;
  letter-spacing:-0.5px;
}
.page-sub { font-size:0.88rem; color:var(--gray); margin-bottom:1.2rem; line-height:1.6; }

/* Footer */
.footer {
  text-align:center; padding:2rem 0 1rem;
  color:var(--gray); font-size:0.78rem;
  border-top:1px solid rgba(255,255,255,0.05);
  margin-top:2rem;
}
.footer strong { color:var(--green); }
</style>
"""

st.markdown(MEGA_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  LANDING PAGE
# ══════════════════════════════════════════════════════
if not st.session_state.app_started:

    _bg=f"url('data:image/png;base64,{hand_b64}')" if hand_b64 else "none"

    components.html(f"""
<!DOCTYPE html><html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=Outfit:wght@500;700;900&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Inter',sans-serif;background:#0A0E27;color:#fff;min-height:100vh;overflow-x:hidden;}}

/* Animated radial BG */
.bg{{position:fixed;inset:0;z-index:0;
  background:
    radial-gradient(ellipse 65% 55% at 12% 18%, rgba(0,255,136,0.11) 0%,transparent 60%),
    radial-gradient(ellipse 50% 45% at 88% 78%, rgba(157,78,221,0.12) 0%,transparent 60%),
    radial-gradient(ellipse 40% 35% at 55% 5%, rgba(0,217,255,0.08) 0%,transparent 50%),
    linear-gradient(135deg,#0A0E27 0%,#0D0A20 50%,#0A1428 100%);
}}
/* Grid */
.grid{{position:fixed;inset:0;z-index:0;
  background-image:linear-gradient(rgba(0,255,136,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,136,0.04) 1px,transparent 1px);
  background-size:55px 55px;
}}
/* Hand panels */
.hand-l,.hand-r{{position:fixed;top:0;bottom:0;width:280px;z-index:1;pointer-events:none;
  background-image:{_bg};background-size:cover;background-position:center;opacity:.09;}}
.hand-l{{left:0;-webkit-mask-image:linear-gradient(to right,rgba(0,0,0,.8),transparent);mask-image:linear-gradient(to right,rgba(0,0,0,.8),transparent);}}
.hand-r{{right:0;transform:scaleX(-1);-webkit-mask-image:linear-gradient(to left,rgba(0,0,0,.8),transparent);mask-image:linear-gradient(to left,rgba(0,0,0,.8),transparent);}}

/* Floating letter chips */
.chip{{position:fixed;font-family:'Outfit',sans-serif;font-weight:900;pointer-events:none;z-index:1;
  color:rgba(0,255,136,.15);animation:rise linear infinite;}}
@keyframes rise{{0%{{transform:translateY(105vh);opacity:0;}}8%{{opacity:1;}}90%{{opacity:.5;}}100%{{transform:translateY(-10vh);opacity:0;}}}}

/* Navbar */
.nav{{position:fixed;top:0;left:0;right:0;z-index:100;
  height:68px;display:flex;align-items:center;justify-content:space-between;
  padding:0 2.5rem;
  background:rgba(10,14,39,.88);
  backdrop-filter:blur(20px);
  border-bottom:1px solid rgba(0,255,136,.18);
  animation:slideDown .5s ease;
}}
@keyframes slideDown{{from{{transform:translateY(-100%);opacity:0}}to{{transform:translateY(0);opacity:1}}}}
.nav-l{{display:flex;align-items:center;gap:.7rem;font-family:'Outfit',sans-serif;font-size:1.25rem;font-weight:900;color:#00FF88;text-shadow:0 0 20px rgba(0,255,136,.4);}}
.nav-r{{display:flex;gap:1rem;align-items:center;}}
.badge{{background:rgba(0,255,136,.09);border:1px solid rgba(0,255,136,.28);border-radius:100px;padding:.3rem .9rem;font-size:.7rem;font-weight:700;color:#00FF88;letter-spacing:1.5px;text-transform:uppercase;}}
.dot{{width:7px;height:7px;background:#00FF88;border-radius:50%;display:inline-block;margin-right:.35rem;animation:pulse 1.5s ease infinite;}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.3;transform:scale(.6)}}}}

/* Main content */
.page{{position:relative;z-index:2;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:5rem 1.5rem 4rem;}}

.eyebrow{{
  display:inline-flex;align-items:center;gap:.45rem;
  background:rgba(0,255,136,.07);border:1px solid rgba(0,255,136,.25);
  border-radius:100px;padding:.4rem 1.2rem;
  font-size:.72rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#00FF88;
  margin-bottom:1.8rem;animation:fadeD .7s ease both;
}}
.title{{
  font-family:'Outfit',sans-serif;font-size:clamp(2.8rem,8vw,6rem);
  font-weight:900;line-height:1.04;letter-spacing:-2.5px;margin-bottom:1.3rem;
  animation:fadeD .8s .1s ease both;
}}
.g1{{background:linear-gradient(135deg,#00FF88,#00D9FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}}
.g2{{background:linear-gradient(135deg,#9D4EDD,#FF6B9D);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}}
.subtitle{{font-size:clamp(.95rem,2vw,1.18rem);color:#94a3b8;max-width:600px;line-height:1.8;margin-bottom:2.5rem;animation:fadeD .8s .2s ease both;}}

.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:1.1rem;max-width:840px;width:100%;margin-bottom:2.5rem;animation:fadeU .9s .3s ease both;}}
@media(max-width:640px){{.cards{{grid-template-columns:1fr;}}}}
.card{{
  background:rgba(26,31,58,.5);border:1px solid rgba(0,255,136,.12);
  border-radius:20px;padding:1.6rem 1.1rem;
  backdrop-filter:blur(12px);position:relative;overflow:hidden;
  transition:all .3s ease;
}}
.card:hover{{border-color:rgba(0,255,136,.3);transform:translateY(-6px);box-shadow:0 14px 40px rgba(0,255,136,.1);background:rgba(0,255,136,.06);}}
.card-icon{{font-size:2.1rem;margin-bottom:.6rem;}}
.card-title{{font-weight:700;font-size:.92rem;color:#f1f5f9;margin-bottom:.4rem;}}
.card-desc{{font-size:.76rem;color:#64748b;line-height:1.6;}}

.stats{{display:flex;gap:2rem;flex-wrap:wrap;justify-content:center;animation:fadeU .9s .45s ease both;}}
.st-item{{display:flex;align-items:center;gap:.4rem;font-size:.83rem;color:#94a3b8;}}
.st-item strong{{color:#00FF88;font-weight:700;}}

@keyframes fadeD{{from{{opacity:0;transform:translateY(-16px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes fadeU{{from{{opacity:0;transform:translateY(16px)}}to{{opacity:1;transform:translateY(0)}}}}
</style>
</head>
<body>
<div class="bg"></div>
<div class="grid"></div>
<div class="hand-l"></div>
<div class="hand-r"></div>

<script>
const L='ABCDEFGHIJKLMNOPQRSTUVWXYZ';
for(let i=0;i<20;i++){{
  const el=document.createElement('div');el.className='chip';
  el.textContent=L[Math.floor(Math.random()*26)];
  el.style.left=Math.random()*100+'vw';
  el.style.animationDuration=(10+Math.random()*20)+'s';
  el.style.animationDelay=(Math.random()*25)+'s';
  el.style.fontSize=(.8+Math.random()*1.5)+'rem';
  document.body.appendChild(el);
}}
</script>

<nav class="nav">
  <div class="nav-l"><span style="font-size:1.9rem">🤟</span> Sign Language AI</div>
  <div class="nav-r">
    <span class="badge"><span class="dot"></span>99.73% Accurate</span>
    <span style="font-size:.75rem;color:#64748b;">MediaPipe × Random Forest</span>
  </div>
</nav>

<div class="page">
  <div class="eyebrow"><span class="dot"></span>Production Grade · Real-Time · 26 Letters</div>
  <h1 class="title">
    <span class="g1">Sign Language</span><br>
    <span style="color:#f1f5f9;">meets&nbsp;</span><span class="g2">Artificial Intelligence</span>
  </h1>
  <p class="subtitle">
    Show any ASL hand sign to your webcam and watch our AI recognize it instantly.<br>
    Trained on <strong style="color:#00FF88">126,718 samples</strong> with 99.73% accuracy — no GPU required.
  </p>
  <div class="cards">
    <div class="card"><div class="card-icon">⚡</div><div class="card-title">Real-Time Detection</div><div class="card-desc">Sub-50ms per frame using MediaPipe hand landmarks — runs on any CPU.</div></div>
    <div class="card"><div class="card-icon">🎯</div><div class="card-title">99.73% Accuracy</div><div class="card-desc">126K augmented samples · wrist-normalized · 88-dimensional features.</div></div>
    <div class="card"><div class="card-icon">✍️</div><div class="card-title">Word Builder</div><div class="card-desc">Build words letter by letter with auto-type, TTS voice output & history.</div></div>
  </div>
  <div class="stats">
    <div class="st-item">🔤 <strong>26</strong>&nbsp;ASL Letters</div>
    <div class="st-item">✋ <strong>21</strong>&nbsp;Hand Landmarks</div>
    <div class="st-item">📸 <strong>126K</strong>&nbsp;Samples</div>
    <div class="st-item">🌳 <strong>500</strong>&nbsp;Trees</div>
    <div class="st-item">⚡ <strong>&lt;50ms</strong>&nbsp;Latency</div>
  </div>
</div>
</body></html>
""", height=680, scrolling=False)

    st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)
    _, cb, _ = st.columns([2,1.5,2])
    with cb:
        if st.button("🤟  Launch App  →", use_container_width=True, type="primary"):
            st.session_state.app_started = True
            st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════
#  STICKY NAVBAR (main app)
# ══════════════════════════════════════════════════════
last_pred = st.session_state.get("_last_prediction","–")
last_conf = st.session_state.get("_last_confidence", 0.0)
pred_display = f"{last_pred} · {last_conf:.0f}%" if last_pred and last_pred != "–" else "Waiting…"

st.markdown(f"""
<div class="navbar">
  <div class="nav-logo">
    <span class="nav-logo-icon">🤟</span>
    Sign Language AI
  </div>
  <div class="nav-right">
    <div class="nav-stat">Detecting: <strong>{pred_display}</strong></div>
    <div class="nav-badge">v4.0</div>
    <div class="nav-badge" style="background:rgba(0,217,255,.08);border-color:rgba(0,217,255,.3);color:#00D9FF;">99.73% Acc</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div class="sb-logo">🤟 Sign Language AI</div>
<div class="sb-ver">v4.0 · ULTRA PREMIUM</div>
<div class="sb-divider"></div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title">📊 Project Stats</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="sb-stat-grid">
  <div class="sb-stat-box"><div class="sb-stat-num">99.7%</div><div class="sb-stat-lbl">Accuracy</div></div>
  <div class="sb-stat-box"><div class="sb-stat-num">26</div><div class="sb-stat-lbl">Letters</div></div>
  <div class="sb-stat-box"><div class="sb-stat-num">126K</div><div class="sb-stat-lbl">Samples</div></div>
  <div class="sb-stat-box"><div class="sb-stat-num">&lt;50ms</div><div class="sb-stat-lbl">Latency</div></div>
</div>
<div class="sb-divider"></div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title">⚙️ Detection Settings</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Min Confidence %", 10, 95, 40, 5,
        help="Lower if letters aren't being detected")
    smooth_frames  = st.slider("Smoothing Frames", 1, 8, 2, 1,
        help="Same letter required N times before committing")

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section-title">🤖 Auto-Type Mode</div>', unsafe_allow_html=True)
    auto_type = st.toggle("Enable Auto-Type", value=False,
        help="Automatically add letters to word when held steady")
    if auto_type:
        auto_cooldown = st.slider("Hold Frames", 5, 40, 15, 5,
            help="Frames to hold before auto-adding next letter")
        st.caption("💡 Hold each sign still ~0.5s to auto-type")
    else:
        auto_cooldown = 15

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    with st.expander("🛠 Tech Stack"):
        st.markdown("""
| Component | Detail |
|-----------|--------|
| Hand AI | MediaPipe 0.10.11 |
| Classifier | Random Forest |
| Trees | 500 |
| Features | 88 dims |
| Framework | Streamlit |
| Language | Python 3.11 |
""")

    with st.expander("🚀 How to Use"):
        st.markdown("""
1. Go to **🎥 Live Detection** tab
2. Click **📷 Take Photo**
3. Show an **ASL hand sign** clearly
4. AI detects & displays the letter
5. Click **➕ Add Letter** to build words
6. Use **⎵ Space / ⌫ Delete** to edit
""")

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    if st.button("⬅️ Back to Home", use_container_width=True):
        st.session_state.app_started = False; st.rerun()

    st.markdown("""
<div style="text-align:center;padding:1rem 0;font-size:.7rem;color:#334155;">
  Built by <strong style="color:#00FF88">Vishal Ponia</strong><br>
  <span style="color:#1e293b">Computer Vision × ML Engineer</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🎥  Live Detection","📖  Learn Signs","ℹ️  About Project"])

# ────────────────────────────────────
#  TAB 1 — LIVE DETECTION
# ────────────────────────────────────
with tab1:

    # Metric row
    st.markdown("""
<div class="metric-grid">
  <div class="metric-card"><div class="metric-num">99.7%</div><div class="metric-lbl">Accuracy</div></div>
  <div class="metric-card"><div class="metric-num">26</div><div class="metric-lbl">Letters A–Z</div></div>
  <div class="metric-card"><div class="metric-num">&lt;50ms</div><div class="metric-lbl">Latency</div></div>
  <div class="metric-card"><div class="metric-num">88</div><div class="metric-lbl">Features</div></div>
</div>
""", unsafe_allow_html=True)

    cam_col, panel_col = st.columns([3, 2], gap="large")

    # ── Camera column ──
    with cam_col:
        st.markdown('<div class="sec-label">📷 Live Webcam Feed</div>', unsafe_allow_html=True)
        st.caption("Allow camera access · Keep hand well-lit & centred in frame")

        camera_input = st.camera_input("webcam", label_visibility="collapsed", key="cam")
        annotated_ph = st.empty()

        with st.expander("🔬 Debug Info"):
            debug_ph = st.empty()

    # ── Detection panel column ──
    with panel_col:
        col_letter, col_conf = st.columns(2)
        with col_letter:
            st.markdown('<div class="sec-label">🔤 Detected Letter</div>', unsafe_allow_html=True)
            letter_ph = st.empty()
        with col_conf:
            st.markdown('<div class="sec-label">🎯 Confidence</div>', unsafe_allow_html=True)
            conf_ph = st.empty()

        st.markdown("<br>", unsafe_allow_html=True)

        # Word builder
        st.markdown("""
<div class="wb-title">✍️ Word Builder</div>
""", unsafe_allow_html=True)
        word_ph = st.empty()

        # Add letter button — full width
        if st.button("➕  Add Detected Letter", use_container_width=True, key="btn_add",
                     help="Append the currently detected letter to your word"):
            p = st.session_state.get("_last_prediction")
            if p: st.session_state.current_word += p

        wb1, wb2, wb3 = st.columns(3)
        with wb1:
            if st.button("⌫ Delete", use_container_width=True, key="btn_back"):
                st.session_state.current_word = st.session_state.current_word[:-1]
        with wb2:
            if st.button("⎵ Space", use_container_width=True, key="btn_space"):
                st.session_state.current_word += " "
        with wb3:
            if st.button("🗑 Clear", use_container_width=True, key="btn_clear"):
                st.session_state.current_word = ""

        speak_c, copy_c = st.columns([1, 2])
        with speak_c:
            speak_btn = st.button("🔊 Speak", use_container_width=True, key="btn_speak")
        with copy_c:
            if st.session_state.current_word:
                st.text_input("cp", value=st.session_state.current_word,
                              label_visibility="collapsed", key="copy_box",
                              help="Ctrl+A, Ctrl+C to copy")

        if speak_btn and st.session_state.current_word:
            safe = st.session_state.current_word.replace("'","\\'").replace('"','\\"')
            st.html(f"<script>var u=new SpeechSynthesisUtterance('{safe}');u.rate=0.9;window.speechSynthesis.speak(u);</script>")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">📜 Last 10 Detections</div>', unsafe_allow_html=True)
        hist_ph = st.empty()

    # ── Process camera frame ──
    if camera_input is not None:
        arr   = np.frombuffer(camera_input.getvalue(), np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h0,w0 = frame.shape[:2]
        if max(h0,w0)<640:
            sc=640/max(h0,w0); frame=cv2.resize(frame,(int(w0*sc),int(h0*sc)),interpolation=cv2.INTER_LINEAR)
        rgb=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),dtype=np.uint8)
        result=landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
        num_hands=len(result.hand_landmarks) if result.hand_landmarks else 0

        if num_hands>0:
            hand=result.hand_landmarks[0]
            feats=extract_features(hand)
            if len(feats)!=88:
                letter_ph.markdown(f'<div class="letter-hero"><div class="no-detect">⚠️ Feature error ({len(feats)}/88)</div></div>',unsafe_allow_html=True)
            else:
                prediction=model.predict([feats])[0]
                confidence=float(model.predict_proba([feats]).max()*100)
                st.session_state._last_prediction=prediction
                st.session_state._last_confidence=confidence
                cc=conf_color(confidence)

                if confidence>=conf_threshold:
                    committed=smooth_prediction(prediction,smooth_frames)
                    if committed:
                        st.session_state.detection_history.append(committed)
                        if auto_type:
                            if st.session_state.auto_cooldown_ctr==0:
                                st.session_state.current_word+=committed
                                st.session_state.auto_cooldown_ctr=auto_cooldown
                            else: st.session_state.auto_cooldown_ctr-=1
                        else: st.session_state.auto_cooldown_ctr=0

                    letter_ph.markdown(
                        f'<div class="letter-hero">'
                        f'<div class="letter-main">{prediction}</div>'
                        f'<div class="conf-wrap">'
                        f'<div class="conf-header"><span class="conf-label">Confidence</span><span class="conf-pct" style="color:{cc}">{confidence:.1f}%</span></div>'
                        f'<div class="conf-track"><div class="conf-fill" style="width:{confidence}%;background:{cc};"></div></div>'
                        f'</div></div>',unsafe_allow_html=True)
                else:
                    letter_ph.markdown(
                        f'<div class="letter-hero">'
                        f'<div class="letter-main letter-dim">{prediction}</div>'
                        f'<div style="font-size:.68rem;color:#475569;margin-top:.4rem;">'
                        f'Conf {confidence:.0f}% &lt; threshold {conf_threshold}%</div>'
                        f'<div class="conf-wrap">'
                        f'<div class="conf-track"><div class="conf-fill" style="width:{confidence}%;background:{cc};"></div></div>'
                        f'</div></div>',unsafe_allow_html=True)

                conf_ph.markdown(
                    f'<div class="letter-hero" style="min-height:180px;">'
                    f'<div style="font-size:.63rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#475569;margin-bottom:.4rem;">Score</div>'
                    f'<div style="font-family:Outfit,sans-serif;font-size:2.8rem;font-weight:900;color:{cc};line-height:1;">{confidence:.0f}%</div>'
                    f'<div class="conf-wrap">'
                    f'<div class="conf-track"><div class="conf-fill" style="width:{confidence}%;background:{cc};"></div></div>'
                    f'</div>'
                    f'<div style="font-size:.65rem;color:#475569;margin-top:.4rem;">Threshold: {conf_threshold}%</div>'
                    f'</div>',unsafe_allow_html=True)

                annotated=draw_skeleton(frame.copy(),hand)
                annotated_ph.image(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB),
                                   use_container_width=True,caption="Hand skeleton — 21 MediaPipe landmarks")

                debug_ph.markdown(f"""
| Field | Value |
|-------|-------|
| Hands | `{num_hands}` |
| Prediction | `{prediction}` |
| Confidence | `{confidence:.1f}%` |
| Threshold | `{conf_threshold}%` |
| Buffer | `{st.session_state.frame_buffer}` |
| Frame | `{frame.shape}` |
""")
        else:
            st.session_state._last_prediction=None
            st.session_state.frame_buffer=[]; st.session_state.auto_cooldown_ctr=0

            letter_ph.markdown(
                '<div class="letter-hero"><span class="no-detect-icon">✋</span>'
                '<div class="no-detect"><strong style="color:#cbd5e1;font-size:.9rem">No hand detected</strong><br>'
                '<span style="font-size:.74em">Ensure hand is lit &amp; fully in frame</span></div></div>',
                unsafe_allow_html=True)
            conf_ph.markdown(
                '<div class="letter-hero"><span class="no-detect-icon">🔍</span>'
                '<div class="no-detect"><strong style="color:#cbd5e1;font-size:.9rem">Scanning…</strong><br>'
                '<span style="font-size:.74em">Try brighter lighting</span></div></div>',
                unsafe_allow_html=True)
            annotated_ph.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),use_container_width=True)
            debug_ph.markdown(f"| Hands | `0` | Frame | `{frame.shape}` |")
    else:
        letter_ph.markdown(
            '<div class="letter-hero"><span class="no-detect-icon">📷</span>'
            '<div class="no-detect"><strong style="color:#cbd5e1;font-size:.9rem">Click · Take Photo</strong><br>'
            '<span style="font-size:.74em">Then show your ASL hand sign</span></div></div>',
            unsafe_allow_html=True)
        conf_ph.markdown(
            '<div class="letter-hero"><span class="no-detect-icon">🎯</span>'
            '<div class="no-detect"><strong style="color:#cbd5e1;font-size:.9rem">AI Ready</strong><br>'
            '<span style="font-size:.74em">99.73% accuracy · A–Z</span></div></div>',
            unsafe_allow_html=True)

    # Word display
    w=st.session_state.current_word
    if w:
        word_ph.markdown(f'<div class="word-display">{w}<span class="word-cursor"></span></div>',unsafe_allow_html=True)
    else:
        word_ph.markdown('<div class="word-display"><span class="word-placeholder">Your word will appear here…</span><span class="word-cursor"></span></div>',unsafe_allow_html=True)

    # History badges
    hist=list(st.session_state.detection_history)
    if hist:
        badges="".join(f'<span class="hist-badge">{l}</span>' for l in hist)
        hist_ph.markdown(f'<div class="hist-wrap">{badges}</div>',unsafe_allow_html=True)
    else:
        hist_ph.markdown('<span style="color:#334155;font-size:.84em;font-style:italic;">Detected letters will appear here…</span>',unsafe_allow_html=True)

# ────────────────────────────────────
#  TAB 2 — LEARN SIGNS
# ────────────────────────────────────
with tab2:
    st.markdown('<div class="page-title">📖 Learn the ASL Alphabet</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Hover over any card to see how to form that sign. Practice in good lighting before live detection.</div>', unsafe_allow_html=True)

    search=st.text_input("🔍  Search letter or keyword…","",key="search",label_visibility="collapsed",
                         placeholder="🔍  Search letter or keyword…").upper().strip()

    letters=[L for L in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             if not search or search in L or search in SIGN_DESCRIPTIONS[L].upper()]

    # Build the full grid as one HTML block for reliable rendering
    N=4
    for i in range(0,len(letters),N):
        row=letters[i:i+N]
        cols=st.columns(N)
        for col,letter in zip(cols,row):
            col.markdown(
                f'<div class="sign-card" data-letter="{letter}">'
                f'<div class="sign-letter">{letter}</div>'
                f'<div class="sign-desc">{SIGN_DESCRIPTIONS[letter]}</div>'
                f'</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="sb-divider" style="margin:1.5rem 0"></div>', unsafe_allow_html=True)
    st.markdown("""
### 💡 Tips for Best Detection
- **Hold still** — keep your hand steady for 2–3 frames  
- **Good lighting** — face a bright light (window or lamp)  
- **Centred hand** — position in the middle of the frame  
- **Palm outward** — most ASL signs face the camera  
- **J & Z** — involve motion; hold the *final position* for detection  
- **Lower threshold** — use the sidebar slider if signs aren't detecting  
""")

# ────────────────────────────────────
#  TAB 3 — ABOUT
# ────────────────────────────────────
with tab3:
    st.markdown('<div class="page-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">A production-ready real-time ASL recognition system — from raw dataset to live inference, built for accessibility and portfolio demonstration.</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<div class="about-card">', unsafe_allow_html=True)
        st.markdown("### 🔬 How It Works")
        steps=[
            ("Frame Capture","Webcam snapshot decoded from JPEG bytes to NumPy array via OpenCV."),
            ("Hand Detection","MediaPipe HandLandmarker finds 21 3-D keypoints in real-time."),
            ("Feature Extraction","Wrist-centered, scale-normalized coords + 25 pairwise distances = <strong>88-dim vector</strong>."),
            ("Classification","RandomForestClassifier (500 trees) predicts ASL letter + confidence."),
            ("Confidence Gate","Predictions below user threshold are silently discarded."),
            ("Smoothing","Same letter required N consecutive frames to avoid flickering."),
        ]
        for i,(title,desc) in enumerate(steps,1):
            st.markdown(f'<div class="pipeline-step"><div class="step-num">{i}</div><div class="step-txt"><strong>{title}</strong> — {desc}</div></div>',unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="about-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Model Performance")
        st.markdown("""
| Metric | Value |
|--------|-------|
| **Test Accuracy** | **99.73%** |
| Training Samples | 126,718 |
| Feature Dims | 88 |
| Trees | 500 |
| Inference Time | ~5–10 ms |
| **Total Latency** | **< 50 ms / frame** |
""")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="about-card">', unsafe_allow_html=True)
        st.markdown("### 🛠 Tech Stack")
        st.markdown("""
| Component | Technology |
|-----------|-----------|
| Hand Detection | MediaPipe 0.10.11 |
| Classifier | scikit-learn RF |
| Video | OpenCV 4.8.1 |
| Web UI | Streamlit |
| Language | Python 3.11 |
| Numerics | NumPy 1.26.4 |
""")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="about-card">', unsafe_allow_html=True)
        st.markdown("### 🌍 Social Impact")
        st.markdown("""
- **18 lakh+** deaf & hard-of-hearing individuals in India  
- No special hardware — just a webcam & browser  
- Real-time text output from hand gestures  
- Foundation for full sentence + ISL support  
""")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="about-card">', unsafe_allow_html=True)
        st.markdown("### 🗂 Dataset")
        st.markdown("""
- **Source**: [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Images used**: 3,000 per letter × 26 = 78K  
- **With augmentation**: 126,718 samples  
- **Format**: 200×200 JPG  
""")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("🎙 Interview Talking Points"):
        st.markdown("""
> *"I built an end-to-end AI pipeline using MediaPipe + scikit-learn that converts
> hand gestures to text in real-time at <50ms latency with 99.73% accuracy —
> all on CPU, no GPU required. Trained on 126K samples with custom feature engineering."*

**Skills demonstrated:**
- **Computer Vision** — MediaPipe landmark extraction, feature normalization, skeleton overlay
- **ML Pipeline** — dataset → augmentation → normalization → training → evaluation → inference
- **Real-time Processing** — <50ms per frame with 88-dimensional feature vectors
- **Product Thinking** — confidence gates, smoothing, word builder, auto-type, TTS
- **Social Impact** — accessibility tooling for the deaf community
""")

    st.markdown("""
<div class="footer">
  <strong>Sign Language AI v4.0</strong> &nbsp;·&nbsp;
  Built with ❤️ by <strong>Vishal Ponia</strong> &nbsp;·&nbsp;
  MediaPipe × Random Forest × Streamlit &nbsp;·&nbsp; MIT License
</div>
""", unsafe_allow_html=True)
