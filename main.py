"""
main.py — Sign Language AI orchestration script
Checks dependencies → hand landmarker → model → launches Streamlit app.
"""

import os
import sys
import subprocess
import importlib

# ─────────────────────────────────────────────
#  Terminal colour helpers
# ─────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"{GREEN}  ✅ {msg}{RESET}")
def warn(msg):  print(f"{YELLOW}  ⚠️  {msg}{RESET}")
def err(msg):   print(f"{RED}  ❌ {msg}{RESET}")
def info(msg):  print(f"{CYAN}  ℹ️  {msg}{RESET}")
def header(msg): print(f"\n{BOLD}{CYAN}{msg}{RESET}")


# ─────────────────────────────────────────────
#  Step 1 — Verify Python requirements
# ─────────────────────────────────────────────
def check_requirements():
    header("Step 1 — Checking Python dependencies…")

    # Map package name → importable module name
    required = {
        "streamlit":        "streamlit",
        "mediapipe":        "mediapipe",
        "opencv-python":    "cv2",
        "scikit-learn":     "sklearn",
        "numpy":            "numpy",
        "Pillow":           "PIL",
        "pandas":           "pandas",
    }

    missing = []
    for pkg, mod in required.items():
        try:
            importlib.import_module(mod)
            ok(f"{pkg}")
        except ImportError:
            warn(f"{pkg} not found — will install")
            missing.append(pkg)

    if missing:
        info("Installing missing packages…")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
            check=True,
        )
        ok("All packages installed successfully")
    else:
        ok("All dependencies satisfied")


# ─────────────────────────────────────────────
#  Step 2 — Verify hand_landmarker.task
# ─────────────────────────────────────────────
def check_hand_landmarker():
    header("Step 2 — Checking MediaPipe hand landmarker model…")

    if os.path.exists("hand_landmarker.task"):
        size_mb = os.path.getsize("hand_landmarker.task") / (1024 * 1024)
        ok(f"hand_landmarker.task found ({size_mb:.1f} MB)")
        return

    err("hand_landmarker.task NOT found!")
    print(f"""
  Download it from Google:
  {CYAN}https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task{RESET}

  Then place it in this directory:
  {YELLOW}{os.path.abspath('.')}{RESET}
""")
    sys.exit(1)


# ─────────────────────────────────────────────
#  Step 3 — Train model if missing
# ─────────────────────────────────────────────
def check_and_train_model():
    header("Step 3 — Checking AI model (model.pkl)…")

    if os.path.exists("model.pkl"):
        size_mb = os.path.getsize("model.pkl") / (1024 * 1024)
        ok(f"model.pkl found ({size_mb:.1f} MB) — skipping training")
        return

    warn("model.pkl not found — starting training…")
    info("This will take 3–5 minutes on first run.")
    print()

    if not os.path.exists("asl_dataset"):
        err("asl_dataset/ folder not found!")
        print(f"""
  Download the ASL Alphabet dataset from Kaggle:
  {CYAN}https://www.kaggle.com/datasets/grassknoted/asl-alphabet{RESET}

  Extract and rename the folder to:
  {YELLOW}{os.path.abspath('asl_dataset')}{RESET}
""")
        sys.exit(1)

    try:
        subprocess.run([sys.executable, "train_asl.py"], check=True)
        ok("Model trained and saved as model.pkl")
    except subprocess.CalledProcessError as e:
        err(f"Training failed: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────
#  Step 4 — Launch Streamlit
# ─────────────────────────────────────────────
def run_app():
    header("Step 4 — Launching Sign Language AI…")
    print(f"""
{BOLD}{'='*54}
  🤟 Sign Language AI is starting…
{'='*54}{RESET}

  Open your browser at: {CYAN}http://localhost:8501{RESET}
  Press Ctrl+C to stop.
""")
    subprocess.run(
        ["streamlit", "run", "app.py",
         "--server.headless", "false",
         "--theme.base", "dark",
         "--theme.primaryColor", "#00FF88",
         "--theme.backgroundColor", "#0d0d0d",
         "--theme.secondaryBackgroundColor", "#111827",
         "--theme.textColor", "#e8e8e8"],
    )


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    print(f"""
{BOLD}{CYAN}
╔══════════════════════════════════════════════╗
║        🤟  Sign Language AI — v2.0          ║
║   Real-time ASL Recognition System           ║
╚══════════════════════════════════════════════╝{RESET}
""")

    check_requirements()
    check_hand_landmarker()
    check_and_train_model()
    run_app()


if __name__ == "__main__":
    main()