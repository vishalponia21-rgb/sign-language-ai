cat > README.md << 'EOF'
# Sign Language AI 🤟

Real-time American Sign Language (ASL) letter detection using Computer Vision and Machine Learning.

## 🎯 Features
- Detects 26 alphabets (A-Z) in real-time
- 96% accuracy on Kaggle ASL dataset
- Beautiful Streamlit web interface
- Word builder functionality
- Detection history tracking

## 🛠️ Tech Stack
- **MediaPipe** - Hand landmark detection (21 points)
- **OpenCV** - Video capture and processing
- **Scikit-learn** - Random Forest classifier
- **Streamlit** - Interactive web UI
- **Python 3.11**

## 📊 Model Performance
- **Accuracy**: 96%
- **Training Samples**: 10,045 ASL images
- **Detection Latency**: <50ms per frame
- **Supported Classes**: 26 letters (A-Z)

## 🚀 How to Run

```bash
# Clone repository
git clone https://github.com/vishalponia21-rgb/sign-language-ai.git
cd sign-language-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📚 Dataset
- Source: Kaggle ASL Alphabet Dataset
- Total Images: 10,045
- Categories: A-Z, space, delete, nothing
- Size: ~1 GB

## 🎮 How to Use
1. Start the app - camera will open
2. Make ASL hand signs
3. AI detects the letter in real-time
4. Build words by pressing letter buttons
5. Clear and backspace available

## 💡 Use Cases
- Accessibility tool for deaf communication
- ASL learning aid
- Real-time sign language translation
- Interactive communication system

## 👨‍💻 Author
Vishal Ponia

## 📄 License
MIT License
EOF