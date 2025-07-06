import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import time
import sqlite3
import hashlib
import os
from datetime import datetime
import pandas as pd
from fpdf import FPDF
import plotly.express as px

# ---------------------------
# CONFIGURATION & STYLING
# ---------------------------
st.set_page_config(
    page_title="AI Helmet Detection",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern sexy theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 8px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #ffffff;
        font-weight: 500;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
        padding: 12px 16px;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stRadio > div > div > div > label {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .stRadio > div > div > div > label:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: #667eea;
    }
    
    .stRadio > div > div > div > label[data-testid="stRadio"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-color: #667eea;
    }
    
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stFileUploader > div > div > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div > div > div:hover {
        border-color: #667eea;
        background: rgba(255, 255, 255, 0.1);
    }
    
    .stSidebar {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stSidebar .sidebar-content {
        background: transparent;
    }
    
    .stMarkdown {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .stMarkdown h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .stMarkdown h2 {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }
    
    .stMarkdown h3 {
        color: #e0e0e0;
        font-weight: 500;
        font-size: 1.3rem;
        margin-bottom: 0.8rem;
    }
    
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stSuccess {
        background: rgba(76, 175, 80, 0.1);
        border-color: rgba(76, 175, 80, 0.3);
    }
    
    .stError {
        background: rgba(244, 67, 54, 0.1);
        border-color: rgba(244, 67, 54, 0.3);
    }
    
    .stWarning {
        background: rgba(255, 152, 0, 0.1);
        border-color: rgba(255, 152, 0, 0.3);
    }
    
    .stInfo {
        background: rgba(33, 150, 243, 0.1);
        border-color: rgba(33, 150, 243, 0.3);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    .webcam-container {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin: 16px 0;
        backdrop-filter: blur(10px);
    }
    
    .upload-container {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        backdrop-filter: blur(10px);
    }
    
    .analytics-container {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        backdrop-filter: blur(10px);
    }
    
    .login-container {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 32px;
        margin: 32px auto;
        max-width: 500px;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .stPlotlyChart {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a5acd 100%);
    }
</style>
""", unsafe_allow_html=True)

DB_NAME = 'violations.db'
FRAME_SAVE_DIR = 'violations'
HELMET_KEYWORDS = ['helmet', 'hardhat', 'headgear', 'safety_hat']

# ---------------------------
# DB SETUP
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT UNIQUE NOT NULL,
                 password TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS violations (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 timestamp TEXT,
                 reason TEXT,
                 frame_path TEXT)''')
    try:
        c.execute("ALTER TABLE violations ADD COLUMN frame_path TEXT")
    except sqlite3.OperationalError:
        pass  # already exists
    
    # Check if any users exist
    c.execute("SELECT COUNT(*) FROM users")
    user_count = c.fetchone()[0]
    
    # Create default admin user if no users exist
    if user_count == 0:
        default_username = "admin"
        default_password = "admin123"  # You should change this in production
        hashed_pw = hashlib.sha256(default_password.encode()).hexdigest()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                 (default_username, hashed_pw))
        print("Created default admin user - Username: admin, Password: admin123")
    
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
    except sqlite3.IntegrityError:
        st.error("🚨 Username already exists!")
    conn.close()

def check_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_pw))
    user = c.fetchone()
    conn.close()
    return user

def log_violation(reason, frame):
    if not os.path.exists(FRAME_SAVE_DIR):
        os.makedirs(FRAME_SAVE_DIR)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    frame_path = os.path.join(FRAME_SAVE_DIR, f"{timestamp}.jpg")
    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO violations (timestamp, reason, frame_path) VALUES (?, ?, ?)",
              (timestamp, reason, frame_path))
    conn.commit()
    conn.close()

# ---------------------------
# LOAD YOLOv5 MODEL
# ---------------------------
@st.cache_resource
def load_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
        return model
    except Exception as e:
        st.error(f"❌ Error loading YOLOv5 model: {e}")
        raise e

model = load_model()

# ---------------------------
# HELMET CLASS DETECTION
# ---------------------------
@st.cache_resource
def find_helmet_classes():
    sample_img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model(sample_img)
    class_names = list(results.names.values())
    
    helmet_classes = [cls for cls in class_names if any(k in cls.lower() for k in HELMET_KEYWORDS)]
    return helmet_classes

HELMET_CLASSES = find_helmet_classes()

# ---------------------------
# DETECTION & ALERT LOGIC
# ---------------------------
def draw_restricted_zone(frame, coords=(100, 100, 500, 400)):
    x1, y1, x2, y2 = coords
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, "Restricted Zone", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def detect_and_alert(frame, confidence_thresh):
    results = model(frame)
    detections = results.pandas().xyxy[0]
    frame = np.squeeze(results.render())

    persons = detections[detections['name'] == 'person']
    helmets = detections[detections['name'].isin(HELMET_CLASSES)] if HELMET_CLASSES else pd.DataFrame()

    alert_triggered = False

    for _, person in persons.iterrows():
        person_box = [person['xmin'], person['ymin'], person['xmax'], person['ymax']]
        helmet_found = False
        for _, helmet in helmets.iterrows():
            if (
                helmet['xmin'] > person_box[0] and
                helmet['ymin'] > person_box[1] and
                helmet['xmax'] < person_box[2] and
                helmet['ymax'] < person_box[3]
            ):
                helmet_found = True
                break
        if not helmet_found:
            alert_triggered = True
            log_violation("No Helmet Detected", frame)

        x_center = int((person_box[0] + person_box[2]) / 2)
        y_center = int((person_box[1] + person_box[3]) / 2)
        if 100 < x_center < 500 and 100 < y_center < 400:
            alert_triggered = True
            log_violation("Person entered Restricted Zone", frame)

    draw_restricted_zone(frame)
    return frame, alert_triggered

# ---------------------------
# AUTHENTICATION
# ---------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 3rem; font-weight: 700; margin-bottom: 10px;">🪖 AI Helmet Detection</h1>
        <p style="color: #b0b0b0; font-size: 1.2rem; margin-bottom: 40px;">Advanced Safety Monitoring System</p>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["🔐 Login", "📝 Sign Up"])
    
    with tabs[0]:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center; margin-bottom: 30px;">🔐 Admin Login</h2>', unsafe_allow_html=True)
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Login", use_container_width=True):
                if check_user(username, password):
                    st.session_state.logged_in = True
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center; margin-bottom: 30px;">📝 Create Admin Account</h2>', unsafe_allow_html=True)
        new_user = st.text_input("👤 New Username", placeholder="Choose a username")
        new_pass = st.text_input("🔒 New Password", type="password", placeholder="Choose a password")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✨ Sign Up", use_container_width=True):
                if new_user and new_pass:
                    add_user(new_user, new_pass)
                    st.success("✅ Account created! You can log in now.")
                else:
                    st.warning("⚠️ Please enter both username and password.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # ---------------------------
    # ADMIN DASHBOARD
    # ---------------------------
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 2.5rem; font-weight: 700; margin-bottom: 10px;">🪖 AI Helmet Detection Dashboard</h1>
        <p style="color: #b0b0b0; font-size: 1.1rem;">Advanced Safety Monitoring & Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with logout and settings
    with st.sidebar:
        st.markdown('<div style="background: rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 20px; margin-bottom: 20px;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #ffffff; margin-bottom: 15px;">⚙️ Settings</h3>', unsafe_allow_html=True)
        confidence_thresh = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div style="background: rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 20px;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #ffffff; margin-bottom: 15px;">👤 Account</h3>', unsafe_allow_html=True)
        if st.button("🔓 Logout", use_container_width=True):
            st.session_state.update({'logged_in': False})
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎥 Live Detection", "📂 Violation Logs", "📊 Analytics", "👥 Admin Panel"])

    # ---- Live Detection ----
    with tab1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #ffffff; margin-bottom: 20px;">🎥 Live Detection Options</h2>', unsafe_allow_html=True)
        
        detect_mode = st.radio(
            "Select Detection Mode:",
            ["📡 Webcam", "📸 Image Upload", "🎥 Video Upload"],
            horizontal=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if detect_mode == "📡 Webcam":
            st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #ffffff; margin-bottom: 15px;">📡 Real-time Webcam Detection</h3>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("📷 Start Webcam", use_container_width=True):
                    st.session_state['webcam_running'] = True
            with col2:
                if st.button("🛑 Stop Webcam", use_container_width=True):
                    st.session_state['webcam_running'] = False
            with col3:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state['webcam_running'] = False
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

            if st.session_state.get('webcam_running', False):
                st.markdown('<div style="background: rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 20px; margin-top: 20px;">', unsafe_allow_html=True)
                st.markdown('<h4 style="color: #ffffff; margin-bottom: 15px;">🎥 Live Feed</h4>', unsafe_allow_html=True)
                
                cap = cv2.VideoCapture(0)
                stframe = st.empty()
                
                while st.session_state['webcam_running']:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Webcam not available.")
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frame, alert = detect_and_alert(frame, confidence_thresh)
                    stframe.image(processed_frame, channels="RGB", use_container_width=True)
                    if alert:
                        st.warning("🚨 Violation Detected!")
                    time.sleep(0.1)
                cap.release()
                st.markdown('</div>', unsafe_allow_html=True)

        elif detect_mode == "📸 Image Upload":
            st.markdown('<div class="upload-container">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #ffffff; margin-bottom: 15px;">📸 Image Analysis</h3>', unsafe_allow_html=True)
            
            img_file = st.file_uploader("Choose Image", type=['jpg', 'png', 'jpeg'], help="Upload an image to analyze for helmet violations")
            
            if img_file is not None:
                img = Image.open(img_file)
                img_array = np.array(img)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<h4 style="color: #ffffff; margin-bottom: 10px;">📸 Original Image</h4>', unsafe_allow_html=True)
                    st.image(img, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    st.markdown('<h4 style="color: #ffffff; margin-bottom: 10px;">🔍 Detection Result</h4>', unsafe_allow_html=True)
                    if st.button("🔍 Analyze Image", use_container_width=True):
                        processed_img, alert = detect_and_alert(img_array, confidence_thresh)
                        st.image(processed_img, caption="Detection Result", use_container_width=True)
                        if alert:
                            st.warning("🚨 Violation Detected!")
                        else:
                            st.success("✅ No violations detected!")
            st.markdown('</div>', unsafe_allow_html=True)

        elif detect_mode == "🎥 Video Upload":
            st.markdown('<div class="upload-container">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #ffffff; margin-bottom: 15px;">🎥 Video Analysis</h3>', unsafe_allow_html=True)
            
            vid_file = st.file_uploader("Choose Video", type=['mp4', 'avi', 'mov'], help="Upload a video to analyze for helmet violations")
            
            if vid_file is not None:
                tfile = open("temp_video.mp4", 'wb')
                tfile.write(vid_file.read())
                tfile.close()
                
                if st.button("🎬 Process Video", use_container_width=True):
                    st.markdown('<div style="background: rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 20px; margin-top: 20px;">', unsafe_allow_html=True)
                    st.markdown('<h4 style="color: #ffffff; margin-bottom: 15px;">🎬 Processing Video</h4>', unsafe_allow_html=True)
                    
                    cap = cv2.VideoCapture("temp_video.mp4")
                    stframe = st.empty()
                    progress_bar = st.progress(0)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    count = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        processed_frame, alert = detect_and_alert(frame, confidence_thresh)
                        stframe.image(processed_frame, channels="RGB", use_container_width=True)
                        if alert:
                            st.warning("🚨 Violation Detected!")
                        count += 1
                        progress_bar.progress(count / total_frames)
                    
                    cap.release()
                    st.success("✅ Video Processing Complete!")
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ---- Violation Logs ----
    with tab2:
        st.markdown('<div class="analytics-container">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #ffffff; margin-bottom: 20px;">📂 Violation Logs</h2>', unsafe_allow_html=True)
        
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM violations", conn)
        conn.close()
        
        if not df.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin-bottom: 5px;">📊 Total Violations</h3>
                    <h2 style="color: #ffffff; font-size: 2rem;">{len(df)}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin-bottom: 5px;">📅 Today</h3>
                    <h2 style="color: #ffffff; font-size: 2rem;">{len(df[df['timestamp'].str.startswith(datetime.now().strftime('%Y-%m-%d'))])}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin-bottom: 5px;">🚨 No Helmet</h3>
                    <h2 style="color: #ffffff; font-size: 2rem;">{len(df[df['reason'] == 'No Helmet Detected'])}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin-bottom: 5px;">🚫 Restricted Zone</h3>
                    <h2 style="color: #ffffff; font-size: 2rem;">{len(df[df['reason'] == 'Person entered Restricted Zone'])}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Data table
            st.markdown('<h3 style="color: #ffffff; margin: 30px 0 15px 0;">📋 Recent Violations</h3>', unsafe_allow_html=True)
            st.dataframe(df.tail(10), use_container_width=True)
            
            # Recent alerts with images
            st.markdown('<h3 style="color: #ffffff; margin: 30px 0 15px 0;">📸 Recent Alerts</h3>', unsafe_allow_html=True)
            cols = st.columns(3)
            for idx, row in df.tail(6).iterrows():
                with cols[idx % 3]:
                    st.markdown(f"**🕒 {row['timestamp']}**")
                    st.markdown(f"**🚨 {row['reason']}**")
                    if os.path.exists(row['frame_path']):
                        st.image(row['frame_path'], width=200)
            
            # Export options
            st.markdown('<h3 style="color: #ffffff; margin: 30px 0 15px 0;">📥 Export Reports</h3>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬇️ Download CSV", use_container_width=True):
                    df.to_csv("violations_report.csv", index=False)
                    st.success("✅ CSV exported as violations_report.csv")
            
            with col2:
                if st.button("⬇️ Download PDF", use_container_width=True):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    for idx, row in df.iterrows():
                        pdf.cell(200, 10, text=f"{row['timestamp']} - {row['reason']}", ln=True)
                    pdf.output("violations_report.pdf")
                    st.success("✅ PDF exported as violations_report.pdf")
        else:
            st.info("📭 No violations logged yet. Start detection to log violations.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Analytics ----
    with tab3:
        st.markdown('<div class="analytics-container">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #ffffff; margin-bottom: 20px;">📊 Violation Analytics</h2>', unsafe_allow_html=True)
        
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM violations", conn)
        conn.close()

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d_%H-%M-%S", errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df['date'] = df['timestamp'].dt.date
            daily_counts = df.groupby('date').size().reset_index(name='count').sort_values('date')

            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h3 style="color: #ffffff; margin-bottom: 15px;">📈 Daily Violation Trend</h3>', unsafe_allow_html=True)
                trend_chart = px.line(
                    daily_counts, 
                    x='date', 
                    y='count', 
                    markers=True, 
                    title="Daily Violation Trend",
                    template="plotly_dark"
                )
                trend_chart.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(trend_chart, use_container_width=True)
            
            with col2:
                st.markdown('<h3 style="color: #ffffff; margin-bottom: 15px;">🍩 Violation Type Distribution</h3>', unsafe_allow_html=True)
                pie_chart = px.pie(
                    df, 
                    names='reason', 
                    title="Violation Type Distribution",
                    template="plotly_dark"
                )
                pie_chart.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(pie_chart, use_container_width=True)
            
            st.markdown('<h3 style="color: #ffffff; margin: 30px 0 15px 0;">📊 Violation Type Counts</h3>', unsafe_allow_html=True)
            bar_chart = px.bar(
                df, 
                x='reason', 
                title="Violation Type Counts",
                template="plotly_dark"
            )
            bar_chart.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(bar_chart, use_container_width=True)
        else:
            st.info("📊 No data yet. Start detection to generate analytics.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Admin Panel ----
    with tab4:
        st.markdown('<div class="analytics-container">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #ffffff; margin-bottom: 20px;">👥 Admin Panel</h2>', unsafe_allow_html=True)
        
        conn = sqlite3.connect(DB_NAME)
        admins = pd.read_sql_query("SELECT id, username FROM users", conn)
        conn.close()
        
        st.markdown('<h3 style="color: #ffffff; margin-bottom: 15px;">👤 Registered Administrators</h3>', unsafe_allow_html=True)
        st.dataframe(admins, use_container_width=True)
        
        st.markdown('<h3 style="color: #ffffff; margin: 30px 0 15px 0;">🔧 System Information</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea; margin-bottom: 5px;">🪖 Model Status</h3>
                <h2 style="color: #ffffff; font-size: 1.5rem;">✅ Loaded</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea; margin-bottom: 5px;">📊 Database Status</h3>
                <h2 style="color: #ffffff; font-size: 1.5rem;">✅ Connected</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<h3 style="color: #ffffff; margin: 30px 0 15px 0;">📁 File Management</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Violation Logs", use_container_width=True):
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                c.execute("DELETE FROM violations")
                conn.commit()
                conn.close()
                st.success("✅ Violation logs cleared!")
                st.rerun()
        
        with col2:
            if st.button("📁 Open Violations Folder", use_container_width=True):
                os.system(f"open {FRAME_SAVE_DIR}")
                st.success("✅ Opened violations folder!")
        st.markdown('</div>', unsafe_allow_html=True)

init_db()















