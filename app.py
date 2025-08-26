

import os
import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2
import time
import sqlite3
import hashlib
from datetime import datetime
import pandas as pd
from fpdf import FPDF
import plotly.express as px
from io import BytesIO
from streamlit_webrtc import webrtc_streamer
import av

# ---------------------------
# CONFIGURATION & STYLING
# ---------------------------
st.set_page_config(
    page_title="AI-Powered Safety Gear Detection for Industrial Monitoring",

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





# Database configuration
import os.path
import tempfile

# Initialize storage paths in session state
if 'storage_initialized' not in st.session_state:
    # Create unique subdirectories for this session
    session_id = str(hash(datetime.now().isoformat()))
    base_dir = tempfile.gettempdir()

    st.session_state.temp_dir = os.path.join(base_dir, f"helmet_detection_{session_id}")
    st.session_state.db_name = os.path.join(st.session_state.temp_dir, 'violations.db')
    st.session_state.frame_dir = os.path.join(st.session_state.temp_dir, 'violations')

    # Create necessary directories
    os.makedirs(st.session_state.temp_dir, exist_ok=True)
    os.makedirs(st.session_state.frame_dir, exist_ok=True)

    st.session_state.storage_initialized = True

# Use session state variables
DB_NAME = st.session_state.db_name
FRAME_SAVE_DIR = st.session_state.frame_dir
HELMET_KEYWORDS = ['helmet', 'hardhat', 'headgear', 'safety_hat']

def init_db():
    """Initialize SQLite database with required tables and default admin user"""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(DB_NAME), exist_ok=True)

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        # Create users table if not exists
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        ''')

        # Create violations table if not exists
        c.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                reason TEXT NOT NULL,
                frame_path TEXT NOT NULL
            )
        ''')

        # Add default admin user if not exists
        try:
            default_username = "admin"
            default_password = "admin123"

            # Check if admin user exists
            c.execute("SELECT username FROM users WHERE username = ?", (default_username,))
            if not c.fetchone():
                # Insert new admin user
                c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                         (default_username, hash_password(default_password)))
                st.success(f"Default admin account created! Username: {default_username}, Password: {default_password}")
        except sqlite3.IntegrityError:
            pass  # Admin user already exists
        except Exception as e:
            st.error(f"Error creating default admin: {str(e)}")

        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")
        raise e

def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_password(password):
    """Validate password meets minimum requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    return True, "Password meets requirements"

def add_user(username, password):
    """Add a new user to the database with validation"""
    # Validate username
    if len(username) < 3:
        st.error("❌ Username must be at least 3 characters long")
        return False

    # Validate password
    is_valid, message = validate_password(password)
    if not is_valid:
        st.error(f"❌ {message}")
        return False

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                 (username, hash_password(password)))
        conn.commit()
        st.success("Account created successfully!")
        st.info("You can now log in with your credentials")
        return True
    except sqlite3.IntegrityError:
        st.error("Username already exists!")
        return False
    finally:
        conn.close()

def check_user(username, password):
    """Verify user credentials"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()

    if result and result[0] == hash_password(password):
        return True
    return False

def sanitize_text_for_pdf(text):
    """Sanitize text for PDF generation by replacing Unicode characters"""
    if text is None:
        return ""

    # Convert to string if not already
    text = str(text)

    # Replace common Unicode characters with ASCII equivalents
    replacements = {
        '\u2022': '-',  # bullet point
        '\u2013': '-',  # en dash
        '\u2014': '--', # em dash
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote
        '\u201c': '"',  # left double quote
        '\u201d': '"',  # right double quote
        '\u2026': '...', # ellipsis
        '\u00a0': ' ',  # non-breaking space
        '\u00b0': 'deg', # degree symbol
        '\u2190': '<-', # left arrow
        '\u2192': '->', # right arrow
        '\u2191': '^',  # up arrow
        '\u2193': 'v',  # down arrow
        '\u2713': 'v',  # check mark
        '\u2717': 'x',  # cross mark
        '\u26a0': '!',  # warning sign
        '\u2705': 'v',  # check mark button
        '\u274c': 'x',  # cross mark
        '\u2b05': '<-', # left arrow
        '\u27a1': '->', # right arrow
        '\u2b06': '^',  # up arrow
        '\u2b07': 'v',  # down arrow
        '\u2139': 'i',  # information
        '\u2764': '<3', # heart
        '\ud83d\udcc4': '[DOC]', # document
        '\ud83d\udcc8': '[CHART]', # chart
        '\ud83d\udcca': '[BAR]',   # bar chart
        '\ud83d\udcdd': '[NOTE]',  # memo
        '\ud83d\uddbc': '[FRAME]', # framed picture
    }

    # Apply replacements
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)

    # Remove any remaining non-ASCII characters
    text = text.encode('ascii', errors='ignore').decode('ascii')

    return text

def log_violation(reason, frame):
    """Log a violation with timestamp and save the frame"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create violations directory if it doesn't exist
    if not os.path.exists(FRAME_SAVE_DIR):
        os.makedirs(FRAME_SAVE_DIR)

    # Save the frame
    frame_path = os.path.join(FRAME_SAVE_DIR, f"{timestamp}.jpg")
    cv2.imwrite(frame_path, frame)

    # Log to database
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO violations (timestamp, reason, frame_path) VALUES (?, ?, ?)",
              (timestamp, reason, frame_path))
    conn.commit()
    conn.close()

# Initialize database at startup
init_db()

class ModelWrapper:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Load YOLOv8 model"""
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            raise e

    def get_names(self):
        """Get class names from the model"""
        if not self.model:
            return {}
        return self.model.names

    def predict(self, frame):
        """Run prediction and return results in a consistent format"""
        if not self.model:
            raise ValueError("Model not loaded properly")

        results = self.model(frame)
        # Convert YOLOv8 results to pandas DataFrame for consistency
        pred_df = pd.DataFrame([
            {
                'xmin': box.xyxy[0][0],
                'ymin': box.xyxy[0][1],
                'xmax': box.xyxy[0][2],
                'ymax': box.xyxy[0][3],
                'confidence': box.conf,
                'name': self.model.names[int(box.cls)]
            }
            for r in results
            for box in r.boxes
        ])
        return pred_df, results[0].plot()  # Return both DataFrame and rendered image

def create_chart_config():
    return {
        'layout': {
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': 'white'},
            'title': {'font': {'color': 'white'}},
            'xaxis': {
                'gridcolor': 'rgba(255,255,255,0.1)',
                'zerolinecolor': 'rgba(255,255,255,0.1)',
                'tickfont': {'color': 'white'}
            },
            'yaxis': {
                'gridcolor': 'rgba(255,255,255,0.1)',
                'zerolinecolor': 'rgba(255,255,255,0.1)',
                'tickfont': {'color': 'white'}
            }
        }
    }

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    try:
        if os.path.exists('best.pt'):
            return ModelWrapper('best.pt')
        raise FileNotFoundError("YOLOv8 model file not found")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        raise e

model = load_model()

# ---------------------------
# HELMET CLASS DETECTION
# ---------------------------
@st.cache_resource
def find_helmet_classes():
    sample_img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model.predict(sample_img)[0]  # Get DataFrame results
    class_names = list(model.get_names().values())

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
    detections, rendered_frame = model.predict(frame)

    # Log a violation for every 'NO-Hardhat' or 'NO-Mask' detected above the threshold
    for _, row in detections.iterrows():
        if row['confidence'] > confidence_thresh:
            if row['name'] == "NO-Hardhat":
                log_violation("No Hardhat Detected", rendered_frame)
                st.warning("Violation Detected!")
                draw_restricted_zone(rendered_frame)
                return rendered_frame, True
            if row['name'] == "NO-Mask":
                log_violation("No Mask Detected", rendered_frame)
                st.warning("Violation Detected!")
                draw_restricted_zone(rendered_frame)
                return rendered_frame, True

        # Check for restricted zone violations
        if row['name'] == 'person':
            person_box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            x_center = int((person_box[0] + person_box[2]) / 2)
            y_center = int((person_box[1] + person_box[3]) / 2)
            if 100 < x_center < 500 and 100 < y_center < 400:
                log_violation("Person entered Restricted Zone", rendered_frame)
                draw_restricted_zone(rendered_frame)
                return rendered_frame, True

    draw_restricted_zone(rendered_frame)
    return rendered_frame, False

# ---------------------------
# AUTHENTICATION
# ---------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 3rem; font-weight: 700; margin-bottom: 10px;">AI-Powered Safety Gear Detection for Industrial Monitoring

</h1>
        <p style="color: #b0b0b0; font-size: 1.2rem; margin-bottom: 40px;">Advanced Safety Monitoring System</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["Login", "Sign Up"])

    with tabs[0]:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center; margin-bottom: 30px;">🔐 Admin Login</h2>', unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Login", use_container_width=True):
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
        new_user = st.text_input("New Username", placeholder="Choose a username")
        new_pass = st.text_input("New Password", type="password", placeholder="Choose a password")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Sign Up", use_container_width=True):
                if new_user and new_pass:
                    # Show password requirements
                    st.info("""
                    Password requirements:
                    - At least 8 characters long
                    - At least one uppercase letter
                    - At least one lowercase letter
                    - At least one number
                    """)
                    if add_user(new_user, new_pass):
                        time.sleep(2)  # Give user time to read the success message
                        st.rerun()  # Refresh to login tab
                else:
                    st.warning("⚠️ Please enter username and password.")
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

    # Sidebar with settings and account
    with st.sidebar:
        st.markdown("""
<div style="background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);">
    <h3 style="color: #ffffff;
               margin-bottom: 1rem;
               font-size: 1.3rem;
               letter-spacing: 0.5px;">Settings</h3>
    <div style="margin-bottom: 1rem;">
        <p style="color: #b0b0b0;
                  margin-bottom: 0.5rem;
                  font-size: 0.9rem;">Adjust detection sensitivity</p>
    </div>
</div>
""", unsafe_allow_html=True)

    confidence_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    st.markdown("""
<div style="background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);">
    <h3 style="color: #ffffff;
               margin-bottom: 1rem;
               font-size: 1.3rem;
               letter-spacing: 0.5px;">Account</h3>
</div>
""", unsafe_allow_html=True)

    if st.button("Logout", use_container_width=True):
        st.session_state.update({'logged_in': False})
        st.rerun()

    # Main content tabs
    detection_tab, logs_tab, analytics_tab, admin_tab = st.tabs([
        "Live Detection",
        "Violation Logs",
        "Analytics",
        "Admin Panel"
    ])

    # ---- Live Detection ----
    with detection_tab:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #ffffff; margin-bottom: 20px;">🎥 Detection Options</h2>', unsafe_allow_html=True)

        detect_mode = st.radio(
            "Select Detection Mode:",
            ["Webcam", "Single Image Upload", "Batch Image Upload"],
            horizontal=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if detect_mode == "Webcam":
            st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #ffffff; margin-bottom: 15px;">📡 Real-time Webcam Detection</h3>', unsafe_allow_html=True)

            def video_frame_callback(frame):
                img = frame.to_ndarray(format="bgr24")
                processed_img, alert = detect_and_alert(img, confidence_thresh)
                if alert:
                    st.warning("Violation Detected!")
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

            webrtc_streamer(
                key="helmet-detection",
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            )

            st.markdown('</div>', unsafe_allow_html=True)

        elif detect_mode == "Single Image Upload":
            st.markdown('<div class="upload-container">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #ffffff; margin-bottom: 15px;">📸 Single Image Analysis</h3>', unsafe_allow_html=True)

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
                    if st.button("Analyze Image", use_container_width=True):
                        processed_img, alert = detect_and_alert(img_array, confidence_thresh)
                        st.image(processed_img, caption="Detection Result", use_container_width=True)
                        if alert:
                            st.warning("⚠️ Violation Detected!")
                        else:
                            st.success("✅ No violations detected!")
            st.markdown('</div>', unsafe_allow_html=True)

        elif detect_mode == "Batch Image Upload":
            st.markdown('<div class="upload-container">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #ffffff; margin-bottom: 15px;">📸 Batch Image Analysis</h3>', unsafe_allow_html=True)

            img_files = st.file_uploader("Choose Images", type=['jpg', 'png', 'jpeg'],
                                       help="Upload multiple images to analyze for helmet violations",
                                       accept_multiple_files=True)

            if img_files:
                st.markdown(f'<p style="color: #b0b0b0;">Selected {len(img_files)} image(s) for processing</p>', unsafe_allow_html=True)

                if st.button("Analyze All Images", use_container_width=True):
                    st.markdown('<div style="background: rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 20px; margin-top: 20px;">', unsafe_allow_html=True)
                    st.markdown('<h4 style="color: #ffffff; margin-bottom: 15px;">🔍 Batch Processing Results</h4>', unsafe_allow_html=True)

                    progress_bar = st.progress(0)
                    total_violations = 0

                    for i, img_file in enumerate(img_files):
                        img = Image.open(img_file)
                        img_array = np.array(img)

                        st.markdown(f'<h5 style="color: #667eea; margin: 20px 0 10px 0;">Image {i+1}: {img_file.name}</h5>', unsafe_allow_html=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown('<h6 style="color: #ffffff; margin-bottom: 10px;">📸 Original</h6>', unsafe_allow_html=True)
                            st.image(img, caption=f"Original - {img_file.name}", use_container_width=True)

                        with col2:
                            st.markdown('<h6 style="color: #ffffff; margin-bottom: 10px;">🔍 Detection Result</h6>', unsafe_allow_html=True)
                            processed_img, alert = detect_and_alert(img_array, confidence_thresh)
                            st.image(processed_img, caption=f"Analyzed - {img_file.name}", use_container_width=True)

                            if alert:
                                st.warning(f"⚠️ Violation detected in {img_file.name}")
                                total_violations += 1
                            else:
                                st.success(f"✅ No violations in {img_file.name}")

                        # Update progress
                        progress_bar.progress((i + 1) / len(img_files))

                        # Add separator except for last image
                        if i < len(img_files) - 1:
                            st.markdown('<hr style="border: 1px solid rgba(255,255,255,0.1); margin: 30px 0;">', unsafe_allow_html=True)

                    # Summary
                    st.markdown('<div style="background: rgba(102, 126, 234, 0.1); border-radius: 12px; padding: 20px; margin-top: 30px; border-left: 4px solid #667eea;">', unsafe_allow_html=True)
                    # Create variables for conditional styling
                    status_color = '#ff6b6b' if total_violations > 0 else '#51cf66'
                    status_icon = '⚠️' if total_violations > 0 else '✅'
                    status_text = 'Issues Found' if total_violations > 0 else 'All Clear'

                    st.markdown(f'''
                    <h4 style="color: #667eea; margin-bottom: 15px;">📊 Batch Processing Summary</h4>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <p style="color: #ffffff; margin: 5px 0;"><strong>Total Images Processed:</strong> {len(img_files)}</p>
                            <p style="color: #ffffff; margin: 5px 0;"><strong>Violations Detected:</strong> {total_violations}</p>
                            <p style="color: #ffffff; margin: 5px 0;"><strong>Clean Images:</strong> {len(img_files) - total_violations}</p>
                        </div>
                        <div style="text-align: right;">
                            <p style="color: {status_color}; font-size: 2rem; margin: 0;"><strong>{status_icon}</strong></p>
                            <p style="color: #b0b0b0; font-size: 0.9rem;">{status_text}</p>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ---- Violation Logs ----
    with logs_tab:
        st.markdown('<div class="analytics-container">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #ffffff; margin-bottom: 20px;">📂 Violation Logs</h2>', unsafe_allow_html=True)

        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM violations", conn)
        conn.close()

        if df.empty:
            st.info("No violations logged yet. Start detection to log violations.")
        else:
            # Update the metric cards in the logs tab
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin-bottom: 10px; font-size: 1.1rem;">Total Violations</h3>
                    <h2 style="color: #ffffff; font-size: 2.5rem; margin-bottom: 5px;">{len(df)}</h2>
                    <p style="color: #b0b0b0; font-size: 0.9rem;">All time</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin-bottom: 10px; font-size: 1.1rem;">Today's Alerts</h3>
                    <h2 style="color: #ffffff; font-size: 2.5rem; margin-bottom: 5px;">{len(df[df['timestamp'].str.startswith(datetime.now().strftime('%Y-%m-%d'))])}</h2>
                    <p style="color: #b0b0b0; font-size: 0.9rem;">Last 24 hours</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin-bottom: 10px; font-size: 1.1rem;">Safety Gear Violations</h3>
                    <h2 style="color: #ffffff; font-size: 2.5rem; margin-bottom: 5px;">{len(df[df['reason'] == 'No Hardhat Detected'])}</h2>
                    <p style="color: #b0b0b0; font-size: 0.9rem;">Missing equipment</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin-bottom: 10px; font-size: 1.1rem;">Zone Violations</h3>
                    <h2 style="color: #ffffff; font-size: 2.5rem; margin-bottom: 5px;">{len(df[df['reason'] == 'Person entered Restricted Zone'])}</h2>
                    <p style="color: #b0b0b0; font-size: 0.9rem;">Restricted areas</p>
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
                if st.button("Download CSV", use_container_width=True):
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV Report",
                        data=csv_data,
                        file_name='violations_report.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                    st.success("CSV exported successfully!")
            with col2:
                # st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; padding: 15px; margin-bottom: 10px;">', unsafe_allow_html=True)
                # st.markdown('<h4 style="color: #ffffff; margin-bottom: 10px; text-align: center;">📄 Enhanced PDF Report</h4>', unsafe_allow_html=True)
                # st.markdown('<p style="color: #ffffff; font-size: 0.9rem; text-align: center; margin-bottom: 15px;">Comprehensive report with violation screenshots, timestamps, and statistical analysis</p>', unsafe_allow_html=True)

                if st.button("Download PDF with Screenshots", use_container_width=True):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=16, style='B')
                    pdf.cell(200, 15, sanitize_text_for_pdf("SAFETY VIOLATION REPORT"), ln=True, align='C')
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, sanitize_text_for_pdf(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), ln=True, align='C')
                    pdf.ln(10)

                    # Summary Section
                    pdf.set_font("Arial", size=14, style='B')
                    pdf.cell(200, 10, sanitize_text_for_pdf("EXECUTIVE SUMMARY"), ln=True)
                    pdf.set_font("Arial", size=11)
                    pdf.cell(200, 8, sanitize_text_for_pdf(f"Total Violations: {len(df)}"), ln=True)

                    violation_types = df['reason'].value_counts() if len(df) > 0 else pd.Series()
                    if len(df) > 0:
                        # Convert timestamp strings to datetime for PDF report
                        try:
                            df_temp = df.copy()
                            df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'], format="%Y-%m-%d_%H-%M-%S", errors='coerce')
                            df_temp = df_temp.dropna(subset=['timestamp'])
                            if not df_temp.empty:
                                pdf.cell(200, 8, sanitize_text_for_pdf(f"Report Period: {df_temp['timestamp'].min().strftime('%Y-%m-%d')} to {df_temp['timestamp'].max().strftime('%Y-%m-%d')}"), ln=True)
                            else:
                                pdf.cell(200, 8, sanitize_text_for_pdf("Report Period: Unable to parse timestamps"), ln=True)
                        except:
                            pdf.cell(200, 8, sanitize_text_for_pdf(f"Report Period: {df['timestamp'].min()} to {df['timestamp'].max()}"), ln=True)
                        if not violation_types.empty:
                            pdf.cell(200, 8, sanitize_text_for_pdf(f"Most Common Violation: {violation_types.index[0]} ({violation_types.iloc[0]} cases)"), ln=True)
                    pdf.ln(10)

                    # Violation Details with Screenshots
                    pdf.set_font("Arial", size=14, style='B')
                    pdf.cell(200, 10, sanitize_text_for_pdf("DETAILED VIOLATION LOG"), ln=True)
                    pdf.set_font("Arial", size=10)

                    for idx, row in df.iterrows():
                        if pdf.get_y() > 250:  # Check if near end of page
                            pdf.add_page()

                        pdf.set_font("Arial", size=12, style='B')
                        pdf.cell(200, 8, sanitize_text_for_pdf(f"Violation #{idx + 1}"), ln=True)
                        pdf.set_font("Arial", size=10)
                        pdf.cell(200, 6, sanitize_text_for_pdf(f"Timestamp: {row['timestamp']}"), ln=True)
                        pdf.cell(200, 6, sanitize_text_for_pdf(f"Type: {row['reason']}"), ln=True)

                        # Add screenshot if available
                        if os.path.exists(row['frame_path']):
                            try:
                                # Resize image to fit PDF
                                img_width = 80
                                img_height = 60
                                pdf.image(row['frame_path'], x=10, y=pdf.get_y(), w=img_width, h=img_height)
                                pdf.ln(img_height + 5)
                            except:
                                pdf.cell(200, 6, sanitize_text_for_pdf("Screenshot: Available but could not be embedded"), ln=True)
                        else:
                            pdf.cell(200, 6, sanitize_text_for_pdf("Screenshot: Not available"), ln=True)

                        pdf.cell(200, 6, sanitize_text_for_pdf(f"Evidence Path: {row['frame_path']}"), ln=True)
                        pdf.ln(5)

                    # Statistics Section
                    pdf.add_page()
                    pdf.set_font("Arial", size=14, style='B')
                    pdf.cell(200, 10, sanitize_text_for_pdf("STATISTICAL ANALYSIS"), ln=True)
                    pdf.set_font("Arial", size=11)

                    # Violation type breakdown
                    if not violation_types.empty:
                        pdf.cell(200, 8, sanitize_text_for_pdf("Violation Type Breakdown:"), ln=True)
                        for violation_type, count in violation_types.items():
                            percentage = (count / len(df)) * 100
                            pdf.cell(200, 6, sanitize_text_for_pdf(f"  - {violation_type}: {count} cases ({percentage:.1f}%)"), ln=True)
                    else:
                        pdf.cell(200, 8, sanitize_text_for_pdf("No violations to analyze"), ln=True)

                    pdf.ln(10)

                    # Time-based analysis
                    if len(df) > 0:
                        try:
                            # Convert timestamps for time analysis
                            df_time = df.copy()
                            df_time['timestamp'] = pd.to_datetime(df_time['timestamp'], format="%Y-%m-%d_%H-%M-%S", errors='coerce')
                            df_time = df_time.dropna(subset=['timestamp'])

                            if not df_time.empty:
                                hourly_stats = df_time.groupby(df_time['timestamp'].dt.hour).size()
                                if not hourly_stats.empty:
                                    peak_hour = hourly_stats.idxmax()
                                    pdf.cell(200, 8, sanitize_text_for_pdf("Time-based Analysis:"), ln=True)
                                    pdf.cell(200, 6, sanitize_text_for_pdf(f"  - Peak violation hour: {peak_hour}:00"), ln=True)
                                    pdf.cell(200, 6, sanitize_text_for_pdf(f"  - Violations during peak hour: {hourly_stats.max()}"), ln=True)
                                else:
                                    pdf.cell(200, 8, sanitize_text_for_pdf("Time-based Analysis: No hourly data available"), ln=True)
                            else:
                                pdf.cell(200, 8, sanitize_text_for_pdf("Time-based Analysis: Unable to parse timestamps"), ln=True)
                        except Exception as e:
                            pdf.cell(200, 8, sanitize_text_for_pdf("Time-based Analysis: Data insufficient"), ln=True)

                    # Create temporary file with proper path
                    temp_pdf_path = os.path.join(tempfile.gettempdir(), f"enhanced_violation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                    pdf.output(temp_pdf_path)

                    with open(temp_pdf_path, "rb") as pdf_file:
                        pdf_data = pdf_file.read()

                    st.download_button(
                        label="Get Enhanced PDF Report with Screenshots",
                        data=pdf_data,
                        file_name=f"violation_report_with_screenshots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                    try:
                        os.remove(temp_pdf_path)
                    except:
                        pass  # File might not exist or be locked

                    st.success("Enhanced PDF report with screenshots generated successfully!")
                    st.info("This PDF includes violation screenshots, detailed timestamps, statistical analysis, and executive summary.")

                st.markdown('</div>', unsafe_allow_html=True)

    # ---- Analytics ----
    with analytics_tab:
        st.markdown('<div class="analytics-container">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #ffffff; margin-bottom: 20px;">📊 Advanced Violation Analytics</h2>', unsafe_allow_html=True)

        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM violations", conn)
        conn.close()

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d_%H-%M-%S", errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            df['weekday'] = df['timestamp'].dt.day_name()
            df['month'] = df['timestamp'].dt.month_name()

            # Enhanced Summary Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            total_violations = len(df)
            today_violations = len(df[df['date'] == pd.Timestamp.now().date()])
            this_week = len(df[df['timestamp'] >= pd.Timestamp.now() - pd.Timedelta(days=7)])
            avg_daily = df.groupby('date').size().mean() if len(df.groupby('date')) > 0 else 0
            most_common_violation = df['reason'].mode().iloc[0] if not df['reason'].mode().empty else "None"

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin-bottom: 10px; font-size: 1rem;">Total Violations</h3>
                    <h2 style="color: #ffffff; font-size: 2rem; margin-bottom: 5px;">{total_violations}</h2>
                    <p style="color: #b0b0b0; font-size: 0.8rem;">All time</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin-bottom: 10px; font-size: 1rem;">Today</h3>
                    <h2 style="color: #ffffff; font-size: 2rem; margin-bottom: 5px;">{today_violations}</h2>
                    <p style="color: #b0b0b0; font-size: 0.8rem;">Current day</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin-bottom: 10px; font-size: 1rem;">This Week</h3>
                    <h2 style="color: #ffffff; font-size: 2rem; margin-bottom: 5px;">{this_week}</h2>
                    <p style="color: #b0b0b0; font-size: 0.8rem;">Last 7 days</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin-bottom: 10px; font-size: 1rem;">Daily Average</h3>
                    <h2 style="color: #ffffff; font-size: 2rem; margin-bottom: 5px;">{avg_daily:.1f}</h2>
                    <p style="color: #b0b0b0; font-size: 0.8rem;">Per day</p>
                </div>
                """, unsafe_allow_html=True)

            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin-bottom: 10px; font-size: 1rem;">Top Issue</h3>
                    <h2 style="color: #ffffff; font-size: 1.2rem; margin-bottom: 5px;">{most_common_violation[:15]}{'...' if len(most_common_violation) > 15 else ''}</h2>
                    <p style="color: #b0b0b0; font-size: 0.8rem;">Most frequent</p>
                </div>
                """, unsafe_allow_html=True)

            # Row 1: Trend and Distribution
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<h3 style="color: #ffffff; margin: 30px 0 15px 0;">📈 Daily Violation Trend</h3>', unsafe_allow_html=True)
                daily_counts = df.groupby('date').size().reset_index(name='count').sort_values('date')
                trend_chart = px.line(
                    daily_counts,
                    x='date',
                    y='count',
                    markers=True,
                    template="plotly_dark"
                )
                trend_chart.update_traces(
                    line=dict(width=3, color='#667eea'),
                    marker=dict(size=8, color='#764ba2')
                )
                trend_chart.update_layout(height=400)
                st.plotly_chart(trend_chart, use_container_width=True)

            with col2:
                st.markdown('<h3 style="color: #ffffff; margin: 30px 0 15px 0;">🎯 Violation Type Distribution</h3>', unsafe_allow_html=True)
                pie_chart = px.pie(
                    df,
                    names='reason',
                    template="plotly_dark",
                    color_discrete_sequence=['#667eea', '#764ba2', '#ff6b6b', '#51cf66', '#ffd93d', '#6bcf7f']
                )
                pie_chart.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
                )
                pie_chart.update_layout(height=400)
                st.plotly_chart(pie_chart, use_container_width=True)

            # Row 2: Hourly and Weekly Patterns
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<h3 style="color: #ffffff; margin: 30px 0 15px 0;">⏰ Hourly Violation Pattern</h3>', unsafe_allow_html=True)
                hourly_counts = df.groupby('hour').size().reset_index(name='count')
                hourly_chart = px.bar(
                    hourly_counts,
                    x='hour',
                    y='count',
                    template="plotly_dark",
                    color_discrete_sequence=['#667eea']
                )
                hourly_chart.update_layout(
                    xaxis_title="Hour of Day",
                    yaxis_title="Violations Count",
                    height=400
                )
                st.plotly_chart(hourly_chart, use_container_width=True)

            with col2:
                st.markdown('<h3 style="color: #ffffff; margin: 30px 0 15px 0;">📅 Weekly Violation Pattern</h3>', unsafe_allow_html=True)
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekday_counts = df.groupby('weekday').size().reindex(weekday_order, fill_value=0).reset_index()
                weekday_counts.columns = ['weekday', 'count']
                weekly_chart = px.bar(
                    weekday_counts,
                    x='weekday',
                    y='count',
                    template="plotly_dark",
                    color_discrete_sequence=['#764ba2']
                )
                weekly_chart.update_layout(
                    xaxis_title="Day of Week",
                    yaxis_title="Violations Count",
                    height=400
                )
                weekly_chart.update_xaxes(tickangle=45)
                st.plotly_chart(weekly_chart, use_container_width=True)

            # Row 3: Monthly and Severity Analysis
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<h3 style="color: #ffffff; margin: 30px 0 15px 0;">🗓️ Monthly Trend</h3>', unsafe_allow_html=True)
                monthly_counts = df.groupby('month').size().reset_index(name='count')
                monthly_chart = px.line(
                    monthly_counts,
                    x='month',
                    y='count',
                    markers=True,
                    template="plotly_dark"
                )
                monthly_chart.update_traces(
                    line=dict(width=3, color='#51cf66'),
                    marker=dict(size=10, color='#ffd93d')
                )
                monthly_chart.update_layout(height=400)
                monthly_chart.update_xaxes(tickangle=45)
                st.plotly_chart(monthly_chart, use_container_width=True)

            with col2:
                st.markdown('<h3 style="color: #ffffff; margin: 30px 0 15px 0;">📊 Violation Severity Analysis</h3>', unsafe_allow_html=True)
                violation_counts = df['reason'].value_counts().reset_index()
                violation_counts.columns = ['reason', 'count']
                severity_chart = px.bar(
                    violation_counts,
                    x='count',
                    y='reason',
                    orientation='h',
                    template="plotly_dark",
                    color_discrete_sequence=['#ff6b6b']
                )
                severity_chart.update_layout(
                    xaxis_title="Number of Violations",
                    yaxis_title="Violation Type",
                    height=400
                )
                st.plotly_chart(severity_chart, use_container_width=True)

            # Additional Insights
            st.markdown('<h3 style="color: #ffffff; margin: 30px 0 15px 0;">🔍 Key Insights</h3>', unsafe_allow_html=True)

            insights_col1, insights_col2 = st.columns(2)
            with insights_col1:
                peak_hour = df.groupby('hour').size().idxmax() if not df.empty else 0
                peak_day = df.groupby('weekday').size().idxmax() if not df.empty else "N/A"
                st.markdown(f"""
                <div style="background: rgba(102, 126, 234, 0.1); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                    <h4 style="color: #667eea; margin-bottom: 15px;">⏰ Peak Activity Times</h4>
                    <p style="color: #ffffff; margin: 8px 0;">• Peak Hour: <strong>{peak_hour}:00</strong></p>
                    <p style="color: #ffffff; margin: 8px 0;">• Peak Day: <strong>{peak_day}</strong></p>
                    <p style="color: #ffffff; margin: 8px 0;">• Most violations occur during <strong>{peak_hour}:00-{peak_hour+1}:00</strong></p>
                </div>
                """, unsafe_allow_html=True)

            with insights_col2:
                safety_score = max(0, 100 - (total_violations * 2)) if total_violations < 50 else 0
                risk_level = "Low" if safety_score > 70 else "Medium" if safety_score > 40 else "High"
                risk_color = "#51cf66" if risk_level == "Low" else "#ffd93d" if risk_level == "Medium" else "#ff6b6b"
                st.markdown(f"""
                <div style="background: rgba(255, 107, 107, 0.1); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                    <h4 style="color: #ff6b6b; margin-bottom: 15px;">⚠️ Safety Assessment</h4>
                    <p style="color: #ffffff; margin: 8px 0;">• Safety Score: <strong style="color: {risk_color};">{safety_score}/100</strong></p>
                    <p style="color: #ffffff; margin: 8px 0;">• Risk Level: <strong style="color: {risk_color};">{risk_level}</strong></p>
                    <p style="color: #ffffff; margin: 8px 0;">• Total Issues: <strong>{total_violations}</strong></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No data available for analytics. Start detection to generate comprehensive analytics.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Admin Panel ----
    with admin_tab:
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
                <h3 style="color: #667eea; margin-bottom: 5px;">Model Status</h3>
                <h2 style="color: #ffffff; font-size: 1.5rem;">Active</h2>
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
            if st.button("Clear Violation Logs", use_container_width=True):
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                c.execute("DELETE FROM violations")
                conn.commit()
                conn.close()
                st.success("Violation logs cleared!")
                st.rerun()
        with col2:
            if st.button("Open Violations Folder", use_container_width=True):
                os.system(f"open {FRAME_SAVE_DIR}")
                st.success("Opened violations folder!")
        st.markdown('</div>', unsafe_allow_html=True)

#pull















