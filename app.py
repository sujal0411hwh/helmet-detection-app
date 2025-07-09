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

# Update the main title with better styling
st.markdown("""
<div style="text-align: center; padding: 3rem 0;">
    <h1 style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               background-clip: text;
               font-size: 3.5rem;
               font-weight: 700;
               margin-bottom: 1.5rem;
               letter-spacing: 1px;">
        AI Safety Gear Detection
    </h1>
    <p style="color: #b0b0b0;
              font-size: 1.4rem;
              margin-bottom: 2rem;
              max-width: 800px;
              margin-left: auto;
              margin-right: auto;
              line-height: 1.6;">
        Advanced Safety Monitoring System with Real-time Detection
    </p>
</div>
""", unsafe_allow_html=True)

# Update the dashboard title
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 2.5rem; font-weight: 700; margin-bottom: 10px;">Safety Gear Detection Dashboard</h1>
    <p style="color: #b0b0b0; font-size: 1.1rem;">Advanced Safety Monitoring & Analytics</p>
</div>
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
TEMP_DIR = st.session_state.temp_dir
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
    
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                 (username, hash_password(password)))
        conn.commit()
        conn.close()
        st.success("Account created successfully!")
        st.info("You can now log in with your credentials")
        return True
    except sqlite3.IntegrityError:
        st.error("Username already exists!")
        return False
    except Exception as e:
        st.error(f"Error creating user: {str(e)}")
        return False

def check_user(username, password):
    """Verify user credentials"""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        conn.close()
        
        if result and result[0] == hash_password(password):
            return True
        return False
    except Exception as e:
        st.error(f"Error checking user credentials: {str(e)}")
        return False

def log_violation(reason, frame):
    """Log a violation with timestamp and save the frame"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create violations directory if it doesn't exist
        os.makedirs(FRAME_SAVE_DIR, exist_ok=True)
        
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
    except Exception as e:
        st.error(f"Error logging violation: {str(e)}")
        # Continue execution even if logging fails
        pass

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
def download_file_from_url(url, dest_path):
    """Download a file from a URL to a destination path"""
    try:
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Show download progress
        progress_text = "Downloading model file..."
        progress_bar = st.progress(0)
        
        # Download with progress updates
        with open(dest_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        # Update progress bar
                        progress = int((downloaded / total_size) * 100)
                        progress_bar.progress(progress / 100)
        
        progress_bar.empty()
        return True
    except Exception as e:
        st.error(f"Error downloading file: {str(e)}")
        return False

@st.cache_resource
def load_model():
    """Load or download YOLOv8 model"""
    try:
        # Define model paths
        model_filename = 'yolov8n.pt'
        model_path = os.path.join(TEMP_DIR, model_filename)
        
        # Check if model exists in temp directory
        if not os.path.exists(model_path):
            st.info("Downloading YOLOv8 model... This may take a few minutes.")
            
            # URL for the YOLOv8n model
            model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
            
            # Download the model
            if not download_file_from_url(model_url, model_path):
                raise Exception("Failed to download model file")
            
            st.success("✅ Model downloaded successfully!")
        
        # Load the model
        return ModelWrapper(model_path)
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        raise e

# Add requests to requirements
if 'requests' not in st.session_state:
    with open('requirements.txt', 'a') as f:
        f.write('\nrequests==2.31.0  # Required for model download\n')
    st.session_state.requests = True

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("❌ Could not load the model. Please check your internet connection and try again.")
    st.stop()  # Stop the app if model loading fails

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
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 3rem; font-weight: 700; margin-bottom: 10px;">🪖 AI Helmet Detection</h1>
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
        st.markdown('<h2 style="color: #ffffff; margin-bottom: 20px;">🎥 Live Detection Options</h2>', unsafe_allow_html=True)
        
        detect_mode = st.radio(
            "Select Detection Mode:",
            ["Webcam", "Image Upload", "Video Upload"],
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

        elif detect_mode == "Image Upload":
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
                    if st.button("Analyze Image", use_container_width=True):
                        processed_img, alert = detect_and_alert(img_array, confidence_thresh)
                        st.image(processed_img, caption="Detection Result", use_container_width=True)
                        if alert:
                            st.warning("Violation Detected!")
                        else:
                            st.success("No violations detected!")
            st.markdown('</div>', unsafe_allow_html=True)

        elif detect_mode == "Video Upload":
            st.markdown('<div class="upload-container">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #ffffff; margin-bottom: 15px;">🎥 Video Analysis</h3>', unsafe_allow_html=True)
            
            vid_file = st.file_uploader("Choose Video", type=['mp4', 'avi', 'mov'], help="Upload a video to analyze for helmet violations")
            
            if vid_file is not None:
                tfile = open("temp_video.mp4", 'wb')
                tfile.write(vid_file.read())
                tfile.close()
                
                if st.button("Process Video", use_container_width=True):
                    st.markdown('<div style="background: rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 20px; margin-top: 20px;">', unsafe_allow_html=True)
                    st.markdown('<h4 style="color: #ffffff; margin-bottom: 15px;">🎬 Processing Video</h4>', unsafe_allow_html=True)
                    import cv2  # 💥 Delay cv2 import
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
                            st.warning("Violation Detected!")
                        count += 1
                        progress_bar.progress(count / total_frames)
                    cap.release()
                    st.success("Video Processing Complete!")
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ---- Violation Logs ----
    with logs_tab:
        st.markdown('<div class="analytics-container">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #ffffff; margin-bottom: 20px;">📂 Violation Logs</h2>', unsafe_allow_html=True)
        
        try:
            conn = sqlite3.connect(DB_NAME)
            df = pd.read_sql_query("SELECT * FROM violations", conn)
            conn.close()
        except Exception as e:
            st.error(f"Error loading violation logs: {str(e)}")
            df = pd.DataFrame()  # Empty DataFrame as fallback
        
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
                if st.button("Download PDF", use_container_width=True):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, "Violation Report", ln=True, align='C')
                    for idx, row in df.iterrows():
                        pdf.cell(200, 10, f"{row['timestamp']} - {row['reason']}", ln=True)
                    temp_pdf_path = "temp_report.pdf"
                    pdf.output(temp_pdf_path)
                    with open(temp_pdf_path, "rb") as pdf_file:
                        pdf_data = pdf_file.read()
                    os.remove(temp_pdf_path)
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_data,
                        file_name="violations_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

    # ---- Analytics ----
    with analytics_tab:
        st.markdown('<div class="analytics-container">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #ffffff; margin-bottom: 20px;">Violation Analytics</h2>', unsafe_allow_html=True)
        
        try:
            conn = sqlite3.connect(DB_NAME)
            df = pd.read_sql_query("SELECT * FROM violations", conn)
            conn.close()
        except Exception as e:
            st.error(f"Error loading analytics data: {str(e)}")
            df = pd.DataFrame()  # Empty DataFrame as fallback
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d_%H-%M-%S", errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df['date'] = df['timestamp'].dt.date
            daily_counts = df.groupby('date').size().reset_index(name='count').sort_values('date')
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<h3 style="color: #ffffff; margin-bottom: 15px;">Daily Violation Trend</h3>', unsafe_allow_html=True)
                trend_chart = px.line(
                    daily_counts, 
                    x='date', 
                    y='count', 
                    markers=True,
                    template="plotly_dark"
                )
                trend_chart.update_layout(**create_chart_config()['layout'])
                trend_chart.update_traces(
                    line_color='#667eea',
                    marker=dict(size=8, color='#764ba2')
                )
                st.plotly_chart(trend_chart, use_container_width=True)
                
            with col2:
                st.markdown('<h3 style="color: #ffffff; margin-bottom: 15px;">Violation Type Distribution</h3>', unsafe_allow_html=True)
                pie_chart = px.pie(
                    df,
                    names='reason',
                    template="plotly_dark",
                    color_discrete_sequence=['#667eea', '#764ba2', '#a78bfa']
                )
                pie_chart.update_layout(**create_chart_config()['layout'])
                pie_chart.update_traces(
                    textfont_color='white',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
                )
                st.plotly_chart(pie_chart, use_container_width=True)
                
            st.markdown('<h3 style="color: #ffffff; margin: 30px 0 15px 0;">Violation Type Counts</h3>', unsafe_allow_html=True)
            bar_chart = px.bar(
                df,
                x='reason',
                template="plotly_dark",
                color_discrete_sequence=['#667eea']
            )
            bar_chart.update_layout(**create_chart_config()['layout'])
            bar_chart.update_traces(
                marker_line_color='rgba(255,255,255,0.2)',
                marker_line_width=1,
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )
            st.plotly_chart(bar_chart, use_container_width=True)
        else:
            st.info("No data yet. Start detection to generate analytics.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Admin Panel ----
    with admin_tab:
        st.markdown('<div class="analytics-container">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #ffffff; margin-bottom: 20px;">👥 Admin Panel</h2>', unsafe_allow_html=True)
        try:
            conn = sqlite3.connect(DB_NAME)
            admins = pd.read_sql_query("SELECT id, username FROM users", conn)
            conn.close()
        except Exception as e:
            st.error(f"Error loading admin data: {str(e)}")
            admins = pd.DataFrame()  # Empty DataFrame as fallback
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
                try:
                    conn = sqlite3.connect(DB_NAME)
                    c = conn.cursor()
                    c.execute("DELETE FROM violations")
                    conn.commit()
                    conn.close()
                    st.success("Violation logs cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing violation logs: {str(e)}")
        with col2:
            if st.button("Open Violations Folder", use_container_width=True):
                os.system(f"open {FRAME_SAVE_DIR}")
                st.success("Opened violations folder!")
        st.markdown('</div>', unsafe_allow_html=True)















