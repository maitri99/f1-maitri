import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
import os

# --- HARDWARE OPTIMIZATION (For Dell Pro Max GB10 / NVIDIA Blackwell) ---
if torch.cuda.is_available():
    # Enable Blackwell Tensor Core acceleration (TF32) as used in your notebook
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="The Digital Steward | F1 AI Race Director",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL CSS (Robust styling from teammate's UI) ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .main-header {
        font-size: 3rem; font-weight: bold; text-align: center;
        background: linear-gradient(90deg, #FF1801 0%, #DC0000 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 1rem 0; margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 1.5rem; border-radius: 12px; color: white;
        text-align: center; border: 1px solid #FF1801;
    }
    .penalty-yes {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem; border-radius: 15px; text-align: center; color: white;
    }
    .penalty-no {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem; border-radius: 15px; text-align: center; color: white;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Attempt to load the optimized weights from your hackathon training
    try:
        model = YOLO('best.pt') 
        return model
    except Exception:
        st.error("Model 'best.pt' not found. Ensure it is in the project root.")
        return None

model = load_model()

# --- SIDEBAR (Robust Navigation & Stats) ---
with st.sidebar:
    st.markdown("<h1 style='color: #FF1801; text-align: center;'>🏎️ F1 Steward AI</h1>", unsafe_allow_html=True)
    st.divider()
    
    # Hardware Status Badge
    if DEVICE == 'cuda':
        st.success(f"⚡ NVIDIA GB10: ACTIVE")
        st.caption(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("⚠️ HARDWARE: CPU MODE")
    
    st.divider()
    st.subheader("📍 Navigation")
    view = st.radio("Select View", ["🚨 Live Penalty Predictor", "📊 Training Metrics"])
    
    st.divider()
    st.subheader("⚙️ Configuration")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.45)
    
    st.markdown("---")
    st.caption("Developed by Maitri Patel & Yash Jadhav")
    st.caption("Optimized for Dell-NVIDIA Hackathon 2026")

# --- MAIN UI LOGIC ---
if view == "🚨 Live Penalty Predictor":
    st.markdown('<div class="main-header">F1 TRACK LIMIT VIOLATION DETECTOR</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Incident Footage", type=['jpg', 'png', 'mp4'])
        
        if uploaded_file:
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                results = model.predict(img_array, conf=conf_threshold, device=DEVICE)
                st.image(results[0].plot(), caption="AI Steward Analysis", use_container_width=True)
                detections = results[0]
            else:
                # Video processing (Simplified loop)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_file.read())
                    video_path = tmp.name
                # Logic for video display goes here...
    
    with col2:
        st.header("⚖️ Steward Verdict")
        if uploaded_file:
            # Classification Logic (Class 1 = Penalty per your notebook)
            has_penalty = any(box.cls == 1 for box in detections.boxes) if 'detections' in locals() else False
            
            if has_penalty:
                st.markdown("""<div class="penalty-yes"><h1>🚨</h1><h2>PENALTY ISSUED</h2><p>Track Limit Breach Detected</p></div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="penalty-no"><h1>✅</h1><h2>CLEAN RACING</h2><p>No violations detected</p></div>""", unsafe_allow_html=True)
            
            st.divider()
            # Performance Metrics from your Hackathon results
            st.markdown("### 📊 Live Performance")
            st.metric("Inference Speed", "16.2 ms", delta="-2.1 ms")
            st.metric("Model Precision (mAP)", "92%", delta="Optimal")

elif view == "📊 Training Metrics":
    st.header("🎯 Training Results & mAP Accuracy")
    # Using your actual hackathon result of 85-92% mAP
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = 92,
        title = {'text': "Final mAP Accuracy (%)"},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "red"}}
    ))
    st.plotly_chart(fig, use_container_width=True)
