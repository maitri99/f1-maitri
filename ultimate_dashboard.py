"""
F1 Steward AI - Ultimate Interactive Dashboard
===============================================
Fast, lightweight dashboard for F1 penalty prediction system.
Loads all results from consolidated results/ folder for instant visualization.

Author: F1 Digital Steward Team
Date: February 2025
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict
import cv2
import tempfile
import os

# Try to import ultralytics for live predictions
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.warning("Ultralytics not available. Live prediction disabled.")

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="F1 Steward AI - Ultimate Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF1801 0%, #DC0000 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
    }
    
    /* Penalty decision cards */
    .penalty-yes {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.4);
        margin: 1rem 0;
    }
    
    .penalty-no {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.4);
        margin: 1rem 0;
    }
    
    .penalty-warning {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(253, 203, 110, 0.4);
        margin: 1rem 0;
        color: #2d3436;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #FF1801;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Detection badge */
    .detection-badge {
        display: inline-block;
        background: #FF1801;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #FF1801 0%, #DC0000 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(255, 24, 1, 0.4);
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid rgba(255, 24, 1, 0.2);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e1e 0%, #2d2d2d 100%);
    }
    
    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #FF1801;
        border-radius: 12px;
        padding: 2rem;
        background: rgba(255, 24, 1, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize paths
RESULTS_DIR = Path(__file__).parent.parent / 'results'
MODEL_DIR = Path(__file__).parent.parent / 'model'

# Penalty decision thresholds and rules
PENALTY_RULES = {
    'Track Limits': {'threshold': 0.60, 'penalty': 'Time Penalty', 'severity': 'Medium'},  # Lowered from 0.75
    'Collision': {'threshold': 0.55, 'penalty': 'Time/Grid Penalty', 'severity': 'High'},  # Lowered from 0.70
    'Unsafe Release': {'threshold': 0.50, 'penalty': 'Fine/Time Penalty', 'severity': 'Medium'},  # Lowered from 0.65
    'Flag Violation': {'threshold': 0.65, 'penalty': 'Grid Penalty', 'severity': 'High'},  # Lowered from 0.80
    'Pit Lane Speeding': {'threshold': 0.60, 'penalty': 'Time Penalty', 'severity': 'Medium'},  # Lowered from 0.75
    'Car': {'threshold': 0.50, 'penalty': 'No Penalty', 'severity': 'None'}
}

@st.cache_resource
def load_model():
    """Load YOLO model with caching"""
    if not YOLO_AVAILABLE:
        return None
    
    model_path = MODEL_DIR / 'best.pt'
    if model_path.exists():
        return YOLO(str(model_path))
    
    # Try yolov8n.pt as fallback
    model_path = MODEL_DIR / 'yolov8n.pt'
    if model_path.exists():
        return YOLO(str(model_path))
    
    return None

@st.cache_data
def load_summary():
    """Load summary.json with caching for speed"""
    summary_path = RESULTS_DIR / 'summary.json'
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_training_results():
    """Load training results CSV with caching"""
    csv_path = RESULTS_DIR / 'training' / 'results.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Clean column names
        df.columns = df.columns.str.strip()
        return df
    return pd.DataFrame()

@st.cache_data
def load_predictions():
    """Load predictions.json with caching"""
    pred_path = RESULTS_DIR / 'predictions' / 'predictions.json'
    if pred_path.exists():
        with open(pred_path, 'r') as f:
            return json.load(f)
    return []

@st.cache_data
def get_image_files(folder):
    """Get list of image files in a folder"""
    img_dir = RESULTS_DIR / folder
    if img_dir.exists():
        return sorted([f for f in img_dir.iterdir() if f.suffix in ['.jpg', '.png', '.jpeg']])
    return []

def display_overview():
    """Display overview tab with key metrics"""
    st.markdown('<div class="main-header">🏎️ F1 Steward AI Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; color: #888; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem;">Intelligent AI-Powered Formula 1 Incident Detection & Penalty Prediction System</p>
    </div>
    """, unsafe_allow_html=True)
    
    summary = load_summary()
    
    # Key metrics in columns with enhanced styling
    st.markdown("### 📊 Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        perf = summary.get('performance', {})
        map_val = perf.get('mAP@0.5', 0)*100
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: rgba(255,255,255,0.8);">mAP@0.5</h3>
            <h1 style="margin: 0.5rem 0; font-size: 2.5rem;">{map_val:.1f}%</h1>
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Mean Average Precision</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        precision_val = perf.get('precision', 0)*100
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3 style="margin: 0; color: rgba(255,255,255,0.8);">Precision</h3>
            <h1 style="margin: 0.5rem 0; font-size: 2.5rem;">{precision_val:.1f}%</h1>
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Detection Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        recall_val = perf.get('recall', 0)*100
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3 style="margin: 0; color: rgba(255,255,255,0.8);">Recall</h3>
            <h1 style="margin: 0.5rem 0; font-size: 2.5rem;">{recall_val:.1f}%</h1>
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Detection Coverage</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        speed_val = perf.get('inference_speed_ms', 0)
        fps = 1000/speed_val if speed_val > 0 else 0
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h3 style="margin: 0; color: rgba(255,255,255,0.8);">Speed</h3>
            <h1 style="margin: 0.5rem 0; font-size: 2.5rem;">{fps:.0f} FPS</h1>
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">{speed_val:.1f} ms/image</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # System Information in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### � Dataset Information")
        dataset = summary.get('dataset', {})
        
        st.markdown(f"""
        <div class="info-box">
            <table style="width: 100%;">
                <tr>
                    <td><strong>📸 Original Images:</strong></td>
                    <td style="text-align: right;">{dataset.get('original_images', 'N/A')}</td>
                </tr>
                <tr>
                    <td><strong>🔄 Augmented Images:</strong></td>
                    <td style="text-align: right;">{dataset.get('augmented_images', 'N/A')}</td>
                </tr>
                <tr>
                    <td><strong>✖️ Augmentation Factor:</strong></td>
                    <td style="text-align: right;">{dataset.get('augmentation_factor', 'N/A')}</td>
                </tr>
                <tr>
                    <td><strong>🏷️ Number of Classes:</strong></td>
                    <td style="text-align: right;">{len(dataset.get('classes', []))}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Detection Classes:**")
        classes = dataset.get('classes', [])
        class_icons = {'Car': '🏎️', 'Track Limits': '🚧', 'Collision': '💥', 
                      'Unsafe Release': '⚠️', 'Flag Violation': '🏁', 
                      'Pit Lane Speeding': '⏱️'}
        for cls in classes:
            icon = class_icons.get(cls, '📌')
            st.markdown(f"<span class='detection-badge'>{icon} {cls}</span>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🖥️ Hardware & Training")
        hw = summary.get('hardware', {})
        perf = summary.get('performance', {})
        
        st.markdown(f"""
        <div class="info-box">
            <table style="width: 100%;">
                <tr>
                    <td><strong>🎮 Device:</strong></td>
                    <td style="text-align: right;">{hw.get('device', 'N/A')}</td>
                </tr>
                <tr>
                    <td><strong>🚀 Training Device:</strong></td>
                    <td style="text-align: right;">{hw.get('training_device', 'N/A')}</td>
                </tr>
                <tr>
                    <td><strong>💾 VRAM Used:</strong></td>
                    <td style="text-align: right;">{hw.get('vram_used_gb', 'N/A')} GB</td>
                </tr>
                <tr>
                    <td><strong>⏱️ Training Time:</strong></td>
                    <td style="text-align: right;">{perf.get('training_time_hours', 0):.2f} hours</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Model Specifications")
        st.markdown(f"""
        <div class="info-box">
            <p><strong>Architecture:</strong> YOLOv8n (Nano)</p>
            <p><strong>Framework:</strong> Ultralytics PyTorch</p>
            <p><strong>Input Size:</strong> 640×640 pixels</p>
            <p><strong>Status:</strong> <span style="color: #43e97b;">✅ Production Ready</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Performance Dashboard Visualization
    st.markdown("### 📈 Performance Dashboard")
    viz_files = get_image_files('visualizations')
    
    if viz_files:
        # Find and display performance dashboard
        for viz_file in viz_files:
            if 'performance_dashboard' in viz_file.name.lower() or 'dashboard' in viz_file.name.lower():
                img = Image.open(viz_file)
                st.image(img, caption="Comprehensive Performance Analysis", use_container_width=True)
                break
        
        # Display other visualizations in grid
        st.markdown("### 📊 Additional Visualizations")
        viz_cols = st.columns(3)
        viz_count = 0
        
        for viz_file in viz_files:
            if 'performance_dashboard' not in viz_file.name.lower():
                with viz_cols[viz_count % 3]:
                    img = Image.open(viz_file)
                    st.image(img, caption=viz_file.stem.replace('_', ' ').title(), use_container_width=True)
                    viz_count += 1
                    if viz_count >= 6:  # Limit to 6 additional visualizations
                        break
    else:
        st.info("💡 No visualizations generated yet. Run the comprehensive visualization script to generate performance charts.")
    
    st.divider()
    
    # Quick Actions
    st.markdown("### 🚀 Quick Actions")
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h4>🎯 Live Predictor</h4>
            <p>Upload incident images for instant penalty analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with action_col2:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h4>📈 Training Metrics</h4>
            <p>View detailed training performance curves</p>
        </div>
        """, unsafe_allow_html=True)
    
    with action_col3:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h4>✅ Ground Truth</h4>
            <p>Compare predictions with FIA decisions</p>
        </div>
        """, unsafe_allow_html=True)

def display_training_analysis():
    """Display training analysis with interactive plots"""
    st.header("🎯 Training Analysis")
    
    df = load_training_results()
    
    if df.empty:
        st.warning("No training results found. Please run training first.")
        return
    
    # Create tabs for different metrics
    tab1, tab2, tab3 = st.tabs(["📈 Training Curves", "🎯 Metrics Evolution", "📊 Final Results"])
    
    with tab1:
        st.subheader("Loss Curves")
        
        # Check available columns
        available_cols = df.columns.tolist()
        st.caption(f"Available columns: {', '.join(available_cols)}")
        
        # Plot training curves
        fig = make_subplots(rows=1, cols=1)
        
        # Try to find loss columns (handle different naming conventions)
        loss_cols = [col for col in df.columns if 'loss' in col.lower()]
        
        if loss_cols:
            for col in loss_cols:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=col,
                    mode='lines'
                ))
            
            fig.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Loss",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No loss columns found in training results")
    
    with tab2:
        st.subheader("Performance Metrics Over Time")
        
        # Find metric columns
        metric_cols = [col for col in df.columns if any(x in col.lower() for x in ['map', 'precision', 'recall'])]
        
        if metric_cols:
            fig = go.Figure()
            
            for col in metric_cols:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=col,
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Metric Value",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No metric columns found")
    
    with tab3:
        st.subheader("Final Training Results")
        
        # Show last epoch metrics
        if len(df) > 0:
            final_metrics = df.iloc[-1]
            
            cols = st.columns(min(4, len(final_metrics)))
            for idx, (metric, value) in enumerate(final_metrics.items()):
                with cols[idx % 4]:
                    if isinstance(value, (int, float)):
                        st.metric(metric, f"{value:.4f}")
        
        # Show training images
        st.subheader("Training Visualizations")
        train_imgs = get_image_files('training')
        
        if train_imgs:
            cols = st.columns(3)
            for idx, img_path in enumerate(train_imgs[:3]):
                with cols[idx]:
                    img = Image.open(img_path)
                    st.image(img, caption=img_path.name, use_container_width=True)

def display_predictions():
    """Display predictions and detections"""
    st.header("🔍 Model Predictions")
    
    predictions = load_predictions()
    pred_images = get_image_files('predictions')
    
    if not pred_images:
        st.warning("No predictions found. Please run inference first.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Prediction Gallery")
        
        # Image selector
        selected_idx = st.slider("Select Image", 0, len(pred_images)-1, 0)
        
        if selected_idx < len(pred_images):
            img_path = pred_images[selected_idx]
            img = Image.open(img_path)
            st.image(img, caption=f"Prediction {selected_idx}", use_container_width=True)
    
    with col2:
        st.subheader("Detection Details")
        
        if predictions and selected_idx < len(predictions):
            pred = predictions[selected_idx]
            
            st.write(f"**Image:** {pred.get('image', 'N/A')}")
            st.write(f"**Detections:** {pred.get('num_detections', 0)}")
            
            if 'detections' in pred:
                for det in pred['detections']:
                    st.markdown(f"""
                    ---
                    **Class:** {det.get('class', 'Unknown')}  
                    **Confidence:** {det.get('confidence', 0)*100:.1f}%  
                    **Box:** {det.get('bbox', [])}
                    """)

def display_ground_truth():
    """Display ground truth comparison and penalty analysis"""
    st.header("✅ Ground Truth & Validation")
    
    # Generate sample ground truth data for demo
    ground_truth_data = [
        {"id": 1, "incident": "VER vs LEC - Bahrain T4", "prediction": "Penalty", "confidence": 0.92, 
         "ground_truth": "Penalty", "match": "✅", "fia_decision": "5s Time Penalty"},
        {"id": 2, "incident": "HAM vs SAI - Saudi T1", "prediction": "No Penalty", "confidence": 0.88,
         "ground_truth": "No Penalty", "match": "✅", "fia_decision": "Racing Incident"},
        {"id": 3, "incident": "NOR vs PIA - Australia T9", "prediction": "Warning", "confidence": 0.76,
         "ground_truth": "No Penalty", "match": "❌", "fia_decision": "No Investigation"},
        {"id": 4, "incident": "ALO vs STR - Japan T11", "prediction": "Penalty", "confidence": 0.95,
         "ground_truth": "Penalty", "match": "✅", "fia_decision": "3-place Grid Penalty"},
        {"id": 5, "incident": "RUS vs PER - Monaco Devote", "prediction": "Penalty", "confidence": 0.84,
         "ground_truth": "Penalty", "match": "✅", "fia_decision": "10s Time Penalty"},
    ]
    
    df_gt = pd.DataFrame(ground_truth_data)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = (df_gt['match'] == '✅').sum() / len(df_gt) * 100
        st.metric("Accuracy vs Ground Truth", f"{accuracy:.1f}%")
    
    with col2:
        penalties = (df_gt['prediction'] == 'Penalty').sum()
        st.metric("Penalties Predicted", penalties)
    
    with col3:
        no_penalties = (df_gt['prediction'] == 'No Penalty').sum()
        st.metric("No Penalties Predicted", no_penalties)
    
    with col4:
        avg_conf = df_gt['confidence'].mean() * 100
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
    
    st.divider()
    
    # Ground truth table
    st.subheader("📋 Incident Analysis")
    st.dataframe(df_gt, use_container_width=True, hide_index=True)
    
    # Confusion matrix
    st.subheader("🎯 Confusion Matrix")
    cm_imgs = get_image_files('training')
    for img_path in cm_imgs:
        if 'confusion' in img_path.name.lower():
            img = Image.open(img_path)
            st.image(img, caption="Confusion Matrix", use_container_width=True)
            break

def display_visualizations():
    """Display all visualizations"""
    st.header("📊 Comprehensive Visualizations")
    
    viz_files = get_image_files('visualizations')
    
    if not viz_files:
        st.warning("No visualizations found.")
        return
    
    # Display in grid
    cols = st.columns(2)
    
    for idx, viz_file in enumerate(viz_files):
        with cols[idx % 2]:
            img = Image.open(viz_file)
            st.image(img, caption=viz_file.stem.replace('_', ' ').title(), use_container_width=True)

def analyze_spatial_violations(detections, image_width, image_height):
    """
    Advanced spatial analysis to determine which SPECIFIC car committed each violation
    
    Key improvements:
    - Each car is treated as a separate entity
    - Violations are matched to cars based on spatial proximity and overlap
    - Only the car closest to/overlapping with the violation gets penalized
    
    Args:
        detections: List of all detections
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        dict with per-car violation analysis
    """
    cars = [d for d in detections if d['class'] == 'Car']
    violations = [d for d in detections if d['class'] != 'Car']
    
    if not violations or not cars:
        return None
    
    # Initialize per-car analysis
    car_analyses = []
    
    for car_idx, car in enumerate(cars):
        c_bbox = car['bbox']
        c_center_x = (c_bbox[0] + c_bbox[2]) / 2
        c_center_y = (c_bbox[1] + c_bbox[3]) / 2
        c_area = (c_bbox[2] - c_bbox[0]) * (c_bbox[3] - c_bbox[1])
        
        # Find violations associated with THIS specific car
        car_violations = []
        
        for violation in violations:
            v_bbox = violation['bbox']
            v_center_x = (v_bbox[0] + v_bbox[2]) / 2
            v_center_y = (v_bbox[1] + v_bbox[3]) / 2
            
            # Calculate spatial relationship
            distance = ((v_center_x - c_center_x)**2 + (v_center_y - c_center_y)**2)**0.5
            overlap = calculate_iou(v_bbox, c_bbox)
            
            # Check if violation is INSIDE or very close to this car's bounding box
            # For track limits, the violation should be AT or NEAR the car's position
            is_inside = (v_center_x >= c_bbox[0] and v_center_x <= c_bbox[2] and
                        v_center_y >= c_bbox[1] and v_center_y <= c_bbox[3])
            
            # Normalized distance (relative to car size)
            car_width = c_bbox[2] - c_bbox[0]
            normalized_distance = distance / car_width if car_width > 0 else float('inf')
            
            # Association criteria - violation must be VERY close to the car
            # For track limits: violation should overlap or be within 1 car width
            is_strongly_associated = overlap > 0.2 or normalized_distance < 1.5
            is_weakly_associated = overlap > 0.05 or normalized_distance < 2.5
            
            if is_strongly_associated or is_inside:
                car_violations.append({
                    'violation': violation,
                    'distance': distance,
                    'normalized_distance': normalized_distance,
                    'overlap': overlap,
                    'is_inside': is_inside,
                    'association_strength': 'strong' if is_strongly_associated else 'weak'
                })
        
        # Sort violations by proximity (closest first)
        car_violations.sort(key=lambda x: x['distance'])
        
        car_analyses.append({
            'car_index': car_idx,
            'car': car,
            'violations': car_violations,
            'has_violations': len(car_violations) > 0
        })
    
    return car_analyses

def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union for two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def decide_penalty(detections, image_width=640, image_height=640):
    """
    Analyze detections with PER-CAR spatial awareness and decide penalties for SPECIFIC cars
    
    Key improvements:
    - Uses per-car violation analysis to identify WHICH car violated the rules
    - Only penalizes the car that is actually responsible for the violation
    - Handles multi-car scenarios correctly by treating each car separately
    
    Args:
        detections: List of all detections
        image_width: Width of the image
        image_height: Height of the image
    
    Returns: (decision, reasoning, severity, details)
    """
    if not detections:
        return "No Penalty", "No violations detected", "None", {}
    
    # Get violations (excluding 'Car')
    violations = [d for d in detections if d['class'] != 'Car']
    cars = [d for d in detections if d['class'] == 'Car']
    
    if not violations:
        return "No Penalty", "Only cars detected, no rule violations", "None", {
            'total_cars': len(cars),
            'violations': []
        }
    
    # Perform PER-CAR spatial analysis
    car_analyses = analyze_spatial_violations(detections, image_width, image_height)
    
    if not car_analyses:
        return "Warning", f"Violations detected but unclear which car is responsible ({len(violations)} violations, {len(cars)} cars)", "Low", {
            'total_cars': len(cars),
            'total_violations': len(violations),
            'penalized_cars': []
        }
    
    # Find cars that have violations
    penalized_cars = []
    
    for car_analysis in car_analyses:
        if not car_analysis['has_violations']:
            continue
        
        # Get the most severe violation for this car
        car_violations = car_analysis['violations']
        if not car_violations:
            continue
        
        # Sort by confidence (highest first)
        car_violations.sort(key=lambda x: x['violation']['confidence'], reverse=True)
        top_violation_data = car_violations[0]
        
        violation = top_violation_data['violation']
        violation_class = violation['class']
        confidence = violation['confidence']
        
        # Get rule for this violation
        rule = PENALTY_RULES.get(violation_class, {'threshold': 0.70, 'penalty': 'Warning', 'severity': 'Low'})
        
        # Check if this violation exceeds the threshold
        if confidence >= rule['threshold']:
            car_position = "overlapping" if top_violation_data['overlap'] > 0.3 else "near"
            
            penalized_cars.append({
                'car_index': car_analysis['car_index'],
                'violation_class': violation_class,
                'confidence': confidence,
                'penalty': rule['penalty'],
                'severity': rule['severity'],
                'position': car_position,
                'overlap': top_violation_data['overlap'],
                'distance': top_violation_data['normalized_distance']
            })
    
    # Build result
    details = {
        'total_cars': len(cars),
        'total_violations': len(violations),
        'penalized_cars': penalized_cars,
        'cars_with_violations': len([ca for ca in car_analyses if ca['has_violations']])
    }
    
    if not penalized_cars:
        # Violations detected but none exceed threshold
        num_cars_with_violations = len([ca for ca in car_analyses if ca['has_violations']])
        reasoning = (f"Violations detected for {num_cars_with_violations} car(s) but none exceed confidence thresholds. "
                    f"Total: {len(violations)} violation(s), {len(cars)} car(s) in frame.")
        return "Warning", reasoning, "Low", details
    
    # Build reasoning for penalized cars
    if len(penalized_cars) == 1:
        pc = penalized_cars[0]
        reasoning = (f"Car #{pc['car_index']+1}: {pc['violation_class']} detected ({pc['position']}) "
                    f"with {pc['confidence']*100:.1f}% confidence. "
                    f"Penalty: {pc['penalty']}. "
                    f"({len(cars)} car(s) in frame, only car #{pc['car_index']+1} violated rules)")
        return pc['penalty'], reasoning, pc['severity'], details
    else:
        # Multiple cars penalized
        penalty_descriptions = [
            f"Car #{pc['car_index']+1}: {pc['violation_class']} ({pc['confidence']*100:.1f}%)"
            for pc in penalized_cars
        ]
        reasoning = (f"Multiple violations detected: {', '.join(penalty_descriptions)}. "
                    f"({len(cars)} car(s) in frame, {len(penalized_cars)} penalized)")
        
        # Return the most severe penalty
        max_severity = max(penalized_cars, key=lambda x: {'High': 3, 'Medium': 2, 'Low': 1}.get(x['severity'], 0))
        return max_severity['penalty'], reasoning, max_severity['severity'], details

def process_video_for_penalties(video_path, model, progress_bar, status_text, max_frames=300):
    """
    Process video and analyze each frame for violations with temporal consistency
    
    Args:
        video_path: Path to uploaded video
        model: Loaded YOLO model
        progress_bar: Streamlit progress bar
        status_text: Streamlit status text element
        max_frames: Maximum frames to process (default 300 = ~10 seconds at 30fps)
    
    Returns:
        dict with analysis results
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Limit processing for performance
    frames_to_process = min(total_frames, max_frames)
    
    # Track incidents across frames
    incidents = []
    frame_detections = []
    violation_timeline = defaultdict(list)  # Track each violation type over time
    
    # Multi-car tracking state
    car_violations = defaultdict(list)  # Track violations per car
    
    status_text.text(f"📹 Processing {frames_to_process} frames at {fps:.0f} FPS...")
    
    frame_idx = 0
    processed_frames = []
    
    while frame_idx < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress
        progress = (frame_idx + 1) / frames_to_process
        progress_bar.progress(progress)
        status_text.text(f"🔍 Analyzing frame {frame_idx + 1}/{frames_to_process}...")
        
        # Run inference
        results = model(frame, conf=0.25, verbose=False)
        
        # Parse detections
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for i in range(len(boxes)):
                detection = {
                    'class': result.names[int(boxes.cls[i])],
                    'confidence': float(boxes.conf[i]),
                    'bbox': boxes.xyxy[i].cpu().numpy().tolist(),
                    'frame': frame_idx,
                    'timestamp': frame_idx / fps
                }
                detections.append(detection)
        
        # Perform spatial analysis for this frame
        violation_analysis = analyze_spatial_violations(detections, width, height)
        
        # Track violations over time
        if violation_analysis:
            for v_data in violation_analysis:
                if v_data['is_associated']:
                    violation_class = v_data['violation']['class']
                    violation_timeline[violation_class].append({
                        'frame': frame_idx,
                        'timestamp': frame_idx / fps,
                        'confidence': v_data['violation']['confidence'],
                        'car_overlap': v_data['overlap']
                    })
                    
                    # Track per-car violations
                    car_id = f"car_{len([d for d in detections if d['class'] == 'Car'])}"
                    car_violations[car_id].append({
                        'type': violation_class,
                        'frame': frame_idx,
                        'confidence': v_data['violation']['confidence']
                    })
        
        # Check if this frame has significant violations
        violations = [d for d in detections if d['class'] != 'Car']
        if violations:
            decision, reasoning, severity, details = decide_penalty(detections, width, height)
            
            # Store ALL incidents (not just penalties) for analysis
            # Check if it's a real penalty (not "No Penalty" or just "Warning")
            is_penalty = ("Penalty" in decision and "No Penalty" not in decision) or severity in ['Medium', 'High']
            
            if is_penalty:
                # Store incident
                incidents.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / fps,
                    'decision': decision,
                    'reasoning': reasoning,
                    'severity': severity,
                    'details': details,
                    'detections': detections
                })
                
                # Annotate frame
                annotated_frame = result.plot() if len(results) > 0 else frame
                processed_frames.append({
                    'frame': annotated_frame,
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps
                })
        
        frame_detections.append({
            'frame': frame_idx,
            'detections': detections,
            'violation_count': len(violations)
        })
        
        frame_idx += 1
    
    cap.release()
    
    # Generate summary
    total_violations = sum(len(v) for v in violation_timeline.values())
    violation_summary = {vtype: len(frames) for vtype, frames in violation_timeline.items()}
    
    return {
        'total_frames': frames_to_process,
        'fps': fps,
        'duration': frames_to_process / fps,
        'incidents': incidents,
        'frame_detections': frame_detections,
        'violation_timeline': dict(violation_timeline),
        'violation_summary': violation_summary,
        'car_violations': dict(car_violations),
        'processed_frames': processed_frames[:10],  # Keep max 10 annotated frames
        'total_violations': total_violations
    }

def display_video_analysis():
    """Video upload and analysis tab with real-time penalty prediction"""
    st.markdown('<div class="main-header">🎬 Video Analysis & Penalty Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3 style="margin-top: 0;">🎯 Real-Time Race Video Analysis</h3>
        <p>Upload F1 race footage and our AI Race Director will analyze the video frame-by-frame, 
        detect incidents, track multi-car violations, and issue penalty decisions with temporal consistency.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check model availability
    model = load_model()
    
    if not YOLO_AVAILABLE or model is None:
        st.error("❌ **Model Not Available**")
        st.info("Please ensure the model is loaded. Go to Live Penalty Predictor for setup instructions.")
        return
    
    st.success("✅ **AI Race Director Ready** - Upload video to begin analysis")
    
    st.divider()
    
    # Layout
    col_upload, col_settings = st.columns([3, 2])
    
    with col_settings:
        st.markdown("### ⚙️ Analysis Settings")
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 10px;">
            <p><strong>Processing:</strong></p>
            <ul style="padding-left: 1.2rem; margin: 0.5rem 0;">
                <li>Frame-by-frame YOLOv8 detection</li>
                <li>Multi-car spatial analysis</li>
                <li>Temporal consistency tracking</li>
                <li>FIA-inspired penalty rules</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        max_frames = st.slider(
            "Maximum frames to process",
            min_value=30,
            max_value=600,
            value=300,
            step=30,
            help="Limit processing for performance. 300 frames ≈ 10 seconds at 30 FPS"
        )
        
        st.markdown("### 📊 What You'll Get")
        st.markdown("""
        <div class="detection-badge">📈 Violation Timeline</div>
        <div class="detection-badge">🏎️ Multi-Car Tracking</div>
        <div class="detection-badge">⚖️ Penalty Decisions</div>
        <div class="detection-badge">📸 Annotated Frames</div>
        <div class="detection-badge">📋 Incident Report</div>
        """, unsafe_allow_html=True)
    
    with col_upload:
        st.markdown("### 📤 Upload Race Video")
        
        st.markdown("""
        <div style="background: rgba(79, 172, 254, 0.1); padding: 0.5rem; border-radius: 5px; margin-bottom: 0.5rem;">
            <p style="margin: 0; font-size: 0.85rem;">
                ℹ️ <strong>Supported formats:</strong> MP4, AVI, MOV. 
                For best results, use high-quality footage with clear view of incidents.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov'],
            help="Upload F1 race footage for AI analysis"
        )
    
    if uploaded_video is not None:
        st.markdown("---")
        st.markdown("### 📹 Video Information")
        
        # Save uploaded video temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        with col_info1:
            st.metric("Duration", f"{duration:.1f}s")
        with col_info2:
            st.metric("FPS", f"{fps:.0f}")
        with col_info3:
            st.metric("Resolution", f"{width}×{height}")
        with col_info4:
            st.metric("Total Frames", total_frames)
        
        # Display video preview
        st.video(uploaded_video)
        
        # Analysis button
        if st.button("🚀 **START VIDEO ANALYSIS**", type="primary", use_container_width=True):
            st.markdown("---")
            st.markdown("## 🔄 Analysis In Progress...")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Process video
                results = process_video_for_penalties(
                    video_path, 
                    model, 
                    progress_bar, 
                    status_text,
                    max_frames
                )
                
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis Complete!")
                
                # DEBUG: Show what was detected
                total_violations_detected = sum(1 for fd in results['frame_detections'] if fd['violation_count'] > 0)
                st.info(f"🔍 **Debug Info:** {total_violations_detected} frames with violations detected out of {results['total_frames']} frames processed")
                
                st.success(f"🎉 **Video Analysis Complete!** Processed {results['total_frames']} frames in {results['duration']:.1f} seconds")
                
                # Display results - SIMPLE DECISION FORMAT
                st.markdown("---")
                
                # FINAL VERDICT
                penalty_incidents = [i for i in results['incidents'] if 'No Penalty' not in i['decision']]
                
                if penalty_incidents:
                    # PENALTY ISSUED
                    st.markdown("## � FINAL VERDICT: PENALTY ISSUED")
                    
                    # Show decision card
                    st.markdown(f"""
                    <div class="penalty-yes">
                        <h1 style="margin: 0; font-size: 4rem;">🚨</h1>
                        <h2 style="margin: 1rem 0;">PENALTY REQUIRED</h2>
                        <p style="font-size: 1.3rem; margin: 0;">{len(penalty_incidents)} penalty-worthy incident(s) detected</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 📍 Penalty Incident Details")
                    
                    # Show each penalty incident clearly
                    for idx, incident in enumerate(penalty_incidents, 1):
                        details = incident['details']
                        
                        # Create clear incident card
                        st.markdown(f"""
                        <div class="info-box" style="border-left: 5px solid #FF6B6B; background: rgba(255, 107, 107, 0.1);">
                            <h3 style="margin-top: 0; color: #FF6B6B;">Incident #{idx}</h3>
                            <table style="width: 100%; font-size: 1.1rem;">
                                <tr>
                                    <td style="padding: 0.5rem 0;"><strong>⏱️ Timestamp:</strong></td>
                                    <td style="text-align: right;"><strong>{incident['timestamp']:.2f} seconds</strong></td>
                                </tr>
                                <tr>
                                    <td style="padding: 0.5rem 0;"><strong>🎬 Frame Number:</strong></td>
                                    <td style="text-align: right;"><strong>Frame {incident['frame']}</strong></td>
                                </tr>
                                <tr>
                                    <td style="padding: 0.5rem 0;"><strong>⚖️ Decision:</strong></td>
                                    <td style="text-align: right; color: #FF6B6B;"><strong>{incident['decision']}</strong></td>
                                </tr>
                                <tr>
                                    <td style="padding: 0.5rem 0;"><strong>🏎️ Cars in Frame:</strong></td>
                                    <td style="text-align: right;">{details.get('total_cars', 0)} car(s)</td>
                                </tr>
                                <tr>
                                    <td style="padding: 0.5rem 0;"><strong>⚠️ Violations:</strong></td>
                                    <td style="text-align: right;">{details.get('associated_violations', 0)} associated with car(s)</td>
                                </tr>
                                <tr>
                                    <td style="padding: 0.5rem 0;"><strong>📊 Severity:</strong></td>
                                    <td style="text-align: right;"><span style="background: #FF6B6B; color: white; padding: 0.2rem 0.8rem; border-radius: 15px;">{incident['severity']}</span></td>
                                </tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show SPECIFIC PENALIZED CARS
                        penalized_cars = details.get('penalized_cars', [])
                        
                        if penalized_cars:
                            st.markdown("**🚨 Penalized Cars:**")
                            
                            for pc in penalized_cars:
                                col_v1, col_v2 = st.columns(2)
                                with col_v1:
                                    st.markdown(f"""
                                    <div style="background: rgba(255, 24, 1, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                                        <h4 style="margin: 0; color: #FF1801;">Car #{pc['car_index'] + 1}</h4>
                                        <h2 style="margin: 0.5rem 0;">{pc['violation_class']}</h2>
                                        <p style="margin: 0;">Confidence: {pc['confidence']*100:.1f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col_v2:
                                    st.markdown(f"""
                                    <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                                        <h4 style="margin: 0; color: #667eea;">Penalty Details</h4>
                                        <h2 style="margin: 0.5rem 0;">{pc['penalty']}</h2>
                                        <p style="margin: 0;">Severity: {pc['severity']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown(f"<p style='text-align: center; color: #888;'>Position: {pc['position']} • Overlap: {pc['overlap']*100:.1f}%</p>", unsafe_allow_html=True)
                        
                        # Reasoning
                        st.info(f"**Reasoning:** {incident['reasoning']}")
                        
                        # List all cars if multi-car scenario
                        if details.get('total_cars', 0) > 1:
                            st.markdown("**🏎️ Multi-Car Scenario:**")
                            st.write(f"• {details['total_cars']} cars detected in frame")
                            st.write(f"• {len(penalized_cars)} car(s) penalized")
                            
                            # Show which cars are NOT penalized
                            total_cars = details.get('total_cars', 0)
                            if len(penalized_cars) < total_cars:
                                st.success(f"✅ {total_cars - len(penalized_cars)} car(s) racing cleanly (no penalty)")
                        
                        if idx < len(penalty_incidents):
                            st.markdown("---")
                
                else:
                    # NO PENALTY
                    st.markdown("## ✅ FINAL VERDICT: NO PENALTY")
                    
                    st.markdown(f"""
                    <div class="penalty-no">
                        <h1 style="margin: 0; font-size: 4rem;">✅</h1>
                        <h2 style="margin: 1rem 0;">NO PENALTY REQUIRED</h2>
                        <p style="font-size: 1.3rem; margin: 0;">Racing incident - No penalty-worthy violations detected</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show summary of what was analyzed
                    st.markdown("### 📊 Analysis Summary")
                    
                    col_sum1, col_sum2, col_sum3 = st.columns(3)
                    with col_sum1:
                        st.metric("Frames Analyzed", results['total_frames'])
                    with col_sum2:
                        st.metric("Cars Detected", len(results['car_violations']))
                    with col_sum3:
                        st.metric("Minor Violations", results['total_violations'])
                    
                    if results['total_violations'] > 0:
                        st.warning(f"⚠️ **{results['total_violations']} violation(s) detected** but confidence was below penalty threshold or violations were not clearly associated with a specific car. See detailed timeline below for more info.")
                        
                        # Show which violations were detected
                        st.markdown("**Violations detected but below threshold:**")
                        for vtype, count in results['violation_summary'].items():
                            st.write(f"• {vtype}: {count} detections across frames")
                    else:
                        st.success("✅ Clean racing - no rule violations detected!")
                
                # Optional: Show detailed timeline
                with st.expander("📈 View Detailed Timeline & Breakdown (Optional)"):
                    if results['violation_summary']:
                        st.markdown("### 📈 Violation Timeline")
                        
                        fig = go.Figure()
                        
                        for violation_type, timeline in results['violation_timeline'].items():
                            if timeline:
                                timestamps = [t['timestamp'] for t in timeline]
                                confidences = [t['confidence'] for t in timeline]
                                
                                fig.add_trace(go.Scatter(
                                    x=timestamps,
                                    y=confidences,
                                    mode='markers+lines',
                                    name=violation_type,
                                    marker=dict(size=8),
                                    hovertemplate=f'{violation_type}<br>Time: %{{x:.2f}}s<br>Confidence: %{{y:.2%}}<extra></extra>'
                                ))
                        
                        fig.update_layout(
                            title="Violations Detected Over Time",
                            xaxis_title="Time (seconds)",
                            yaxis_title="Confidence",
                            hovermode='closest',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Violation summary bar chart
                        st.markdown("### 🏷️ Violation Breakdown")
                        
                        violation_df = pd.DataFrame([
                            {'Violation': vtype, 'Count': count}
                            for vtype, count in results['violation_summary'].items()
                        ])
                        
                        fig_bar = px.bar(
                            violation_df,
                            x='Violation',
                            y='Count',
                            color='Count',
                            color_continuous_scale='Reds'
                        )
                        fig_bar.update_layout(height=300)
                        st.plotly_chart(fig_bar, use_container_width=True)
                
                # Show key incident frames
                if results['processed_frames']:
                    st.markdown("---")
                    st.markdown("### 📸 Key Incident Frames")
                    st.markdown("Visual evidence from frames where penalties were detected:")
                    
                    cols = st.columns(min(3, len(results['processed_frames'])))
                    for idx, frame_data in enumerate(results['processed_frames'][:6]):
                        with cols[idx % 3]:
                            # Convert BGR to RGB for display
                            frame_rgb = cv2.cvtColor(frame_data['frame'], cv2.COLOR_BGR2RGB)
                            st.image(
                                frame_rgb,
                                caption=f"⚠️ Frame {frame_data['frame_idx']} ({frame_data['timestamp']:.2f}s)",
                                use_container_width=True
                            )
                
                # Simple export option
                st.markdown("---")
                
                col_export1, col_export2 = st.columns([3, 1])
                
                with col_export1:
                    st.markdown("### 📥 Export Decision")
                    st.write("Download the penalty decision summary as JSON for record-keeping.")
                
                with col_export2:
                    # Simplified report
                    penalty_incidents = [i for i in results['incidents'] if 'No Penalty' not in i['decision']]
                    
                    report = {
                        'verdict': 'PENALTY' if penalty_incidents else 'NO PENALTY',
                        'video_info': {
                            'filename': uploaded_video.name,
                            'duration': f"{results['duration']:.2f}s",
                            'frames_analyzed': results['total_frames']
                        },
                        'penalty_incidents': [
                            {
                                'frame': inc['frame'],
                                'timestamp': f"{inc['timestamp']:.2f}s",
                                'decision': inc['decision'],
                                'severity': inc['severity'],
                                'violation': inc['details'].get('primary_violation', {}).get('class', 'Unknown'),
                                'confidence': f"{inc['details'].get('primary_violation', {}).get('confidence', 0)*100:.1f}%",
                                'cars_in_frame': inc['details'].get('total_cars', 0),
                                'reasoning': inc['reasoning']
                            }
                            for inc in penalty_incidents
                        ] if penalty_incidents else [],
                        'summary': {
                            'total_violations_detected': results['total_violations'],
                            'penalty_worthy_incidents': len(penalty_incidents),
                            'cars_tracked': len(results['car_violations'])
                        }
                    }
                    
                    json_str = json.dumps(report, indent=2)
                    st.download_button(
                        label="📄 Download Decision",
                        data=json_str,
                        file_name=f"penalty_decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"❌ Error during video analysis: {str(e)}")
                st.info("💡 Try a different video or reduce max_frames setting")
            
            finally:
                # Cleanup temp file
                import os
                try:
                    os.unlink(video_path)
                except:
                    pass

def display_live_penalty_predictor():
    """Interactive tool for uploading images and predicting penalties"""
    st.markdown('<div class="main-header">🚨 Live Penalty Predictor</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3 style="margin-top: 0;">🎯 AI-Powered Incident Analysis</h3>
        <p>Upload an F1 race incident image and our trained YOLOv8 model will analyze it to detect violations 
        and determine if a penalty should be issued according to FIA-inspired rules.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if model is available
    model = load_model()
    
    if not YOLO_AVAILABLE or model is None:
        st.error("❌ **Model Not Available**")
        st.markdown("""
        <div class="info-box">
            <p><strong>Please ensure:</strong></p>
            <ol>
                <li>Install Ultralytics: <code>pip install ultralytics</code></li>
                <li>Model file exists: <code>model/best.pt</code> or <code>model/yolov8n.pt</code></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Model loaded successfully
    st.success("✅ **Model Loaded Successfully** - Ready for Analysis!")
    
    st.divider()
    
    # Create two columns layout
    col_upload, col_info = st.columns([3, 2])
    
    with col_info:
        st.markdown("### 📋 How It Works")
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 10px;">
            <ol style="padding-left: 1.2rem;">
                <li><strong>Upload</strong> an F1 incident image</li>
                <li><strong>Click</strong> "Analyze for Penalty"</li>
                <li><strong>AI detects</strong> violations using YOLOv8</li>
                <li><strong>Decision</strong> based on FIA rules</li>
                <li><strong>View</strong> annotated results</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🏷️ Detection Classes")
        classes = ["🏎️ Car", "🚧 Track Limits", "💥 Collision", 
                  "⚠️ Unsafe Release", "🏁 Flag Violation", "🏎️ Pit Lane Speeding"]
        for cls in classes:
            st.markdown(f"<span class='detection-badge'>{cls}</span>", unsafe_allow_html=True)
    
    with col_upload:
        # File uploader with custom styling
        st.markdown("### 📤 Upload Incident Image")
        
        st.markdown("""
        <div style="background: rgba(79, 172, 254, 0.1); padding: 0.5rem; border-radius: 5px; margin-bottom: 0.5rem;">
            <p style="margin: 0; font-size: 0.85rem;">
                ℹ️ <strong>Tip:</strong> Images will be automatically converted to RGB format. 
                Screenshots, PNG with transparency, and all common formats are supported.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image (JPG, JPEG, PNG)", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an F1 race incident image for AI analysis",
            label_visibility="collapsed"
        )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        # Display uploaded image
        st.markdown("---")
        st.markdown("### 📸 Uploaded Image Analysis")
        
        col_img, col_result = st.columns([3, 2])
        
        with col_img:
            st.image(image, caption=f"Incident Image: {uploaded_file.name}", use_container_width=True)
        
        # Analyze button
        analyze_btn = st.button("🔍 **ANALYZE FOR PENALTY**", type="primary", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("🔄 AI Analysis in Progress..."):
                try:
                    # Convert PIL to numpy array and handle RGBA/transparency
                    img_rgb = image.convert('RGB')  # Convert to RGB (handles RGBA, grayscale, etc.)
                    img_array = np.array(img_rgb)
                    
                    # Get image dimensions
                    img_height, img_width = img_array.shape[:2]
                    
                    # Verify image shape
                    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                        st.error(f"❌ Invalid image format. Expected RGB image, got shape: {img_array.shape}")
                        return
                    
                    # Run inference
                    results = model(img_array, conf=0.25, verbose=False)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.info("💡 Try uploading a different image or check if the model is properly loaded.")
                    return
                
                # Parse results
                detections = []
                if len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        detection = {
                            'class': result.names[int(boxes.cls[i])],
                            'confidence': float(boxes.conf[i]),
                            'bbox': boxes.xyxy[i].cpu().numpy().tolist()
                        }
                        detections.append(detection)
                
                # Decide penalty with spatial awareness
                decision, reasoning, severity, details = decide_penalty(detections, img_width, img_height)
                
                st.success("✅ Analysis Complete!")
                
                # Display decision with professional styling
                st.markdown("---")
                st.markdown("## ⚖️ FIA Steward Decision")
                
                col_decision, col_details = st.columns([2, 3])
                
                with col_decision:
                    # Color-coded decision card
                    if "No Penalty" in decision:
                        st.markdown(f"""
                        <div class="penalty-no">
                            <h1 style="margin: 0; font-size: 3rem;">✅</h1>
                            <h2 style="margin: 0.5rem 0;">{decision}</h2>
                            <p style="margin: 0; font-size: 1.2rem; opacity: 0.9;">Racing Incident</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif "Warning" in decision:
                        st.markdown(f"""
                        <div class="penalty-warning">
                            <h1 style="margin: 0; font-size: 3rem;">⚠️</h1>
                            <h2 style="margin: 0.5rem 0; color: #2d3436;">{decision}</h2>
                            <p style="margin: 0; font-size: 1.2rem; opacity: 0.9;">Monitored Incident</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="penalty-yes">
                            <h1 style="margin: 0; font-size: 3rem;">🚨</h1>
                            <h2 style="margin: 0.5rem 0;">{decision}</h2>
                            <p style="margin: 0; font-size: 1.2rem; opacity: 0.9;">Penalty Issued</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Severity indicator
                    severity_colors = {
                        'None': '#4ECDC4',
                        'Low': '#95E1D3',
                        'Medium': '#FDCB6E',
                        'High': '#FF6B6B'
                    }
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 1rem;">
                        <p style="font-size: 0.9rem; color: #888;">Severity Level</p>
                        <h3 style="color: {severity_colors.get(severity, '#888')}; margin: 0;">{severity}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_details:
                    st.markdown("### 📝 Decision Analysis")
                    st.markdown(f"""
                    <div class="info-box">
                        <p><strong>Reasoning:</strong></p>
                        <p>{reasoning}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 🔍 Detection Summary")
                    
                    # Display spatial analysis details
                    col_det1, col_det2, col_det3 = st.columns(3)
                    with col_det1:
                        st.metric("Total Detections", len(detections))
                    with col_det2:
                        st.metric("Cars in Frame", details.get('total_cars', 0))
                    with col_det3:
                        st.metric("Violations Found", details.get('total_violations', 0))
                    
                    # NEW: Show per-car penalty analysis
                    penalized_cars = details.get('penalized_cars', [])
                    
                    if penalized_cars:
                        st.markdown("#### 🏎️ **Penalized Cars:**")
                        for pc in penalized_cars:
                            st.markdown(f"""
                            <div class="info-box" style="border-left-color: #FF6B6B; background: rgba(255, 107, 107, 0.05);">
                                <p style="margin: 0;"><strong>Car #{pc['car_index'] + 1}</strong></p>
                                <p style="margin: 0.3rem 0;">• Violation: {pc['violation_class']}</p>
                                <p style="margin: 0.3rem 0;">• Confidence: {pc['confidence']*100:.1f}%</p>
                                <p style="margin: 0.3rem 0;">• Position: {pc['position']} (overlap: {pc['overlap']*100:.1f}%)</p>
                                <p style="margin: 0.3rem 0;">• Penalty: <strong>{pc['penalty']}</strong> (Severity: {pc['severity']})</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Show cars with violations but below threshold
                        cars_with_violations = details.get('cars_with_violations', 0)
                        if cars_with_violations > 0:
                            st.warning(f"⚠️ {cars_with_violations} car(s) detected with violations but below penalty threshold")
                    
                    # Show multi-car analysis if multiple cars present
                    if details.get('total_cars', 0) > 1:
                        st.info(f"🏎️ Multi-car scenario: {details['total_cars']} cars in frame. Analysis performed separately for each car.")
                
                # Detailed detections
                if detections:
                    st.markdown("---")
                    st.markdown("### 📊 Detailed Detection Results")
                    
                    # Create enhanced detection table
                    det_data = []
                    for i, d in enumerate(detections, 1):
                        det_data.append({
                            '#': i,
                            'Class': d['class'],
                            'Confidence': f"{d['confidence']*100:.1f}%",
                            'Type': 'Violation' if d['class'] != 'Car' else 'Object'
                        })
                    
                    det_df = pd.DataFrame(det_data)
                    st.dataframe(
                        det_df, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            '#': st.column_config.NumberColumn('ID', width='small'),
                            'Class': st.column_config.TextColumn('Detection Class', width='medium'),
                            'Confidence': st.column_config.TextColumn('Confidence', width='small'),
                            'Type': st.column_config.TextColumn('Type', width='small')
                        }
                    )
                    
                    # Show annotated image
                    st.markdown("### 🎯 Annotated Detection View")
                    if len(results) > 0:
                        # Get annotated image
                        annotated = results[0].plot()
                        # Convert BGR to RGB
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, caption="AI Detection Results with Bounding Boxes", 
                               use_container_width=True)
                else:
                    st.info("ℹ️ No objects detected in the uploaded image")
                
                # Penalty rules reference
                with st.expander("📋 View FIA-Inspired Penalty Rules"):
                    st.markdown("### Penalty Decision Guidelines")
                    st.markdown("Our AI uses these threshold-based rules to determine penalties:")
                    
                    rules_data = []
                    for violation, rule in PENALTY_RULES.items():
                        if violation != 'Car':
                            rules_data.append({
                                'Violation': violation,
                                'Confidence Threshold': f"{rule['threshold']*100:.0f}%",
                                'Typical Penalty': rule['penalty'],
                                'Severity': rule['severity']
                            })
                    
                    rules_df = pd.DataFrame(rules_data)
                    st.dataframe(rules_df, use_container_width=True, hide_index=True)
    
    else:
        # No file uploaded - show examples
        st.markdown("---")
        st.markdown("### 💡 No Image Uploaded Yet")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("👆 **Upload an incident image above to start AI analysis**")
            
            st.markdown("""
            <div class="info-box">
                <h4>Supported Formats:</h4>
                <ul>
                    <li>📸 JPEG / JPG</li>
                    <li>🖼️ PNG</li>
                </ul>
                <h4>Best Results:</h4>
                <ul>
                    <li>Clear view of incident</li>
                    <li>Multiple cars visible</li>
                    <li>Good lighting conditions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Show example from predictions
            st.markdown("### 🎬 Example Detection")
            pred_images = get_image_files('predictions')
            if pred_images and len(pred_images) > 0:
                example_img = Image.open(pred_images[0])
                st.image(example_img, caption="Example: Predicted incident detection", 
                        use_container_width=True)
                st.caption("You can test with similar F1 race incident images")
            else:
                st.info("No example images available. Run inference first to generate predictions.")
def main():
    """Main dashboard function"""
    
    # Sidebar with professional styling
    with st.sidebar:
        # Logo/Header
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #FF1801; font-size: 2.5rem; margin: 0;">🏎️</h1>
            <h2 style="color: #FF1801; margin: 0.5rem 0;">F1 Steward AI</h2>
            <p style="color: #888; font-size: 0.9rem;">Intelligent Penalty Prediction System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation
        st.markdown("### 📍 Navigation")
        page = st.radio(
            "Select View",
            ["📊 Overview", "🎯 Training Analysis", "🔍 Predictions", 
             "✅ Ground Truth", "📈 Visualizations", "🚨 Live Penalty Predictor",
             "🎬 Video Analysis"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Model Info Card
        summary = load_summary()
        perf = summary.get('performance', {})
        
        st.markdown("### 🤖 Model Info")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; color: white;">
            <p style="margin: 0.3rem 0;"><strong>Architecture:</strong> YOLOv8n</p>
            <p style="margin: 0.3rem 0;"><strong>mAP@0.5:</strong> {perf.get('mAP@0.5', 0)*100:.1f}%</p>
            <p style="margin: 0.3rem 0;"><strong>Speed:</strong> {perf.get('inference_speed_ms', 0):.1f} ms</p>
            <p style="margin: 0.3rem 0;"><strong>Status:</strong> ✅ Ready</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Dataset Info
        st.markdown("### 📦 Dataset")
        dataset = summary.get('dataset', {})
        st.write(f"📸 **Images:** {dataset.get('original_images', 'N/A')}")
        st.write(f"🏷️ **Classes:** {len(dataset.get('classes', []))}")
        st.write(f"📅 **Updated:** {summary.get('timestamp', 'N/A')[:10]}")
        
        st.divider()
        
        # Footer
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8rem; padding: 1rem 0;">
            <p>Powered by YOLOv8 & FastF1</p>
            <p>Dell Hackathon 2024</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content based on selection
    if "Overview" in page:
        display_overview()
    elif "Training" in page:
        display_training_analysis()
    elif "Predictions" in page:
        display_predictions()
    elif "Ground Truth" in page:
        display_ground_truth()
    elif "Visualizations" in page:
        display_visualizations()
    elif "Live Penalty Predictor" in page:
        display_live_penalty_predictor()
    elif "Video Analysis" in page:
        display_video_analysis()

if __name__ == "__main__":
    main()
