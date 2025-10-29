#!/usr/bin/env python3
"""
Flask backend for the anomaly detection dashboard
Handles video processing and real-time communication
"""

import os
import json
import time
import base64
import tempfile
import threading
import queue
import logging
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from collections import deque, defaultdict
import torch
import torch.nn as nn

# Import our modules
from models import init_models
from sort import Sort
from loitering import LoiterDetector

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'anomaly_detection_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for processing
processing_queue = queue.Queue()
is_processing = False
# Use an Event to control pausing to avoid tight busy-wait loops
pause_event = threading.Event()
current_detector = None
current_tracker = None
yolo_model = None
next_track_id = 1  # Global counter for unique track IDs
# Loitering detector (lightweight, time + small movement)
loiter_detector = LoiterDetector(loiter_seconds=10.0, min_disp_pixels=20.0, max_gap_seconds=1.0, fps=30)
# Fall smoothing and bbox history to reduce false positives from ID switches
FALL_VOTE_K = 5  # number of recent frames to consider for vote smoothing
FALL_VOTE_THRESHOLD = 0.6  # fraction of votes required to accept model fall
fall_vote_history = defaultdict(lambda: deque(maxlen=FALL_VOTE_K))
bbox_history = defaultdict(lambda: deque(maxlen=6))  # keep recent bboxes per track for heuristics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Model architecture (same as dashboard.py)
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class AnomalyDetectionRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=6, dropout=0.3, use_attention=True):
        super(AnomalyDetectionRNN, self).__init__()
        
        self.use_attention = use_attention
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        if use_attention:
            self.attention = AttentionLayer(hidden_dim * 2)
            classifier_input_dim = hidden_dim * 2
        else:
            classifier_input_dim = hidden_dim * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.feature_extractor(x)
        lstm_out, _ = self.lstm(x)
        
        if self.use_attention:
            context_vector, _ = self.attention(lstm_out)
        else:
            context_vector = lstm_out[:, -1, :]
        
        output = self.classifier(context_vector)
        return output

class RealTimeAnomalyDetector:
    def __init__(self, model_path, config_path=None, device='cpu', confidence_threshold=0.5):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = checkpoint.get('config', {})
        
        # Initialize model
        self.model = AnomalyDetectionRNN(
            input_dim=config.get('FEATURE_DIM', 25),
            hidden_dim=config.get('HIDDEN_DIM', 128),
            num_layers=config.get('NUM_LAYERS', 2),
            num_classes=config.get('NUM_CLASSES', 6),
            dropout=config.get('DROPOUT', 0.3),
            use_attention=True
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Configuration
        self.sequence_length = config.get('SEQUENCE_LENGTH', 30)
        self.feature_dim = config.get('FEATURE_DIM', 25)
        self.class_names = checkpoint.get('class_names', 
                                         ['Normal', 'Loitering', 'Boundary_Crossing', 
                                          'Fall', 'Aggression', 'Abandoned_Object'])
        
        # Tracking buffers for each object ID
        self.track_buffers = defaultdict(lambda: deque(maxlen=self.sequence_length))
        self.track_metadata = defaultdict(dict)
        
        # Statistics
        self.frame_count = 0
        self.inference_times = []
        self.alerts = []
        
        logging.info("✓ Model loaded successfully")
        logging.info(f"  Device: {self.device}")
        logging.info(f"  Classes: {self.class_names}")
        logging.info(f"  Sequence length: {self.sequence_length}")
        logging.info(f"  Confidence threshold: {self.confidence_threshold}")
    
    def extract_features(self, detection, track_id, frame_idx, frame_shape, roi_bounds=None):
        """
        Extract features from detection (original format for trained model)
        
        Args:
            detection: YOLO detection dict with 'bbox', 'confidence', 'class'
            track_id: SORT tracking ID
            frame_idx: Current frame index
            frame_shape: (height, width) for normalization
            roi_bounds: (x1, y1, x2, y2) for restricted zone
        
        Returns:
            features: numpy array of shape (feature_dim,)
        """
        bbox = detection['bbox']  # (x1, y1, x2, y2)
        x1, y1, x2, y2 = bbox
        
        # Normalize coordinates
        height, width = frame_shape
        x1_norm = x1 / width
        y1_norm = y1 / height
        x2_norm = x2 / width
        y2_norm = y2 / height
        
        # Basic features
        w = x2_norm - x1_norm
        h = y2_norm - y1_norm
        center_x = (x1_norm + x2_norm) / 2
        center_y = (y1_norm + y2_norm) / 2
        area = w * h
        aspect_ratio = w / (h + 1e-6)
        
        # Motion features
        if track_id in self.track_metadata and 'prev_center' in self.track_metadata[track_id]:
            prev_cx, prev_cy = self.track_metadata[track_id]['prev_center']
            dx = center_x - prev_cx
            dy = center_y - prev_cy
            speed = np.sqrt(dx**2 + dy**2)
            direction = np.arctan2(dy, dx)
        else:
            dx, dy, speed, direction = 0, 0, 0, 0
        
        # Detect sudden large normalized displacement (possible ID switch) and reset history
        # Threshold is in normalized coordinates (0..1). Tune RESET_DISP_THRESHOLD as needed.
        RESET_DISP_THRESHOLD = 0.45
        if 'prev_center' in self.track_metadata[track_id]:
            try:
                if speed > RESET_DISP_THRESHOLD:
                    # Clear track buffer to avoid mixing histories from ID switches
                    try:
                        self.track_buffers[track_id].clear()
                    except Exception:
                        self.track_buffers[track_id] = deque(maxlen=self.sequence_length)
                    # Reset metadata for this track
                    self.track_metadata[track_id] = {'prev_center': (center_x, center_y), 'first_seen': frame_idx}
                    logging.info(f"Track {track_id}: large jump detected (disp={speed:.3f}) -> buffer reset")
            except Exception:
                # If anything goes wrong, ensure prev_center is updated below
                pass

        # Update metadata
        self.track_metadata[track_id]['prev_center'] = (center_x, center_y)
        
        # ROI detection
        roi_flag = 0
        if roi_bounds is not None:
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
            roi_x1_norm = roi_x1 / width
            roi_y1_norm = roi_y1 / height
            roi_x2_norm = roi_x2 / width
            roi_y2_norm = roi_y2 / height
            
            if (x1_norm < roi_x1_norm or x2_norm > roi_x2_norm or 
                y1_norm < roi_y1_norm or y2_norm > roi_y2_norm):
                roi_flag = 1
        
        # Time in region (for loitering detection)
        if 'first_seen' not in self.track_metadata[track_id]:
            self.track_metadata[track_id]['first_seen'] = frame_idx
        time_in_region = (frame_idx - self.track_metadata[track_id]['first_seen']) / 30.0
        
        # Compile feature vector (original format - exactly 25 features)
        features = np.array([
            x1_norm, y1_norm, x2_norm, y2_norm,  # Position (4)
            w, h, area, aspect_ratio,            # Size (4)
            center_x, center_y,                   # Center (2)
            dx, dy, speed, direction,            # Motion (4)
            detection['confidence'],              # Detection confidence (1)
            roi_flag,                            # ROI violation (1)
            time_in_region,                      # Time in frame (1)
            # Pad remaining features to reach 25 total
            *[0] * (self.feature_dim - 17)       # Pad remaining features
        ])
        
        return features
    
        
    def process_frame(self, detections, frame_shape, roi_bounds=None):
        start_time = time.time()
        self.frame_count += 1
        predictions = {}
        
        # Process each detection
        for det in detections:
            track_id = det['track_id']
            
            # Extract features
            features = self.extract_features(det, track_id, self.frame_count, frame_shape, roi_bounds)
            
            # Add to track buffer
            self.track_buffers[track_id].append(features)
            
            # Make prediction if we have enough history
            if len(self.track_buffers[track_id]) >= self.sequence_length:
                # Prepare input sequence
                sequence = np.array(list(self.track_buffers[track_id])[-self.sequence_length:])
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                
                # Get model prediction
                with torch.no_grad():
                    output = self.model(sequence_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    conf, pred = torch.max(probabilities, dim=1)
                    
                    # Use the trained model's prediction directly
                    predictions[track_id] = {
                        'class_id': pred.item(),
                        'confidence': conf.item(),
                        'class_name': self.class_names[pred.item()]
                    }
                    
                    # Debug output to see what the model is predicting
                    if track_id % 10 == 0 or conf.item() > 0.3:  # Log every 10th track or high confidence
                        logging.debug(f"Track {track_id}: {self.class_names[pred.item()]} ({conf.item():.3f})")
        
        # Record inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return predictions
        
    def add_alert(self, track_id, prediction, frame_idx):
        alert = {
            'track_id': track_id,
            'frame_idx': frame_idx,
            'timestamp': time.time(),
            'class_name': prediction['class_name'],
            'confidence': prediction['confidence'],
        }
        self.alerts.append(alert)
        
    def get_active_alerts(self, max_age_seconds=30):
        current_time = time.time()
        return [alert for alert in self.alerts 
                if current_time - alert['timestamp'] < max_age_seconds]
    
    def get_statistics(self):
        stats = {
            'frame_count': self.frame_count,
            'avg_inference_time': np.mean(self.inference_times[-100:]) if self.inference_times else 0,
            'fps': 1.0 / (np.mean(self.inference_times[-100:]) if self.inference_times else 1),
            'total_alerts': len(self.alerts),
            'active_tracks': len(self.track_buffers),
        }
        return stats

def process_video_worker():
    """Background worker for video processing"""
    global is_processing, current_detector, current_tracker, yolo_model, next_track_id
    
    while True:
        if not processing_queue.empty():
            video_path, config = processing_queue.get()
            
            try:
                # Reset track ID counter for new video
                next_track_id = 1
                
                # Initialize models if not already done
                if yolo_model is None:
                    yolo_model = init_models(config['yolo_model'])
                
                if current_detector is None:
                    current_detector = RealTimeAnomalyDetector(
                        model_path=config['model'],
                        confidence_threshold=config['confidence_threshold']
                    )
                
                # Always reset tracker for new video to ensure unique IDs
                if current_tracker is None:
                    # Initialize SORT tracker with better parameters for tracking
                    current_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
                    logging.info("✓ SORT tracker initialized with max_age=30, min_hits=3, iou_threshold=0.3")
                    # Reset loiter detector state for new video
                    loiter_detector.reset()
                else:
                    # Reset existing tracker for new video
                    current_tracker.reset()
                    # Reset loiter detector together with tracker
                    loiter_detector.reset()
                    logging.info("✓ SORT tracker reset for new video")
                
                # Process video
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                
                while cap.isOpened() and is_processing:
                    # If paused, wait using pause_event to avoid busy-wait
                    while pause_event.is_set() and is_processing:
                        # wait with timeout so we can respond to resume/stop
                        pause_event.wait(0.1)
                        continue
                        
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    frame_shape = frame.shape[:2]
                    
                    # Run YOLO detection
                    results = yolo_model(frame, conf=0.25, classes=[0])  # Only detect people
                    
                    detections = []
                    tracked_objects = np.empty((0, 5))
                    
                    # Debug: Print YOLO results at debug level
                    if frame_count % 10 == 0:
                        logging.debug(f"Frame {frame_count}: YOLO Results: {len(results)} result(s)")
                    
                    if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                        if frame_count % 10 == 0:
                            logging.debug(f"Frame {frame_count}: Number of boxes detected: {len(results[0].boxes)}")
                        
                        if len(results[0].boxes) > 0:
                            boxes = results[0].boxes
                            detections_yolo = []
                            
                            for i, box in enumerate(boxes):
                                try:
                                    # Get coordinates and confidence
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    conf = box.conf[0].cpu().numpy()
                                    
                                    # Ensure coordinates are valid
                                    if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
                                        detections_yolo.append(np.array([x1, y1, x2, y2, conf]))
                                        if frame_count % 10 == 0:
                                            logging.debug(f"Frame {frame_count} Detection {i}: bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) conf={conf:.3f}")
                                except Exception as e:
                                    logging.exception(f"Frame {frame_count} Error processing box {i}")
                                    continue
                            
                            if len(detections_yolo) > 0:
                                detections_yolo = np.stack(detections_yolo)
                                if frame_count % 10 == 0:
                                    logging.debug(f"Frame {frame_count}: Processing {len(detections_yolo)} valid detections")
                                
                                try:
                                    # Update SORT tracker
                                    tracked_objects = current_tracker.update(detections_yolo)
                                    if frame_count % 10 == 0:
                                        logging.debug(f"Frame {frame_count}: Tracked objects shape: {tracked_objects.shape}")
                                        if len(tracked_objects) > 0:
                                            logging.debug(f"Frame {frame_count}: Track IDs: {tracked_objects[:, 4] if len(tracked_objects.shape) > 1 else 'N/A'}")
                                except Exception:
                                    logging.exception(f"Frame {frame_count} Error in SORT tracking")
                                    # Create fallback with proper track IDs
                                    tracked_objects = np.column_stack([detections_yolo, np.arange(len(detections_yolo)) + 1])
                        else:
                            if frame_count % 10 == 0:
                                logging.debug(f"Frame {frame_count}: No valid boxes detected")
                    
                    # Convert tracked objects to detection format
                    for i, track in enumerate(tracked_objects):
                        try:
                            if len(track) >= 5:  # Ensure track has all required elements
                                x1, y1, x2, y2, track_id = track[:5]
                                
                                # Use SORT's track_id directly (it should be unique after reset)
                                # SORT returns track_id + 1, so we use it as-is
                                detections.append({
                                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                                    'confidence': 1.0,
                                    'track_id': int(track_id),
                                    'class': 'person'
                                })
                                if frame_count % 10 == 0:
                                    logging.debug(f"Frame {frame_count} Added tracked detection {i}: track_id={int(track_id)}, bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                            elif len(track) >= 4:  # Fallback for raw detections without track_id
                                x1, y1, x2, y2 = track[:4]
                                # Only use fallback if SORT completely fails
                                detections.append({
                                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                                    'confidence': 1.0,
                                    'track_id': next_track_id,  # Use global counter as last resort
                                    'class': 'person'
                                })
                                next_track_id += 1
                                if frame_count % 10 == 0:
                                    logging.debug(f"Frame {frame_count} Added fallback detection {i}: track_id={next_track_id-1}, bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                        except Exception:
                            logging.exception(f"Frame {frame_count} Error processing track {i}")
                            continue
                    
                    if frame_count % 10 == 0:
                        logging.debug(f"Frame {frame_count}: Total detections processed: {len(detections)}")
                    
                    # Get anomaly predictions (only if we have detections)
                    predictions = {}
                    loitering_tracks = []
                    if detections:
                        # First, update loiter detector state for each detection so it can track
                        for det in detections:
                            try:
                                track_id = det['track_id']
                                bbox = det['bbox']
                                # keep bbox history for heuristics / debugging
                                try:
                                    bbox_history[track_id].append(bbox)
                                except Exception:
                                    bbox_history[track_id] = deque([bbox], maxlen=6)
                                # update returns True when loitering threshold is reached this frame
                                if loiter_detector.update(track_id, bbox, frame_count):
                                    loitering_tracks.append(track_id)
                            except Exception:
                                logging.exception(f"Frame {frame_count} Error updating loiter detector for track {det.get('track_id')}")

                        try:
                            # Map of track_id -> bbox for saving debug crops when alerts are generated
                            track_bboxes = {det['track_id']: det['bbox'] for det in detections}

                            predictions = current_detector.process_frame(detections, frame_shape)

                            # Handle model-predicted anomalies with special smoothing for 'Fall'
                            for track_id, pred in predictions.items():
                                try:
                                    cls = pred.get('class_name', 'Normal')
                                    conf = pred.get('confidence', 0.0)

                                    # Maintain fall vote history for smoothing
                                    if cls == 'Fall':
                                        try:
                                            fall_vote_history[track_id].append(1 if conf >= current_detector.confidence_threshold else 0)
                                        except Exception:
                                            fall_vote_history[track_id] = deque([1 if conf >= current_detector.confidence_threshold else 0], maxlen=FALL_VOTE_K)
                                    else:
                                        # push 0 for non-fall to keep window aligned
                                        fall_vote_history[track_id].append(0)

                                    # For non-fall anomalies, emit immediately if confident
                                    if cls != 'Normal' and cls != 'Fall' and conf >= current_detector.confidence_threshold:
                                        current_detector.add_alert(track_id, pred, current_detector.frame_count)
                                        socketio.emit('alert', {
                                            'track_id': track_id,
                                            'class_name': cls,
                                            'confidence': conf,
                                            'timestamp': time.time()
                                        })

                                        # Save debug crop for inspection
                                        try:
                                            debug_dir = os.path.join(os.getcwd(), 'debug_alerts')
                                            os.makedirs(debug_dir, exist_ok=True)
                                            bbox = track_bboxes.get(track_id)
                                            if bbox is not None:
                                                x1, y1, x2, y2 = map(int, bbox)
                                                h, w = frame.shape[:2]
                                                x1, y1 = max(0, x1), max(0, y1)
                                                x2, y2 = min(w-1, x2), min(h-1, y2)
                                                crop = frame[y1:y2, x1:x2]
                                                if crop.size != 0:
                                                    fname = f"alert_track{track_id}_{cls}_f{frame_count}.jpg"
                                                    cv2.imwrite(os.path.join(debug_dir, fname), crop)
                                        except Exception:
                                            logging.exception(f"Error saving debug crop for track {track_id}")
                                except Exception:
                                    logging.exception(f"Error handling prediction alert for track {track_id}")

                            # Handle loitering alerts from the lightweight detector (override or augment predictions)
                            for track_id in loitering_tracks:
                                try:
                                    # Create a loitering prediction object
                                    loiter_pred = {
                                        'class_id': None,
                                        'confidence': 1.0,
                                        'class_name': 'Loitering'
                                    }
                                    # Record in the model's alert list for visibility
                                    current_detector.add_alert(track_id, loiter_pred, current_detector.frame_count)
                                    # Update predictions so frontend sees loitering for this track
                                    predictions[int(track_id)] = loiter_pred
                                    # Emit alert to frontend
                                    socketio.emit('alert', {
                                        'track_id': track_id,
                                        'class_name': 'Loitering',
                                        'confidence': 1.0,
                                        'timestamp': time.time()
                                    })

                                    # Save debug crop for loitering
                                    try:
                                        debug_dir = os.path.join(os.getcwd(), 'debug_alerts')
                                        os.makedirs(debug_dir, exist_ok=True)
                                        bbox = track_bboxes.get(track_id)
                                        if bbox is not None:
                                            x1, y1, x2, y2 = map(int, bbox)
                                            h, w = frame.shape[:2]
                                            x1, y1 = max(0, x1), max(0, y1)
                                            x2, y2 = min(w-1, x2), min(h-1, y2)
                                            crop = frame[y1:y2, x1:x2]
                                            if crop.size != 0:
                                                fname = f"loiter_track{track_id}_f{frame_count}.jpg"
                                                cv2.imwrite(os.path.join(debug_dir, fname), crop)
                                    except Exception:
                                        logging.exception(f"Error saving debug crop for loitering track {track_id}")
                                except Exception:
                                    logging.exception(f"Frame {frame_count} Error handling loiter alert for track {track_id}")

                            # Now evaluate fall detection using smoothed votes + simple bbox-based heuristic
                            for track_id in list(fall_vote_history.keys()):
                                try:
                                    votes = list(fall_vote_history[track_id])
                                    vote_frac = sum(votes) / len(votes) if votes else 0.0

                                    # Simple bbox-based heuristic: large downward movement or height drop
                                    heuristic_flag = False
                                    bh = list(bbox_history.get(track_id, []))
                                    if len(bh) >= 2:
                                        # use last two bboxes
                                        x1a, y1a, x2a, y2a = bh[-2]
                                        x1b, y1b, x2b, y2b = bh[-1]
                                        ha = (y2a - y1a)
                                        hb = (y2b - y1b)
                                        ca = (y1a + y2a) / 2.0
                                        cb = (y1b + y2b) / 2.0
                                        # normalized by frame height
                                        frame_h = frame.shape[0]
                                        # downward center movement (pixels -> normalized)
                                        dy_norm = (cb - ca) / (frame_h + 1e-6)
                                        # relative height change
                                        if ha > 0:
                                            height_drop = (ha - hb) / ha
                                        else:
                                            height_drop = 0

                                        # Heuristic thresholds (tunable)
                                        if dy_norm > 0.05 or height_drop > 0.25:
                                            heuristic_flag = True

                                    # Decide final fall detection: either strong vote or heuristic
                                    fall_detected = False
                                    if vote_frac >= FALL_VOTE_THRESHOLD:
                                        fall_detected = True
                                    elif heuristic_flag:
                                        # if heuristic triggers, allow lower vote requirement
                                        fall_detected = True

                                    if fall_detected:
                                        # emit fall alert (one-time per alerted status)
                                        # avoid duplicate alerts: check current_detector.alerts
                                        already_alerted = any(a['track_id'] == track_id and a['class_name'] == 'Fall' for a in current_detector.alerts)
                                        if not already_alerted:
                                            # Build synthetic prediction to record
                                            fall_pred = {'class_id': None, 'confidence': max(0.5, vote_frac), 'class_name': 'Fall'}
                                            current_detector.add_alert(track_id, fall_pred, current_detector.frame_count)
                                            socketio.emit('alert', {
                                                'track_id': track_id,
                                                'class_name': 'Fall',
                                                'confidence': fall_pred['confidence'],
                                                'timestamp': time.time()
                                            })

                                            # Save debug crop for fall
                                            try:
                                                debug_dir = os.path.join(os.getcwd(), 'debug_alerts')
                                                os.makedirs(debug_dir, exist_ok=True)
                                                bbox = track_bboxes.get(track_id)
                                                if bbox is not None:
                                                    x1, y1, x2, y2 = map(int, bbox)
                                                    h, w = frame.shape[:2]
                                                    x1, y1 = max(0, x1), max(0, y1)
                                                    x2, y2 = min(w-1, x2), min(h-1, y2)
                                                    crop = frame[y1:y2, x1:x2]
                                                    if crop.size != 0:
                                                        fname = f"fall_track{track_id}_f{frame_count}.jpg"
                                                        cv2.imwrite(os.path.join(debug_dir, fname), crop)
                                            except Exception:
                                                logging.exception(f"Error saving debug crop for fall track {track_id}")
                                except Exception:
                                    logging.exception(f"Error evaluating fall for track {track_id}")

                        except Exception:
                            logging.exception("Error in anomaly detection")
                            predictions = {}
                    
                    # Convert frame to base64 for frontend (resize for performance)
                    resized_frame = cv2.resize(frame, (640, 360))  # Resize for better performance
                    _, buffer = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Prepare detection result
                    detection_result = {
                        'frame': frame_base64,
                        'detections': [
                            {
                                'bbox': det['bbox'],
                                'track_id': det['track_id'],
                                'class_name': predictions.get(det['track_id'], {}).get('class_name', 'Normal'),
                                'confidence': predictions.get(det['track_id'], {}).get('confidence', 0.0)
                            }
                            for det in detections
                        ],
                        'processing_time': current_detector.inference_times[-1] if current_detector.inference_times else 0,
                        'frame_count': frame_count,
                        # include original and resized frame sizes so client can scale boxes correctly
                        'orig_frame_size': {'width': frame.shape[1], 'height': frame.shape[0]},
                        'resized_frame_size': {'width': 640, 'height': 360}
                    }
                    
                    # Emit to frontend every 3 frames to improve performance
                    if frame_count % 3 == 0:
                        socketio.emit('detection_result', detection_result)
                    
                    # Also emit on first frame to show initial state
                    if frame_count == 1:
                        socketio.emit('detection_result', detection_result)
                    
                    # Update statistics every 30 frames to reduce overhead
                    if frame_count % 30 == 0:
                        stats = current_detector.get_statistics()
                        socketio.emit('statistics_update', stats)
                    
                    # Control frame rate - reduce to improve performance
                    time.sleep(0.1)  # ~10 FPS for better performance
                
                cap.release()
                
            except Exception as e:
                logging.exception("Error processing video")
                socketio.emit('error', {'message': str(e)})
            
            processing_queue.task_done()

# Start background worker
worker_thread = threading.Thread(target=process_video_worker, daemon=True)
worker_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    global is_processing
    
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'})
        
        video_file = request.files['video']
        config = json.loads(request.form['config'])
        
        # Save video to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_file.save(temp_file.name)
        temp_file.close()
        
        # Add to processing queue
        processing_queue.put((temp_file.name, config))
        is_processing = True
        
        return jsonify({'success': True, 'message': 'Video processing started'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    global is_processing
    is_processing = False
    return jsonify({'success': True, 'message': 'Processing stopped'})

@app.route('/pause_processing', methods=['POST'])
def pause_processing():
    try:
        logging.info("Received pause request")
        pause_event.set()
        logging.info("Processing paused (pause_event set)")
        return jsonify({'success': True, 'message': 'Processing paused'})
    except Exception as e:
        logging.exception("Error in pause_processing")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/resume_processing', methods=['POST'])
def resume_processing():
    try:
        logging.info("Received resume request")
        pause_event.clear()
        logging.info("Processing resumed (pause_event cleared)")
        return jsonify({'success': True, 'message': 'Processing resumed'})
    except Exception as e:
        logging.exception("Error in resume_processing")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/status')
def get_status():
    return jsonify({
        'is_processing': is_processing,
        'queue_size': processing_queue.qsize()
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Move index.html to templates directory if it exists and templates/index.html doesn't
    if os.path.exists('index.html') and not os.path.exists('templates/index.html'):
        os.rename('index.html', 'templates/index.html')
        logging.info("✓ Moved index.html to templates directory")
    
    logging.info("Starting Flask server...")
    logging.info("Dashboard will be available at: http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
