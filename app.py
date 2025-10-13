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

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'anomaly_detection_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for processing
processing_queue = queue.Queue()
is_processing = False
current_detector = None
current_tracker = None
yolo_model = None
next_track_id = 1  # Global counter for unique track IDs

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
        
        print(f"✓ Model loaded successfully")
        print(f"  Device: {self.device}")
        print(f"  Classes: {self.class_names}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Confidence threshold: {self.confidence_threshold}")
    
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
                        print(f"Track {track_id}: {self.class_names[pred.item()]} ({conf.item():.3f})")
        
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
                    print("✓ SORT tracker initialized with max_age=30, min_hits=3, iou_threshold=0.3")
                else:
                    # Reset existing tracker for new video
                    current_tracker.reset()
                    print("✓ SORT tracker reset for new video")
                
                # Process video
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                
                while cap.isOpened() and is_processing:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    frame_shape = frame.shape[:2]
                    
                    # Run YOLO detection
                    results = yolo_model(frame, conf=0.25, classes=[0])  # Only detect people
                    
                    detections = []
                    tracked_objects = np.empty((0, 5))
                    
                    # Debug: Print YOLO results (every 10 frames to avoid spam)
                    if frame_count % 10 == 0:
                        print(f"Frame {frame_count}: YOLO Results: {len(results)} result(s)")
                    
                    if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                        if frame_count % 10 == 0:
                            print(f"Frame {frame_count}: Number of boxes detected: {len(results[0].boxes)}")
                        
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
                                            print(f"Frame {frame_count} Detection {i}: bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) conf={conf:.3f}")
                                except Exception as e:
                                    print(f"Frame {frame_count} Error processing box {i}: {e}")
                                    continue
                            
                            if len(detections_yolo) > 0:
                                detections_yolo = np.stack(detections_yolo)
                                if frame_count % 10 == 0:
                                    print(f"Frame {frame_count}: Processing {len(detections_yolo)} valid detections")
                                
                                try:
                                    # Update SORT tracker
                                    tracked_objects = current_tracker.update(detections_yolo)
                                    if frame_count % 10 == 0:
                                        print(f"Frame {frame_count}: Tracked objects shape: {tracked_objects.shape}")
                                        if len(tracked_objects) > 0:
                                            print(f"Frame {frame_count}: Track IDs: {tracked_objects[:, 4] if len(tracked_objects.shape) > 1 else 'N/A'}")
                                except Exception as e:
                                    print(f"Frame {frame_count} Error in SORT tracking: {e}")
                                    # Create fallback with proper track IDs
                                    tracked_objects = np.column_stack([detections_yolo, np.arange(len(detections_yolo)) + 1])
                    else:
                        if frame_count % 10 == 0:
                            print(f"Frame {frame_count}: No valid boxes detected")
                    
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
                                    print(f"Frame {frame_count} Added tracked detection {i}: track_id={int(track_id)}, bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
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
                                    print(f"Frame {frame_count} Added fallback detection {i}: track_id={next_track_id-1}, bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                        except Exception as e:
                            print(f"Frame {frame_count} Error processing track {i}: {e}")
                            continue
                    
                    if frame_count % 10 == 0:
                        print(f"Frame {frame_count}: Total detections processed: {len(detections)}")
                    
                    # Get anomaly predictions (only if we have detections)
                    predictions = {}
                    if detections:
                        try:
                            predictions = current_detector.process_frame(detections, frame_shape)
                            
                            # Add alerts for anomalies
                            for track_id, pred in predictions.items():
                                if pred['class_name'] != 'Normal':
                                    current_detector.add_alert(track_id, pred, current_detector.frame_count)
                                    # Emit alert to frontend
                                    socketio.emit('alert', {
                                        'track_id': track_id,
                                        'class_name': pred['class_name'],
                                        'confidence': pred['confidence'],
                                        'timestamp': time.time()
                                    })
                        except Exception as e:
                            print(f"Error in anomaly detection: {e}")
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
                        'frame_count': frame_count
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
                print(f"Error processing video: {e}")
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
        print("✓ Moved index.html to templates directory")
    
    print("Starting Flask server...")
    print("Dashboard will be available at: http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
