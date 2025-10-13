import torch
import torch.nn as nn
import numpy as np
from collections import deque, defaultdict
import time
import json
import streamlit as st
import pandas as pd
import tempfile
import io
from PIL import Image
import traceback
import cv2

# Import our model initialization
from models import init_models, init_tracker

# Import the model architecture (ensure this matches your training script)
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
    """
    Real-time anomaly detection for surveillance system
    Integrates with YOLOv8 + SORT pipeline
    """
    
    def __init__(self, model_path, config_path=None, device='cuda', confidence_threshold=0.5):
        """
        Args:
            model_path: Path to trained model (.pt file)
            config_path: Path to config JSON (optional)
            device: 'cuda' or 'cpu'
            confidence_threshold: Minimum confidence for alert triggering
        """
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
        """
        Process a frame of detections and return anomaly predictions
        
        Args:
            detections: List of {'bbox': (x1,y1,x2,y2), 'confidence': float, 'track_id': int}
            frame_shape: (height, width) tuple
            roi_bounds: Optional (x1,y1,x2,y2) for restricted zone
            
        Returns:
            predictions: Dict of track_id -> (class_id, confidence, class_name)
        """
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
        """
        Add a new alert to the system
        """
        alert = {
            'track_id': track_id,
            'frame_idx': frame_idx,
            'timestamp': time.time(),
            'class_name': prediction['class_name'],
            'confidence': prediction['confidence'],
        }
        self.alerts.append(alert)
        
    def get_active_alerts(self, max_age_seconds=30):
        """
        Get list of active alerts within the specified time window
        """
        current_time = time.time()
        return [alert for alert in self.alerts 
                if current_time - alert['timestamp'] < max_age_seconds]
    
    def get_statistics(self):
        """
        Return detection statistics
        """
        stats = {
            'frame_count': self.frame_count,
            'avg_inference_time': np.mean(self.inference_times[-100:]) if self.inference_times else 0,
            'fps': 1.0 / (np.mean(self.inference_times[-100:]) if self.inference_times else 1),
            'total_alerts': len(self.alerts),
            'active_tracks': len(self.track_buffers),
        }
        return stats


def main():
    st.title("Real-time Anomaly Detection Dashboard")
    
    # Check if all required packages are available
    required_packages = {
        "ultralytics": "YOLO object detection",
        "opencv-python": "Video processing",
        "torch": "Deep learning",
        "numpy": "Numerical computations",
        "pandas": "Data processing"
    }
    
    missing_packages = []
    for package, description in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(f"- {package} ({description})")
    
    if missing_packages:
        st.error("Missing required packages:")
        for pkg in missing_packages:
            st.write(pkg)
        st.info("Please run the following command in your terminal:")
        packages_str = " ".join(required_packages.keys())
        st.code(f"pip install {packages_str}")
        return
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    model_path = st.sidebar.selectbox(
        "Select Anomaly Detection Model",
        ["rnn_anomaly_best.pt", "rnn_anomaly_final.pt"]
    )
    yolo_model = st.sidebar.selectbox(
        "Select YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        index=1
    )
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5
    )
    
    # Initialize models
    try:
        with st.spinner("Loading models..."):
            yolo = init_models()
            st.sidebar.success("✓ YOLO model loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Failed to load models: {str(e)}")
        st.error(f"Error details:\n{traceback.format_exc()}")
        return
    
    # Initialize detector
    detector = RealTimeAnomalyDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold
    )
    
    # Video input
    video_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi'])
    
    if video_file is not None:
        # Create a placeholder for the video frame
        frame_placeholder = st.empty()
        
        # Statistics columns
        col1, col2, col3 = st.columns(3)
        fps_metric = col1.empty()
        tracks_metric = col2.empty()
        alerts_metric = col3.empty()
        
        # Alert section
        st.subheader("Active Alerts")
        alert_placeholder = st.empty()
        
        # Initialize SORT tracker with better parameters
        from sort import Sort
        tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        print("✓ SORT tracker initialized with max_age=30, min_hits=3, iou_threshold=0.3")
        
        # Read and process video
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        # Reset tracker for new video to ensure unique IDs
        tracker.reset()
        print("✓ SORT tracker reset for new video")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run YOLOv8 detection
            frame_shape = frame.shape[:2]
            results = yolo(frame, conf=0.25, classes=[0])  # Only detect people
            
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes
                detections_yolo = []
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections_yolo.append(np.array([x1, y1, x2, y2, conf]))
                
                if len(detections_yolo) > 0:
                    detections_yolo = np.stack(detections_yolo)
                    # Update SORT tracker
                    tracked_objects = tracker.update(detections_yolo)
                else:
                    tracked_objects = np.empty((0, 5))
                    
                # Convert tracked objects to detection format for anomaly detection
                detections = []
                for track in tracked_objects:
                    x1, y1, x2, y2, track_id = track
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': 1.0,  # Tracking confidence
                        'track_id': int(track_id),
                        'class': 'person'
                    })
            
            # Get predictions
            predictions = detector.process_frame(detections, frame_shape)
            
            # Add alerts for anomalies
            for track_id, pred in predictions.items():
                if pred['class_name'] != 'Normal':
                    detector.add_alert(track_id, pred, detector.frame_count)
            
            # Update statistics
            stats = detector.get_statistics()
            fps_metric.metric("FPS", f"{stats['fps']:.1f}")
            tracks_metric.metric("Active Tracks", stats['active_tracks'])
            alerts_metric.metric("Total Alerts", stats['total_alerts'])
            
            # Display active alerts
            active_alerts = detector.get_active_alerts()
            if active_alerts:
                alert_df = pd.DataFrame(active_alerts)
                alert_placeholder.table(alert_df)
            
            # Draw detections and tracking
            frame_vis = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = [int(x) for x in det['bbox']]
                track_id = det['track_id']
                
                # Draw bounding box
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw ID
                cv2.putText(frame_vis, f"ID: {track_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # If there's an anomaly prediction for this track_id, draw it
                if track_id in predictions:
                    pred = predictions[track_id]
                    if pred['class_name'] != 'Normal':
                        label = f"{pred['class_name']} ({pred['confidence']:.2f})"
                        cv2.putText(frame_vis, label, (x1, y2+20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Display frame
            frame_rgb = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
            
            # Control frame rate
            time.sleep(0.03)  # ~30 FPS
        
        cap.release()
        
    else:
        st.warning("Please upload a video file to begin analysis.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())