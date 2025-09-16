import cv2
import torch
import numpy as np
from collections import defaultdict, deque
import math
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class AnomalyDetector:
    """
    Advanced Anomaly Detection System that analyzes multiple behavioral patterns
    """
    def __init__(self):
        self.track_history = defaultdict(lambda: {
            'positions': deque(maxlen=50),
            'timestamps': deque(maxlen=50),
            'speeds': deque(maxlen=20),
            'directions': deque(maxlen=20),
            'stillness_counter': 0,
            'erratic_movement_counter': 0,
            'last_position': None,
            'first_seen': None
        })
        
        # Anomaly thresholds (tunable parameters)
        self.STILLNESS_THRESHOLD = 30  # frames of no movement
        self.SPEED_THRESHOLD_HIGH = 50  # pixels per frame (running)
        self.SPEED_THRESHOLD_LOW = 2   # pixels per frame (very slow)
        self.DIRECTION_CHANGE_THRESHOLD = 5  # frequent direction changes
        self.LOITERING_TIME_THRESHOLD = 300  # seconds (5 minutes)
        self.ERRATIC_MOVEMENT_THRESHOLD = 10  # counter for erratic behavior
        
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def calculate_speed(self, positions, timestamps):
        """Calculate average speed over recent positions"""
        if len(positions) < 2:
            return 0
        
        total_distance = 0
        total_time = 0
        
        for i in range(1, len(positions)):
            distance = self.calculate_distance(positions[i-1], positions[i])
            time_diff = timestamps[i] - timestamps[i-1]
            if time_diff > 0:
                total_distance += distance
                total_time += time_diff
        
        return total_distance / total_time if total_time > 0 else 0
    
    def calculate_direction_changes(self, positions):
        """Calculate number of significant direction changes"""
        if len(positions) < 3:
            return 0
        
        direction_changes = 0
        prev_angle = None
        
        for i in range(2, len(positions)):
            # Calculate angle of movement
            dx1 = positions[i-1][0] - positions[i-2][0]
            dy1 = positions[i-1][1] - positions[i-2][1]
            dx2 = positions[i][0] - positions[i-1][0]
            dy2 = positions[i][1] - positions[i-1][1]
            
            if dx1 == 0 and dy1 == 0:
                continue
            if dx2 == 0 and dy2 == 0:
                continue
                
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            
            angle_diff = abs(angle2 - angle1)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            
            # If direction change is significant (> 60 degrees)
            if angle_diff > math.pi / 3:
                direction_changes += 1
                
        return direction_changes
    
    def detect_anomalies(self, track_id, center_point, current_time):
        """
        Main anomaly detection logic
        Returns: (anomaly_type, confidence_score)
        """
        history = self.track_history[track_id]
        
        # Initialize first seen time
        if history['first_seen'] is None:
            history['first_seen'] = current_time
        
        # Update position history
        history['positions'].append(center_point)
        history['timestamps'].append(current_time)
        
        anomalies = []
        
        # 1. Stillness Detection (Loitering)
        if history['last_position'] is not None:
            movement = self.calculate_distance(center_point, history['last_position'])
            
            if movement < 5:  # Very small movement (pixel threshold)
                history['stillness_counter'] += 1
            else:
                history['stillness_counter'] = 0
            
            if history['stillness_counter'] > self.STILLNESS_THRESHOLD:
                time_spent = current_time - history['first_seen']
                if time_spent > self.LOITERING_TIME_THRESHOLD:
                    anomalies.append(("LOITERING", 0.9))
                else:
                    anomalies.append(("STILLNESS", 0.7))
        
        # 2. Speed Analysis
        if len(history['positions']) >= 3:
            speed = self.calculate_speed(
                list(history['positions']), 
                list(history['timestamps'])
            )
            history['speeds'].append(speed)
            
            if speed > self.SPEED_THRESHOLD_HIGH:
                anomalies.append(("RUNNING", 0.8))
            elif speed < self.SPEED_THRESHOLD_LOW and len(history['positions']) > 10:
                anomalies.append(("SUSPICIOUS_SLOW", 0.6))
        
        # 3. Erratic Movement Detection
        if len(history['positions']) >= 10:
            direction_changes = self.calculate_direction_changes(
                list(history['positions'])[-10:]  # Last 10 positions
            )
            
            if direction_changes >= self.DIRECTION_CHANGE_THRESHOLD:
                history['erratic_movement_counter'] += 1
                
                if history['erratic_movement_counter'] > self.ERRATIC_MOVEMENT_THRESHOLD:
                    anomalies.append(("ERRATIC_MOVEMENT", 0.75))
            else:
                history['erratic_movement_counter'] = max(0, history['erratic_movement_counter'] - 1)
        
        # 4. Boundary Analysis (if person is at edges too long)
        frame_width, frame_height = 1280, 720  # Default, should be updated with actual frame size
        edge_threshold = 50
        
        if (center_point[0] < edge_threshold or center_point[0] > frame_width - edge_threshold or
            center_point[1] < edge_threshold or center_point[1] > frame_height - edge_threshold):
            anomalies.append(("BOUNDARY_LURKING", 0.6))
        
        history['last_position'] = center_point
        
        # Return the most severe anomaly
        if anomalies:
            anomalies.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence
            return anomalies[0]
        
        return ("NORMAL", 0.0)

class HumanAnomalyDetectionSystem:
    def __init__(self, source=0, confidence_threshold=0.5):
        """
        Initialize the complete anomaly detection system
        
        Args:
            source: Video source (0 for webcam, path for video file)
            confidence_threshold: Minimum confidence for YOLO detections
        """
        # Load YOLOv8 model
        print("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')
        
        # Initialize DeepSORT tracker
        print("Initializing DeepSORT tracker...")
        self.tracker = DeepSort(
            max_age=50,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.2
        )
        
        # Initialize anomaly detector
        self.anomaly_detector = AnomalyDetector()
        
        # Video capture
        self.cap = cv2.VideoCapture(source)
        self.confidence_threshold = confidence_threshold
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Video initialized: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        
        # Colors for different anomaly types
        self.colors = {
            'NORMAL': (0, 255, 0),      # Green
            'LOITERING': (0, 0, 255),   # Red
            'STILLNESS': (0, 165, 255), # Orange
            'RUNNING': (255, 0, 255),   # Magenta
            'SUSPICIOUS_SLOW': (0, 255, 255),  # Yellow
            'ERRATIC_MOVEMENT': (128, 0, 128), # Purple
            'BOUNDARY_LURKING': (255, 165, 0)  # Blue
        }
        
        # Statistics
        self.total_detections = 0
        self.anomaly_count = defaultdict(int)
        self.frame_count = 0
        
    def process_frame(self, frame):
        """Process a single frame for anomaly detection"""
        current_time = time.time()
        self.frame_count += 1
        
        # Run YOLO detection
        results = self.model(frame, classes=[0], verbose=False)  # Class 0 is 'person'
        
        # Prepare detections for DeepSORT
        detections = []
        if len(results[0].boxes.data) > 0:
            for detection in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
                
                if conf >= self.confidence_threshold:
                    w = x2 - x1
                    h = y2 - y1
                    detections.append(([x1, y1, w, h], conf, cls))
        
        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Process each track
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = [int(coord) for coord in ltrb]
            
            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center_point = (center_x, center_y)
            
            # Detect anomalies
            anomaly_type, confidence = self.anomaly_detector.detect_anomalies(
                track_id, center_point, current_time
            )
            
            # Update statistics
            if anomaly_type != "NORMAL":
                self.anomaly_count[anomaly_type] += 1
            
            # Get color for visualization
            color = self.colors.get(anomaly_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(frame, center_point, 3, color, -1)
            
            # Draw trail (last few positions)
            if track_id in self.anomaly_detector.track_history:
                positions = list(self.anomaly_detector.track_history[track_id]['positions'])
                for i in range(1, len(positions)):
                    cv2.line(frame, 
                           tuple(map(int, positions[i-1])), 
                           tuple(map(int, positions[i])), 
                           color, 1)
            
            # Draw labels
            label = f'ID:{track_id} | {anomaly_type}'
            if confidence > 0:
                label += f' ({confidence:.2f})'
            
            # Calculate label size and draw background
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(frame, 
                         (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1), 
                         color, -1)
            
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw statistics
        self.draw_statistics(frame)
        
        return frame, len(tracks)
    
    def draw_statistics(self, frame):
        """Draw system statistics on frame"""
        stats_y = 30
        
        # System info
        cv2.putText(frame, f"Frame: {self.frame_count}", 
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        stats_y += 25
        cv2.putText(frame, f"Active Tracks: {len(self.anomaly_detector.track_history)}", 
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Anomaly statistics
        stats_y += 30
        cv2.putText(frame, "ANOMALY DETECTIONS:", 
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        for anomaly_type, count in self.anomaly_count.items():
            stats_y += 20
            color = self.colors.get(anomaly_type, (255, 255, 255))
            cv2.putText(frame, f"{anomaly_type}: {count}", 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def run(self):
        """Main execution loop"""
        print("Starting Human Anomaly Detection System...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    print("Failed to read frame or reached end of video")
                    break
                
                # Process frame
                processed_frame, person_count = self.process_frame(frame)
                
                # Display result
                cv2.imshow('Human Anomaly Detection System', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    filename = f"anomaly_detection_{int(time.time())}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Screenshot saved as {filename}")
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            print("\n=== FINAL STATISTICS ===")
            print(f"Total frames processed: {self.frame_count}")
            print(f"Total unique persons tracked: {len(self.anomaly_detector.track_history)}")
            print("Anomaly detections:")
            for anomaly_type, count in self.anomaly_count.items():
                print(f"  {anomaly_type}: {count}")

# Example usage
if __name__ == "__main__":
    # For webcam, use source=0
    # For video file, use source='path/to/video.mp4'
    detector_system = HumanAnomalyDetectionSystem(source='/home/leapfrog/Downloads/fuse_machine_project/test.mp4')
    
    # Initialize system with webcam
    # detector_system = HumanAnomalyDetectionSystem(
    #     source=0,  # Change to video file path if needed
    #     confidence_threshold=0.5
    # )
    
    # Run the system
    detector_system.run()