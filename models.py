"""Initialize models for the dashboard"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='0'

def init_models(yolo_model="yolov8s.pt"):
    """Initialize YOLO model"""
    import torch
    from ultralytics import YOLO
    
    # Initialize YOLO
    model = YOLO(yolo_model)
    return model

def init_tracker():
    """Initialize SORT tracker"""
    from sort import Sort
    return Sort(max_age=30, min_hits=3, iou_threshold=0.3)