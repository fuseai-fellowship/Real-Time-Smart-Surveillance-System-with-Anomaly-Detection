# Real-Time-Smart-Surveillance-System-with-Anomaly-Detection

This repository contains a simple, yet functional, prototype for a real-time human anomaly detection and tracking system in crowded environments. The system is designed to identify and follow individuals whose behavior deviates from a learned "normal" pattern.

The project is built on a two-stage pipeline:

1. **Perception:** Detect and track individuals in a video stream.  
2. **Analysis:** Analyze their movement for anomalous behavior.

---

## Core Technologies

- **Object Detection:** YOLOv8, a state-of-the-art model for real-time object detection.  
- **Multi-Object Tracking:** DeepSORT, a robust algorithm that maintains unique IDs for individuals across frames, even during occlusions.  
- **Anomaly Detection:** An Autoencoder model, which is a neural network trained to recognize "normal" patterns. It identifies anomalies by failing to reconstruct a pattern it has never seen before, using the reconstruction error as an anomaly score.  
- **Development Framework:** PyTorch for building and training the anomaly detection model.  
- **Video Processing:** OpenCV for handling video I/O.  

---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

First, ensure you have Python 3.8+ installed on your system. It is highly recommended to use a virtual environment to manage project dependencies.

#### Create a virtual environment
```bash
python3 -m venv venv
