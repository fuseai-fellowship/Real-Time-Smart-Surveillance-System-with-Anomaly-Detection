# 🏗️ Video Anomaly Detection Web App Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Flask Web Application                        │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (HTML/CSS/JS)          │  Backend (Python/Flask)      │
│  ┌─────────────────────────────┐  │  ┌─────────────────────────┐  │
│  │ • Video Upload Interface   │  │  │ • File Upload Handler   │  │
│  │ • Video Preview Player     │  │  │ • Video Processing      │  │
│  │ • Results Visualization    │  │  │ • Anomaly Detection     │  │
│  │ • Interactive Charts       │  │  │ • Model Integration      │  │
│  └─────────────────────────────┘  │  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AI Processing Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│  Video Input → Frame Extraction → Feature Extraction → Analysis │
│      │              │                    │              │      │
│      ▼              ▼                    ▼              ▼      │
│  Video File    Individual Frames    ResNet18 Features  RNN    │
│  (MP4/AVI)     (JPG Images)         (512-dim vectors)  Model  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Results & Visualization                   │
├─────────────────────────────────────────────────────────────────┤
│  • Anomaly Probabilities  • Timeline Charts  • Statistics     │
│  • Interactive Plots     • Detailed Reports  • Export Options │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Frontend Components
- **Upload Interface**: Drag & drop file upload with validation
- **Video Player**: HTML5 video element for preview
- **Results Dashboard**: Statistics cards and interactive charts
- **Responsive Design**: Mobile-friendly interface

### 2. Backend Components
- **Flask App** (`app.py`): Main application server
- **Upload Handler**: Secure file upload and storage
- **Video Processor**: Frame extraction and preprocessing
- **Model Integration**: AI model loading and inference

### 3. AI Pipeline
- **Frame Extraction**: OpenCV-based video processing
- **Feature Extraction**: ResNet18 CNN for visual features
- **Sequence Creation**: Temporal grouping of features
- **Anomaly Detection**: RNN with attention mechanism

### 4. Data Flow
```
Video Upload → Frame Extraction → Feature Extraction → Sequence Creation → Anomaly Detection → Visualization
```

## File Structure

```
├── app.py                    # Main Flask application
├── run_app.py               # Startup script
├── demo.py                  # Demo and testing script
├── requirements_flask.txt   # Python dependencies
├── templates/
│   └── index.html          # Web UI template
├── uploads/                # Uploaded video storage
├── temp/                   # Temporary processing files
└── src/                    # Existing model code
    ├── models/
    │   └── anomaly_detector.py
    └── feature_extractor.py
```

## Key Features

### 🎥 Video Processing
- **Supported Formats**: MP4, AVI, MOV, MKV
- **Frame Extraction**: Configurable FPS and quality
- **Feature Extraction**: ResNet18-based CNN
- **Sequence Processing**: Temporal analysis with RNN

### 🤖 AI Model
- **Architecture**: Bidirectional LSTM with attention
- **Input**: 512-dimensional feature vectors
- **Output**: Anomaly probability scores
- **Training**: Focal loss for class imbalance

### 📊 Visualization
- **Interactive Charts**: Plotly-based timeline plots
- **Real-time Results**: Live processing updates
- **Statistics Dashboard**: Comprehensive metrics
- **Responsive Design**: Mobile and desktop support

### 🔧 Technical Specifications
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Visualization**: Plotly.js
- **AI Framework**: PyTorch
- **Video Processing**: OpenCV
- **Deployment**: Local development server

## Usage Workflow

1. **Upload**: User uploads video file via web interface
2. **Processing**: Server extracts frames and features
3. **Analysis**: AI model processes sequences for anomalies
4. **Visualization**: Results displayed with interactive charts
5. **Export**: Users can view detailed reports and statistics

## Performance Considerations

- **Memory Usage**: Configurable frame limits
- **Processing Time**: Depends on video length and resolution
- **GPU Support**: CUDA acceleration when available
- **Scalability**: Single-threaded processing (can be optimized)

## Security Features

- **File Validation**: Video format and size checks
- **Secure Upload**: Werkzeug secure filename handling
- **Temporary Storage**: Automatic cleanup of processing files
- **Error Handling**: Graceful failure with user feedback
