# GitHub Copilot Instructions

## 🚀 Project Overview

This project is a real-time parking violation detection system. It uses a YOLOv8 object detection model to identify vehicles in designated parking zones from a video stream. If a vehicle remains in a zone for a specified duration, the system logs a violation and saves a snapshot.

The application is built with Python and relies heavily on `OpenCV` for video processing and `ultralytics` for object detection.

## 🏛️ Architecture

The system is composed of a few key modules:

-   **`main.py`**: The main entry point. It orchestrates the entire process: loading the configuration, initializing the YOLO model, capturing the video stream, processing frames, and detecting violations.
-   **`processing/define_roi.py`**: A crucial utility script for setting up the system. It allows a user to interactively draw polygonal "regions of interest" (ROIs) on a frame from the video source. These polygons define the parking zones.
-   **`config/zones.json`**: This file stores the coordinates of the parking zones defined by `define_roi.py`. It is a critical input for `main.py`.
-   **`capture/stream_handler.py`**: A helper module responsible for connecting to and capturing frames from the video source (e.g., a webcam, IP camera stream, or video file).
-   **`yolo11n.pt`**: The pre-trained YOLO model file used for vehicle detection.
-   **`output/`**: This directory is where all the results are stored.
    -   `snapshots/`: Contains image snapshots of detected violations.
    -   `logs/`: Contains JSON files with details about each violation event.

## 🔄 Developer Workflow

### 1. Setup

The project uses a Python virtual environment. Key dependencies are not yet listed in `requirements.txt`, but they include `opencv-python`, `ultralytics`, `numpy`, and `matplotlib`.

To set up the environment:

```bash
# Create and activate a virtual environment (example)
python -m venv .venv
source .venv/bin/activate # On Linux/macOS
.venv\Scripts\activate # On Windows

# Install dependencies
pip install opencv-python ultralytics numpy matplotlib
```

### 2. Defining Parking Zones (Required First Step)

Before running the main detection, you must define the parking zones.

```bash
python processing/define_roi.py
```

This will open an interactive window with a frame from the video source. Follow the on-screen instructions to draw polygons for each parking zone. When you close the window, the zones will be saved to `config/zones.json`.

### 3. Running Violation Detection

Once the zones are defined, run the main application:

```bash
python main.py
```

The system will start processing the video feed. Any detected violations will be saved to the `output/` directory.

## ⚙️ Key Conventions & Patterns

-   **Configuration**: Most operational parameters (video source URL, model path, violation time threshold) are hardcoded as constants at the top of `main.py`. When adding new features, prefer adding new constants there rather than magic strings/numbers in the code.
-   **Violation Logic**: A violation is determined by tracking how long the *center point* of a detected vehicle's bounding box remains inside a defined zone. The core logic is in the main loop of `run_violation_detection()` in `main.py`.
-   **Output Format**: Each violation triggers two outputs: a JPG snapshot and a JSON log file, both timestamped and saved in the `output/` directory. The JSON log contains the violation time, zone name, and vehicle ID.
-   **GPU/CPU Agnostic**: The code checks for CUDA availability and automatically moves the YOLO model to the GPU if possible, falling back to the CPU otherwise. This is handled by the `DEVICE` constant in `main.py`.
