# AI Coding Agent Instructions for Parking Violation Detection Project

## 1. Project Overview & Architecture

This project is a computer vision system designed to detect parking violations. It uses the YOLOv8 object detection model to identify vehicles and checks if they are located within user-defined parking zones.

The architecture is composed of two main Python scripts and one critical JSON configuration file:

- **`define_roi.py`**: This is the **first step** in the workflow. It's a command-line tool that allows a user to interactively draw polygonal "parking zones" on a source image using a Matplotlib window. The script saves the zone coordinates and source image dimensions into `parking_zones.json`.

- **`parking_zones.json`**: This file is the crucial link between defining zones and detecting violations. It contains the geometric definitions for all parking slots and the resolution of the image they were defined on. This resolution is used to scale the zones correctly for different-sized input images during detection.

- **`run_detection.py`**: This is the **second step**. It performs the actual violation detection. It loads the YOLO model (`yolov8n.pt`) and the zones from `parking_zones.json`, processes a target image, and identifies violations. A violation is defined as a detected `motorcycle` or `motorbike` whose bounding box center falls within any of the defined zones.

- **`website/`**: This directory is currently empty but is intended for a future web-based interface.

## 2. Core Developer Workflow

The standard workflow is a two-step process:

**Step 1: Define Parking Zones (if not already done)**
Run `define_roi.py` from the terminal. You will be prompted to provide the path to a background image. An interactive window will open where you can click to draw polygons.

```bash
python define_roi.py
```
> **Note:** This will create or overwrite `parking_zones.json`.

**Step 2: Run Violation Detection**
Run `run_detection.py` and provide the path to an image you want to analyze.

```bash
python run_detection.py --image path/to/your/image.jpg
```
The script will then:
1. Load the zones from `parking_zones.json`.
2. Scale the zones to match the input image's resolution.
3. Run the YOLOv8 model to detect objects.
4. Check for motorcycles inside the zones.
5. Save an annotated image (e.g., `violation_...jpg`) and a `violation_...json` file if violations are found.

## 3. Key Conventions & Patterns

- **Zone Scaling**: A key feature is the automatic scaling of parking zones. `run_detection.py` reads the source image dimensions from `parking_zones.json` and calculates scaling factors (`sx`, `sy`) to apply the zones to input images of any resolution. See the `scale_zones` function in `run_detection.py`.

- **Violation Logic**: A violation is determined by the `box_overlaps_any_zone` function. It uses `cv2.pointPolygonTest` on the **center point** of a detected object's bounding box. It is not a full IoU (Intersection over Union) check.

- **Dependencies**: The project relies on `ultralytics`, `opencv-python`, `numpy`, and `matplotlib`. Ensure these are installed in your environment.

- **Model**: The object detection model is `yolov8n.pt`, which is included in the repository.
