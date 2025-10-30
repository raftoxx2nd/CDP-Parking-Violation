# Parking Management UB – Real-Time Violation Detection

## Chapter 3: Hierarchical & Iterative Design

### 3.1 System Overview & Architecture (Brief)

The system is designed to monitor parking zones on a university campus (Universitas Brawijaya) and detect unauthorized parking (e.g., motorbikes parked in car‐zones) in real-time, using computer-vision and local PC processing.
Primary components:

* **Input Layer**: IP/ESP32-CAM cameras streaming video.
* **Processing Layer**: PC running a pretrained YOLOv11 model for object detection; logic module checks zone membership and stationary time for violations.
* **Output Layer**: Web dashboard that displays alerts, logs violations, snapshots and provides admin controls.
* **Control & Data Layer**: Zone definition module, configuration database, user interface for defining polygons and parking zone metadata.

Architecture follows a layered design:

1. **Capture & Stream Layer** – inputs from camera endpoints.
2. **Detection & Logic Layer** – executes object detection, ROI, tracking, zone logic.
3. **Notification & Dashboard Layer** – handles alerting, dashboard UI and logs.
4. **Configuration & Data Layer** – supports zone definitions, system parameters and historical data.

### 3.2 Extra Context

- Use Ultralytics official documentation for code or task implementation if possible
- There will be seperate web dashboard or front-end layer

