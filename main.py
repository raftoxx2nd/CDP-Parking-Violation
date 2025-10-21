import cv2
import numpy as np
from ultralytics import YOLO
import json
import time
import os
from collections import defaultdict
import torch
import sys

# Add project root to the Python path to allow sibling imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from capture.stream_handler import get_video_capture

# --- Configuration ---
ZONES_FILE = 'config/zones.json'
MODEL_PATH = 'yolo11n.pt'
# Set a fixed video source path or URL here. 
# If None, the script will prompt the user for input.
# Example: "rtsp://my-stream-url" or "path/to/video.mp4"
FIXED_VIDEO_SOURCE = "https://192.168.1.15:8080/video"
# Confidence and IoU thresholds for object detection
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
# Time in seconds a vehicle must be in a zone to be a violation
VIOLATION_THRESHOLD_SECONDS = 10 
# Output directories for logs and snapshots
SNAPSHOT_DIR = 'output/snapshots'
LOG_DIR = 'output/logs'
# Set device to 'cuda' if GPU is available, otherwise 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Setup ---
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Utility Functions ---

def load_zones(filepath=ZONES_FILE):
    """Loads zone data from the specified JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Zone file not found: '{filepath}'. Please run the ROI definition script first.")
    with open(filepath, 'r') as f:
        data = json.load(f)
    # Convert zone lists back to numpy arrays for OpenCV
    zones = {name: np.array(poly, dtype=np.int32) for name, poly in data['zones'].items()}
    return zones, data['source_image_width'], data['source_image_height']

def scale_zones(zones, sx, sy):
    """Scales polygon coordinates by the given scaling factors."""
    scaled = {}
    for name, poly in zones.items():
        scaled[name] = np.array([[int(x * sx), int(y * sy)] for [x, y] in poly], dtype=np.int32)
    return scaled

def box_center_in_zone(box, zones):
    """Checks if the center of a bounding box is inside any of the defined zones."""
    x1, y1, x2, y2 = box
    center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    for name, poly in zones.items():
        if cv2.pointPolygonTest(poly, center_point, False) >= 0:
            return True, name  # Return True and the name of the zone
    return False, None

# --- Core Processing ---

def run_violation_detection():
    """
    Main loop to run real-time parking violation detection.
    """
    # 1. Load Zones
    try:
        original_zones, src_w, src_h = load_zones()
        print(f"✅ Successfully loaded zones from '{ZONES_FILE}'.")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # 2. Initialize YOLO Model
    try:
        model = YOLO(MODEL_PATH)
        model.to(DEVICE) # Move model to the selected device (GPU or CPU)
        print(f"✅ YOLOv8 model loaded on device: '{DEVICE}'.")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        print("Please ensure the model name in MODEL_PATH is correct and you have an internet connection.")
        return

    # 3. Setup Video Capture
    video_source = FIXED_VIDEO_SOURCE
    if video_source is None:
        video_source = input("Enter video source (e.g., '0' for webcam, or path to video file): ")
    
    try:
        print(f"Connecting to video source: {video_source}")
        cap = get_video_capture(video_source)
    except IOError as e:
        print(f"❌ Error: {e}")
        return

    # Get video properties for scaling and thresholding
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # Default FPS if it cannot be determined
    violation_threshold_frames = int(VIOLATION_THRESHOLD_SECONDS * fps)
    
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 4. Scale Zones to Video Resolution
    sx = frame_w / src_w
    sy = frame_h / src_h
    scaled_zones = scale_zones(original_zones, sx, sy)
    print(f"✅ Zones scaled to video resolution ({frame_w}x{frame_h}).")

    # 5. Initialize Tracking and Violation State
    # {track_id: frame_count} - Counts how many consecutive frames a vehicle is in a zone
    idle_timers = defaultdict(int)
    # {track_id: (zone_name, timestamp)} - Stores confirmed violations to avoid re-logging
    violation_history = {}

    print(f"\n🚀 Starting real-time violation detection on {DEVICE.upper()}... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended.")
            break

        # 6. Run Model Tracking on the specified device
        results = model.track(frame, persist=True, verbose=False, device=DEVICE, 
                              conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)[0]
        
        # Process tracked objects
        if results.boxes.id is not None:
            # Move results to CPU for numpy/cv2 operations
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()
            clss = results.boxes.cls.cpu().tolist()
            confs = results.boxes.conf.cpu().tolist()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, clss, confs):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls_id)]
                
                is_in_zone, zone_name = box_center_in_zone((x1, y1, x2, y2), scaled_zones)

                # Default color is green for non-violating vehicles
                color = (0, 255, 0) 
                
                # Check for violation condition (vehicle in a zone)
                if is_in_zone and label in ['motorcycle', 'motorbike', "car", "truck", "bus"]:
                    idle_timers[track_id] += 1 # Increment idle timer
                    
                    # If timer exceeds threshold, it's a violation
                    if idle_timers[track_id] >= violation_threshold_frames:
                        color = (0, 0, 255)  # Red for confirmed violation
                        # Log violation and save snapshot only once
                        if track_id not in violation_history:
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            
                            # --- Create and Save Snapshot ---
                            # Create a copy of the frame to draw violation-specific info
                            snapshot_frame = frame.copy()
                            
                            # Highlight the violated zone in red on the snapshot
                            violated_zone_poly = scaled_zones[zone_name]
                            cv2.polylines(snapshot_frame, [violated_zone_poly], isClosed=True, color=(0, 0, 255), thickness=2)
                            
                            # Draw the bounding box and ID for ONLY the violating vehicle on the snapshot
                            cv2.rectangle(snapshot_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            text = f"ID: {int(track_id)} ({conf:.2f})"
                            cv2.putText(snapshot_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            # Add the zone name text to the snapshot
                            # Calculate position for the text
                            M = cv2.moments(violated_zone_poly)
                            if M["m00"] != 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                                cv2.putText(snapshot_frame, f"Zone: {zone_name}", (cX - 50, cY), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                            snapshot_filename = os.path.join(SNAPSHOT_DIR, f"violation_{timestamp}_id{track_id}.jpg")
                            cv2.imwrite(snapshot_filename, snapshot_frame)
                            
                            # Save Log
                            log_filename = os.path.join(LOG_DIR, f"violation_{timestamp}_id{track_id}.json")
                            log_data = {
                                "track_id": track_id,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "zone_name": zone_name,
                                "confidence": f"{conf:.2f}",
                                "bounding_box": [x1, y1, x2, y2],
                                "snapshot_file": snapshot_filename
                            }
                            with open(log_filename, 'w') as f:
                                json.dump(log_data, f, indent=4)

                            violation_history[track_id] = (zone_name, timestamp)
                            print(f"🔴 VIOLATION: Vehicle ID {int(track_id)} in '{zone_name}'. Log and snapshot saved.")
                    else:
                        color = (0, 255, 255) # Yellow for potential violation (timer running)
                else:
                    # Reset timer if vehicle leaves the zone
                    idle_timers[track_id] = 0

                # Draw bounding box and ID on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"ID: {int(track_id)} ({conf:.2f})"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw the defined zones on the frame
        for name, poly in scaled_zones.items():
            cv2.polylines(frame, [poly], isClosed=True, color=(255, 255, 0), thickness=2)

        # 7. Display Live View
        cv2.imshow('Real-time Parking Violation Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # 8. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Session Summary ---")
    if violation_history:
        print(f"Total violations detected: {len(violation_history)}")
        for track_id, (zone, ts) in violation_history.items():
            print(f"  - Vehicle ID {int(track_id)} in '{zone}' at {ts}")
    else:
        print("No violations were recorded.")
    print("-----------------------")


if __name__ == '__main__':
    run_violation_detection()
