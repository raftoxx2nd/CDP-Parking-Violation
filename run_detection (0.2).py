import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import json
import time
import os
from collections import defaultdict

# --- Utility Functions ---

def load_zones(filepath='parking_zones.json'):
    """Loads zone data from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Zone file not found: '{filepath}'. Please run 'define_roi.py' first.")
    with open(filepath, 'r') as f:
        data = json.load(f)
    # Convert zone lists back to numpy arrays
    zones = {name: np.array(poly, dtype=np.int32) for name, poly in data['zones'].items()}
    return zones, data['source_image_width'], data['source_image_height']

def scale_zones(zones, sx, sy):
    """Scales polygon coordinates by given factors."""
    scaled = {}
    for name, poly in zones.items():
        scaled[name] = np.array([[int(x * sx), int(y * sy)] for [x, y] in poly], dtype=np.int32)
    return scaled

def box_overlaps_any_zone(box, zones):
    """Checks if a bounding box overlaps with any of the defined zones."""
    x1, y1, x2, y2 = box
    # Check the center point of the box for a quick and efficient test
    center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
    for name, poly in zones.items():
        if cv2.pointPolygonTest(poly, center_point, False) >= 0:
            return True, name
    return False, None

# --- Core Processing ---

def process_image(model, image_path, zones_config):
    
    """Processes a single image for parking violations."""
    # Load the image to be processed
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    target_h, target_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Unpack zone config and scale zones to match the current image's resolution
    original_zones, src_w, src_h = zones_config
    sx = target_w / src_w
    sy = target_h / src_h
    scaled_zones = scale_zones(original_zones, sx, sy)
    
    print(f"Running detection on '{os.path.basename(image_path)}' ({target_w}x{target_h})...")
    results = model(img_rgb, verbose=False)[0]

    # --- Visualization and Violation Logging ---
    output_image = img_rgb.copy()
    violations = []
    
    # Draw zones
    for name, poly in scaled_zones.items():
        cv2.polylines(output_image, [poly], isClosed=True, color=(255, 255, 0), thickness=2) # Yellow for zones
        cv2.putText(output_image, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
        cv2.putText(output_image, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    # Process detections
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        conf = box.conf[0]
        
        is_violation, zone_name = box_overlaps_any_zone((x1, y1, x2, y2), scaled_zones)
        
        # Define violation condition: motorcycle inside a zone
        if is_violation and label in ['motorcycle', 'motorbike']:
            color = (255, 0, 0) # Red for violation
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            text = f'{label} in {zone_name}'
            cv2.putText(output_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            violations.append({
                'class': label, 
                'confidence': float(conf),
                'bbox': [x1, y1, x2, y2], 
                'violation_zone': zone_name
            })
        else:
            # Optional: draw non-violating objects in green
            # color = (0, 255, 0) 
            # cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 1)
            pass

    # --- Output Results ---
    if violations:
        print(f"Found {len(violations)} violation(s).")
        # Save violation data
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        json_filename = f'violation_{timestamp}.json'
        with open(json_filename, 'w') as f:
            json.dump(violations, f, indent=4)
        print(f"  > Violation details saved to '{json_filename}'")
        
        # Save violation image
        img_filename = f'violation_{timestamp}.jpg'
        cv2.imwrite(img_filename, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        print(f"  > Violation image saved to '{img_filename}'")
    else:
        print("No violations detected.")

    # Display the final image
    plt.figure(figsize=(12, 9))
    plt.imshow(output_image)
    plt.title('Hasil Deteksi Pelanggaran Parkir')
    plt.axis('off')
    plt.show()

def process_video(model, video_source, zones_config):
    """Processes a video stream for parking violations."""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {video_source}")

    # Unpack zone config and scale zones once using video frame dimensions
    original_zones, src_w, src_h = zones_config
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return
        
    target_h, target_w = frame.shape[:2]
    sx = target_w / src_w
    sy = target_h / src_h
    scaled_zones = scale_zones(original_zones, sx, sy)
    
    # --- Real-time Tracking State ---
    # Store frame count for each tracked object inside a zone
    # {track_id: frame_count}
    violation_tracker = defaultdict(int)
    # Store confirmed violations to avoid re-logging
    # {track_id: zone_name}
    confirmed_violations = {}
    VIOLATION_THRESHOLD_FRAMES = 30 # e.g., 1 second at 30fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Use model.track() for real-time object tracking
        results = model.track(frame, persist=True, verbose=False)[0]
        
        output_image = frame.copy()

        # Draw zones on the frame
        for name, poly in scaled_zones.items():
            cv2.polylines(output_image, [poly], isClosed=True, color=(255, 255, 0), thickness=2)

        # Process tracked objects
        if results.boxes.id is not None:
            for box, track_id in zip(results.boxes, results.boxes.id.int()):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = results.names[int(box.cls[0])]
                
                is_in_zone, zone_name = box_overlaps_any_zone((x1, y1, x2, y2), scaled_zones)

                # Default color
                color = (0, 255, 0) # Green

                if is_in_zone and label in ['motorcycle', 'motorbike']:
                    violation_tracker[track_id.item()] += 1
                    
                    if violation_tracker[track_id.item()] >= VIOLATION_THRESHOLD_FRAMES:
                        color = (0, 0, 255) # Red for confirmed violation
                        # Log violation only once
                        if track_id.item() not in confirmed_violations:
                            confirmed_violations[track_id.item()] = zone_name
                            print(f"CONFIRMED VIOLATION: Track ID {track_id.item()} in {zone_name}")
                            # (Optional) Save violation data to JSON here
                    else:
                        color = (0, 255, 255) # Yellow for potential violation
                
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(output_image, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Real-time Violation Detection', output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Processing finished.")


def main():
    """Main function to initialize model and run detection."""
    try:
        # 1. Load pre-defined zones
        zones_config = load_zones()
        print("✅ Parking zones loaded successfully.")

        # 2. Initialize YOLO model
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 model initialized.")

        # 3. Get video source to process
        video_source = input("Enter the path to the video file or camera ID (e.g., 0): ")
        
        # 4. Process the video
        process_video(model, video_source, zones_config)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == '__main__':
    main()