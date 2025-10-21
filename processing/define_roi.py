import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

# Add project root to the Python path to allow sibling imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from capture.stream_handler import get_video_capture, get_frame_from_source

# --- Configuration ---
ZONES_OUTPUT_FILE = 'config/zones.json'
FIXED_VIDEO_SOURCE = "https://192.168.1.15:8080/video"

def define_polygons_matplotlib(frame_bgr):
    """
    Define multiple polygons interactively on a given frame using matplotlib.
    Returns a dictionary of zones.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    zones = {}
    
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(frame_rgb)
    ax.set_title('Click to define polygon vertices. Press Enter to finish a polygon. Close window when done.')
    plt.axis('off')

    slot_idx = 1
    print("\n--- ROI Definition Instructions ---")
    print("1. Click on the image to add points for a polygon.")
    print("2. Press 'Enter' to complete the current polygon and start a new one.")
    print("3. Close the plot window to finish and save all defined zones.")
    print("------------------------------------")

    while plt.get_fignums():
        # ginput waits for user clicks. n=-1 means infinite points until Enter.
        pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
        if not pts: # If the user closes the window, ginput returns an empty list
            break
        
        poly = np.array([[int(x), int(y)] for (x, y) in pts], dtype=np.int32)
        
        if poly.shape[0] < 3:
            print("! Warning: A polygon must have at least 3 points. This one was ignored.")
            continue
            
        key = f'zone_{slot_idx}'
        zones[key] = poly
        slot_idx += 1
        
        # Draw the completed polygon on the plot for immediate user feedback
        ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]), '-r', lw=2)
        ax.text(poly[0, 0], poly[0, 1] - 10, key, color='white', backgroundcolor='red', fontsize=9)
        fig.canvas.draw()
        print(f"  > Polygon '{key}' saved with {len(poly)} points. You can draw another or close the window.")

    plt.close(fig)
    return zones

def main():
    """Main function to run the ROI definition process."""
    try:
        # 1. Get a frame from the video source
        video_source = FIXED_VIDEO_SOURCE
        if video_source is None:
            video_source = input("Enter video source (e.g., '0' for webcam, or path to video file): ")
        
        try:
            print(f"Connecting to video source: {video_source}")
            cap = get_video_capture(video_source)
        except IOError as e:
            print(f"❌ Error: {e}")
            return
        frame = get_frame_from_source(cap)
        cap.release() # Release after capturing the frame
        
        orig_h, orig_w = frame.shape[:2]
        print(f"✅ Frame captured ({orig_w}x{orig_h}). Please define zones in the window that opens.")

        # 2. Let the user define zones on the captured frame
        zones = define_polygons_matplotlib(frame)

        if not zones:
            print("\nNo zones were defined. Exiting without saving.")
            return

        # 3. Structure data for saving
        zone_data = {
            "source_image_width": orig_w,
            "source_image_height": orig_h,
            "zones": {name: poly.tolist() for name, poly in zones.items()}
        }

        # 4. Save to the configured JSON file
        # Ensure the config directory exists
        os.makedirs(os.path.dirname(ZONES_OUTPUT_FILE), exist_ok=True)
        with open(ZONES_OUTPUT_FILE, 'w') as f:
            json.dump(zone_data, f, indent=4)
        
        print(f"\n✅ Success! {len(zones)} zone(s) and frame dimensions saved to '{ZONES_OUTPUT_FILE}'.")

    except (IOError, FileNotFoundError) as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
