import cv2
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg') #Use if have isssue with default backend
import matplotlib.pyplot as plt
import json
import os
import sys

# Add project root to the Python path to allow sibling imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from capture.stream_handler import get_video_capture, get_frame_from_source

# --- Configuration ---
ZONES_OUTPUT_FILE = 'config/zones.json'
FIXED_VIDEO_SOURCE = "https://10.26.49.165:8080/video"

def define_polygons_interactive(frame_bgr):
    """
    Define multiple polygons interactively on a given frame using matplotlib's event handling.
    Returns a dictionary of zones.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(frame_rgb)
    ax.set_title('Click to define vertices. Press "Enter" to finish a polygon. Close window when done.')
    plt.axis('off')

    print("\n--- ROI Definition Instructions ---")
    print("1. Click on the image to add points for a polygon.")
    print("2. Press 'Enter' to complete the current polygon and start a new one.")
    print("3. Close the plot window to finish and save all defined zones.")
    print("------------------------------------")

    builder = PolygonBuilder(ax)
    plt.show()  # This is a blocking call

    return builder.zones

class PolygonBuilder:
    def __init__(self, ax):
        self.ax = ax
        self.fig = ax.figure
        self.zones = {}
        self.current_poly_pts = []
        self.zone_idx = 1
        
        self.line = self.ax.plot([], [], 'r-o', lw=2)[0]
        
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        # Add point
        self.current_poly_pts.append((event.xdata, event.ydata))
        
        # Update visualization
        x, y = zip(*self.current_poly_pts)
        self.line.set_data(x, y)
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'enter':
            if len(self.current_poly_pts) < 3:
                print("! Warning: A polygon must have at least 3 points. This one was ignored.")
                self.reset_current_poly()
                return

            key = f'zone_{self.zone_idx}'
            poly = np.array(self.current_poly_pts, dtype=np.int32)
            self.zones[key] = poly
            self.zone_idx += 1
            
            # Draw the completed polygon for feedback
            self.ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]), '-r', lw=2)
            self.ax.text(poly[0, 0], poly[0, 1] - 10, key, color='white', backgroundcolor='red', fontsize=9)
            
            print(f"  > Polygon '{key}' saved with {len(poly)} points. You can draw another or close the window.")
            
            self.reset_current_poly()

    def reset_current_poly(self):
        self.current_poly_pts = []
        self.line.set_data([], [])
        self.fig.canvas.draw()

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
        zones = define_polygons_interactive(frame)

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
