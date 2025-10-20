import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def upload_image():
    """Prompt user for local image file path; returns the path."""
    path = input("Enter the path to the image for defining zones: ")
    return path

def load_image_rgb(path):
    """Loads an image from a path and converts it to RGB format."""
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f'File not found: {path}')
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def define_polygons_matplotlib(img_rgb):
    """
    Define multiple polygons interactively using matplotlib.
    Returns a dictionary of zones.
    """
    zones = {}
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(img_rgb)
    ax.set_title('Click to define polygon vertices. Press Enter to finish a polygon. Close window when done.')
    plt.axis('off')

    slot_idx = 1
    print("\n--- ROI Definition Instructions ---")
    print("1. Click on the image to add points for a polygon.")
    print("2. Press 'Enter' to complete the current polygon.")
    print("3. Repeat to add more polygons.")
    print("4. Close the plot window to finish and save.")
    print("------------------------------------")

    while plt.get_fignums():
        pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
        if not pts:
            break
        
        poly = np.array([[int(x), int(y)] for (x, y) in pts], dtype=np.int32)
        if poly.shape[0] < 3:
            print("! Warning: A polygon must have at least 3 points. This one was ignored.")
            continue
            
        key = f'slot{slot_idx}'
        zones[key] = poly
        slot_idx += 1
        
        # Draw the completed polygon on the plot for user feedback
        ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]), '-r', lw=2)
        ax.text(poly[0, 0], poly[0, 1] - 10, key, color='white', backgroundcolor='red', fontsize=8)
        fig.canvas.draw()
        print(f"  > Polygon '{key}' saved with {len(poly)} points. You can draw another or close the window.")

    plt.close(fig)
    return zones

def main():
    """Main function to run the ROI definition process."""
    try:
        image_path = upload_image()
        img_rgb = load_image_rgb(image_path)
        orig_h, orig_w = img_rgb.shape[:2]

        zones = define_polygons_matplotlib(img_rgb)

        if not zones:
            print("\nNo zones were defined. Exiting.")
            return

        # Structure data for saving
        zone_data = {
            "source_image_width": orig_w,
            "source_image_height": orig_h,
            "zones": {name: poly.tolist() for name, poly in zones.items()}
        }

        # Save to file
        output_filename = 'parking_zones.json'
        with open(output_filename, 'w') as f:
            json.dump(zone_data, f, indent=4)
        
        print(f"\n✅ Success! {len(zones)} zone(s) and image dimensions saved to '{output_filename}'.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()