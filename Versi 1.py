# CDP300 - Simulasi Deteksi Pelanggaran Parkir (Versi 2)
# Fitur baru:
# - ROI dinamis: interaktif (matplotlib fallback) + OpenCV (opsional, lokal)
# - Multiple polygon zones (slot1, slot2, ..., zona_mobil_besar)
# - Scaling otomatis ROI saat gambar di-resize (resolusi input: manual)
# - Simpan zones otomatis ke dict "zones" dengan nama incremental
# - Integrasi dengan pipeline deteksi YOLOv8 (modular)

# Catatan:
# - Di Google Colab, OpenCV GUI windows (cv2.imshow) sering tidak didukung.
#   Oleh karena itu notebook ini menyediakan dua mode interaktif:
#   1) MATPLOTLIB mode (direkomendasikan di Colab): klik pada gambar untuk menambah titik.
#      Selesai satu polygon tekan Enter/Return, lalu lanjut buat polygon berikutnya.
#   2) OPENCV mode (lokal): klik kiri untuk titik, klik kanan untuk menutup polygon.
# - User akan menentukan target resize resolution secara manual (contoh: new_w=640, new_h=360).

# ----------------------------
# 0. Instalasi & Import
# ----------------------------
## Remove Colab-specific pip install line. Install dependencies via terminal:
# pip install ultralytics opencv-python matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
#from google.colab import files
import json
import time

print('✅ Dependencies loaded')

# ----------------------------
# 1. Utilities: I/O & Resize
# ----------------------------

def upload_image():
    """Prompt user for local image file path; returns the path."""
    path = input("Enter image file path: ")
    return path


def load_image_rgb(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f'File not found: {path}')
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def resize_image_keep_aspect(img_rgb, target_w, target_h):
    """Resize image to target resolution (fixed). Returns resized image and scale factors (sx, sy)."""
    orig_h, orig_w = img_rgb.shape[:2]
    resized = cv2.resize(img_rgb, (target_w, target_h))
    sx = target_w / orig_w
    sy = target_h / orig_h
    return resized, sx, sy

# ----------------------------
# 2. ROI Definition (Interactive)
# ----------------------------

def define_polygons_matplotlib(img_rgb):
    """Define multiple polygons interactively using matplotlib.
    Usage:
      - Click vertices for polygon 1.
      - Press Enter/Return to finish polygon 1.
      - Repeat to draw more polygons.
      - Close the figure window when done drawing all polygons.
    Returns: dict zones with keys slot1, slot2, ... each value is np.array of points [[x,y],...]
    """
    zones = {}
    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(img_rgb)
    ax.set_title('Click vertices. Press Enter to finish a polygon. Close window when done.')

    plt.axis('off')
    plt.tight_layout()

    slot_idx = 1
    print('Instructions: click to add polygon vertices. Press Enter to finish current polygon. Close the window when finished adding polygons.')

    while True:
        pts = plt.ginput(n=-1, timeout=0)
        # plt.ginput returns list of (x,y). If user closes window, it may raise
        if not pts:
            # user may have closed or pressed enter without points
            break
        poly = np.array([[int(x), int(y)] for (x,y) in pts])
        if poly.shape[0] < 3:
            print('Polygon must have at least 3 points. Ignored.')
            continue
        key = f'slot{slot_idx}'
        zones[key] = poly
        slot_idx += 1
        # draw polygon on axis for preview
        ax.plot(np.append(poly[:,0], poly[0,0]), np.append(poly[:,1], poly[0,1]), '-r')
        fig.canvas.draw()
        print(f'Polygon {key} recorded ({len(poly)} points). Continue drawing or close window when finished.')
    plt.close(fig)
    print(f'Done. {len(zones)} polygon(s) recorded.')
    return zones


def define_polygon_opencv_local(img_bgr):
    """Define multiple polygons using OpenCV window (works on local machine with GUI). 
    Left click to add points for current polygon. Right click to close current polygon and store it.
    Press 'q' to finish all polygons.
    Returns dict zones.
    """
    zones = {}
    points = []
    slot_idx = 1
    clone = img_bgr.copy()

    def on_mouse(event, x, y, flags, param):
        nonlocal points, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x,y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # close polygon
            if len(points) >= 3:
                poly = np.array(points)
                key = f'slot{slot_idx}'
                zones[key] = poly
                print(f'Polygon {key} stored with {len(points)} points')
                points = []
                # redraw
                cv2.polylines(clone, [poly], True, (0,0,255), 2)

    cv2.namedWindow('Define ROI - OpenCV (local only)')
    cv2.setMouseCallback('Define ROI - OpenCV (local only)', on_mouse)

    while True:
        temp = clone.copy()
        if len(points) > 0:
            for p in range(len(points)-1):
                cv2.line(temp, points[p], points[p+1], (0,255,0), 2)
            cv2.circle(temp, points[-1], 3, (0,255,0), -1)
        cv2.imshow('Define ROI - OpenCV (local only)', temp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    return zones

# ----------------------------
# 3. Scaling Zones
# ----------------------------

def scale_zones(zones, sx, sy):
    """Scale each polygon in zones by scale factors sx, sy.
    zones: dict(name -> np.array of points [[x,y],...])
    returns new dict with scaled polygons (int coords)
    """
    scaled = {}
    for name, poly in zones.items():
        scaled[name] = np.array([[int(x*sx), int(y*sy)] for (x,y) in poly])
    return scaled

# ----------------------------
# 4. Detection helper (YOLO)
# ----------------------------

def init_yolo(model_name='yolov8n.pt', conf=0.35):
    model = YOLO(model_name)
    model.conf = conf
    return model


def detect_yolo_on_image(model, img_path):
    res = model(img_path)  # use path or numpy image
    return res[0]

# reuse earlier box overlap logic but generalized to multi-zones

def box_overlaps_any_zone(box, zones):
    for name, poly in zones.items():
        if box_overlaps_polygon(box, poly):
            return True, name
    return False, None


def box_overlaps_polygon(box, polygon):
    x1, y1, x2, y2 = box
    # test corner points + center
    points = [(x1,y1),(x2,y1),(x2,y2),(x1,y2),((x1+x2)//2,(y1+y2)//2)]
    for pt in points:
        if cv2.pointPolygonTest(polygon, pt, False) >= 0:
            return True
    return False

# ----------------------------
# 5. Visualization & Logging
# ----------------------------

def visualize_with_zones(img_rgb, detection, zones, title=None):
    out = img_rgb.copy()
    # draw zones
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]
    i=0
    for name, poly in zones.items():
        color = colors[i % len(colors)]
        cv2.polylines(out, [poly], True, color, 2)
        # put label near first vertex
        cv2.putText(out, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        i+=1

    violations = []
    # draw detections
    for box in detection.boxes:
        cls_id = int(box.cls[0].item())
        label = detection.names[cls_id]
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        overlap, zone_name = box_overlaps_any_zone((x1,y1,x2,y2), zones)
        if overlap and label in ['motorcycle','motorbike']:
            # violation
            cv2.rectangle(out, (x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(out, f'{label} | {zone_name}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0),2)
            violations.append({'class':label,'bbox':(x1,y1,x2,y2),'zone':zone_name})
        else:
            cv2.rectangle(out, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(out, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

    plt.figure(figsize=(12,8))
    plt.imshow(out)
    plt.axis('off')
    if title is None:
        title = 'Hasil Deteksi dengan Zona'
    plt.title(title)
    plt.show()
    return violations

# ----------------------------
# 6. Main flow: combine everything
# ----------------------------

def interactive_and_run(model, target_w, target_h, mode='matplotlib'):
    # 1) upload
    path = upload_image()
    img_rgb = load_image_rgb(path)
    orig_h, orig_w = img_rgb.shape[:2]
    print(f'Original size: {orig_w}x{orig_h}')

    # 2) let user define polygons (interactive or local-opencv)
    if mode == 'matplotlib':
        print('Using matplotlib interactive ROI. For Colab use this mode.')
        zones = define_polygons_matplotlib(img_rgb)
    else:
        print('Using OpenCV local ROI (requires GUI).')
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        zones = define_polygon_opencv_local(img_bgr)

    if len(zones) == 0:
        print('No zones defined. Exiting.')
        return

    # save original zones
    with open('zones_orig.json','w') as f:
        json.dump({k: v.tolist() for k,v in zones.items()}, f)
    print('Saved original zones to zones_orig.json')

    # 3) resize image to fixed target and scale zones
    resized, sx, sy = resize_image_keep_aspect(img_rgb, target_w, target_h)
    zones_scaled = scale_zones(zones, sx, sy)
    print(f'Resized image to {target_w}x{target_h} (sx={sx:.3f}, sy={sy:.3f}); zones scaled accordingly.')

    # 4) run detection on original path OR on resized (we pass path for inference; for speed you may save resized to disk)
    # Option A: Run detection on resized image (must write temp file)
    tmp_path = 'tmp_resized.jpg'
    cv2.imwrite(tmp_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    detection = detect_yolo_on_image(model, tmp_path)

    # 5) visualize and log violations
    violations = visualize_with_zones(resized, detection, zones_scaled)
    if violations:
        print('Violations detected:')
        for v in violations:
            print(v)
    else:
        print('No violations.')

    # save zones scaled for later reuse
    with open('zones_scaled.json','w') as f:
        json.dump({k: v.tolist() for k,v in zones_scaled.items()}, f)
    print('Saved scaled zones to zones_scaled.json')

# ----------------------------
# 7. Example: initialize model and run
# ----------------------------

# initialize light model for prototype
model = init_yolo('yolov8n.pt', conf=0.35)

# Example usage (user defines target resolution manually):
interactive_and_run(model, target_w=640, target_h=360, mode='matplotlib')

# If running locally with GUI support: mode='opencv'

# ----------------------------
# 8. Notes & Next Steps
# ----------------------------
# - This notebook stores zones_orig.json (original coordinate system) and zones_scaled.json (scaled for target resolution).
# - You can later load zones_scaled.json and run batch detection on many images without redefining ROIs.
# - For production, consider creating a small web UI (Flask + JS) to allow drawing polygons on browser and saving them to server.
# - For video stream, polygons (scaled) can be applied per frame; detection logic is same but you need tracking to evaluate "stationary time" for violation.
