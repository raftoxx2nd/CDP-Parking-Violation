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
