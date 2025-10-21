import cv2

def get_video_capture(source):
    """
    Initializes and returns a cv2.VideoCapture object.

    Args:
        source (str or int): The video source (e.g., '0' for webcam, or path to video file).

    Returns:
        cv2.VideoCapture: The video capture object.
    
    Raises:
        IOError: If the video source cannot be opened.
    """
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {source}")
    return cap

def get_frame_from_source(cap):
    """
    Captures a single valid frame from a video capture object.

    Args:
        cap (cv2.VideoCapture): The video capture object.

    Returns:
        numpy.ndarray: The captured frame.
    
    Raises:
        IOError: If a frame cannot be captured.
    """
    # Read frames until a valid one is found
    while True:
        ret, frame = cap.read()
        if ret:
            return frame
        print("Trying to capture a frame...")
        cv2.waitKey(100) # Wait a bit before retrying
    
    raise IOError("Could not capture a frame from the video source.")
