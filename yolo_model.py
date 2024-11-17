import torch

def load_model(model_name='yolov5s'):
    """
    Load the YOLOv5 model using torch.hub.
    Args:
        model_name (str): Name of the YOLO model (e.g., yolov5s, yolov5m).
    Returns:
        model: Loaded YOLOv5 model.
    """
    model = torch.hub.load('ultralytics/yolov5', 'custom', model_name, trust_repo=True)
    return model

def detect_objects(model, frame):
    """
    Perform object detection on a single frame using YOLOv5.
    Args:
        model: YOLOv5 model.
        frame: Image or video frame (BGR format).
    Returns:
        results: YOLOv5 detection results.
    """
    # Convert BGR frame to RGB (YOLOv5 expects RGB)
    frame_rgb = frame[:, :, ::-1]
    
    # Perform inference
    results = model(frame_rgb)
    return results