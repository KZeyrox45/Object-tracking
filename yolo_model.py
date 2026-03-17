"""YOLO model loading and inference."""
import logging
import torch
import cv2

logger = logging.getLogger(__name__)


def load_model(model_path: str = "yolov5s", model_source: str = "ultralytics/yolov5", device: str = None):
    """
    Load the YOLOv5 model using torch.hub.

    Args:
        model_path: Path to model weights file or model name.
        model_source: Repository source for torch.hub.
        device: Device to load the model on ('cpu', 'cuda', etc.). If None, uses best available.

    Returns:
        model: Loaded YOLOv5 model.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Loading YOLO model from {model_source} with path {model_path} on {device}")
    try:
        model = torch.hub.load(model_source, "custom", path=model_path, trust_repo=True, device=device)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


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


def get_detections_count(results) -> int:
    """
    Get the number of detections from results.

    Args:
        results: YOLOv5 detection results.

    Returns:
        Number of detections.
    """
    if results.xyxy[0] is None or len(results.xyxy[0]) == 0:
        return 0
    return len(results.xyxy[0])


def format_detections(results, model, confidence_threshold: float = 0.3) -> list:
    """
    Format detections into a readable list.

    Args:
        results: YOLOv5 detection results.
        model: YOLOv5 model (for class names).
        confidence_threshold: Minimum confidence to include.

    Returns:
        List of formatted detection strings.
    """
    detections = []
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result[:6]
        if conf < confidence_threshold:
            continue
        label = f"{model.names[int(cls)]}: {conf:.2f}"
        detections.append(label)
    return detections
