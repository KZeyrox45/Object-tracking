"""Image processing module for object detection."""
import cv2
import os
import logging
from typing import Optional, Tuple

from yolo_model import detect_objects, get_detections_count, format_detections

logger = logging.getLogger(__name__)


def resize_image(
    image: cv2.Mat, max_width: int = 800, max_height: int = 600
) -> Tuple[cv2.Mat, float]:
    """
    Resize the image to fit within the max dimensions while maintaining aspect ratio.

    Args:
        image: Input image.
        max_width: Maximum width for the resized image.
        max_height: Maximum height for the resized image.

    Returns:
        Tuple of (resized image, scaling factor).
    """
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, scaling_factor


def annotate_frame(
    frame: cv2.Mat,
    results,
    model,
    confidence_threshold: float,
    scaling_factor: float = 1.0,
) -> cv2.Mat:
    """
    Annotate the frame with detection results.

    Args:
        frame: Frame to annotate.
        results: Detection results from the model.
        model: YOLOv5 model.
        confidence_threshold: Minimum confidence to consider a detection.
        scaling_factor: Scaling factor for the frame.

    Returns:
        Annotated frame.
    """
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result[:6]
        if conf < confidence_threshold:
            continue

        # Adjust coordinates for scaled frames
        x1, y1, x2, y2 = (
            int(x1 * scaling_factor),
            int(y1 * scaling_factor),
            int(x2 * scaling_factor),
            int(y2 * scaling_factor),
        )

        # Create the label
        label = f"{model.names[int(cls)]}: {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label with background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_x1 = x1
        label_y1 = y1 - 10 if y1 - 10 > 10 else y1 + 10
        label_x2 = label_x1 + label_size[0] + 5
        label_y2 = label_y1 + label_size[1] + 5

        cv2.rectangle(
            frame, (label_x1, label_y1), (label_x2, label_y2), (0, 255, 0), -1
        )
        cv2.putText(
            frame,
            label,
            (label_x1 + 2, label_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    return frame


def process_image(
    image_path: str,
    model,
    confidence_threshold: float = 0.2,
    max_width: int = 800,
    max_height: int = 600,
    output_dir: Optional[str] = None,
    display: bool = True,
    window_name: str = "Detected Objects",
    fullscreen: bool = False,
) -> Optional[str]:
    """
    Detect objects in an image and save/display the annotated result.

    Args:
        image_path: Path to the input image.
        model: YOLOv5 model.
        confidence_threshold: Minimum confidence to consider a detection.
        max_width: Maximum width for resized display.
        max_height: Maximum height for resized display.
        output_dir: Directory to save output (uses default if None).
        display: Whether to display the result.
        window_name: Name of the display window.
        fullscreen: Whether to display in fullscreen.

    Returns:
        Path to saved output file, or None if failed.
    """
    logger.info(f"Processing image: {image_path}")

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return None

    # Detect objects
    results = detect_objects(model, image)
    detection_count = get_detections_count(results)
    logger.info(f"Detected {detection_count} objects in {image_path}")

    # Resize the image for display
    resized_image, scaling_factor = resize_image(image, max_width, max_height)

    # Annotate the resized image with results
    annotated_image = annotate_frame(
        resized_image, results, model, confidence_threshold, scaling_factor
    )

    # Save the output image
    if output_dir is None:
        output_dir = "./output/image/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, annotated_image)
    logger.info(f"Output saved to {output_path}")

    # Display the result if requested
    if display:
        cv2.imshow(window_name, annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output_path


def process_image_batch(
    image_paths: list,
    model,
    confidence_threshold: float = 0.2,
    max_width: int = 800,
    max_height: int = 600,
    output_dir: Optional[str] = None,
    display: bool = False,
    window_name: str = "Detected Objects",
    fullscreen: bool = False,
) -> dict:
    """
    Process multiple images in batch.

    Args:
        image_paths: List of image paths to process.
        model: YOLOv5 model.
        confidence_threshold: Minimum confidence threshold.
        max_width: Maximum width for resized display.
        max_height: Maximum height for resized display.
        output_dir: Directory to save outputs.
        display: Whether to display results (not recommended for batch).
        window_name: Name of the display window.
        fullscreen: Whether to display in fullscreen.

    Returns:
        Dictionary with processing statistics.
    """
    logger.info(f"Starting batch processing of {len(image_paths)} images")

    results = {
        "total": len(image_paths),
        "success": 0,
        "failed": 0,
        "total_detections": 0,
        "outputs": [],
        "errors": [],
    }

    for i, image_path in enumerate(image_paths, 1):
        logger.info(f"Processing image {i}/{len(image_paths)}: {image_path}")

        output_path = process_image(
            image_path=image_path,
            model=model,
            confidence_threshold=confidence_threshold,
            max_width=max_width,
            max_height=max_height,
            output_dir=output_dir,
            display=display,
            window_name=window_name,
            fullscreen=fullscreen,
        )

        if output_path:
            results["success"] += 1
            results["outputs"].append(output_path)

            # Count detections
            image = cv2.imread(image_path)
            if image is not None:
                det_results = detect_objects(model, image)
                results["total_detections"] += get_detections_count(det_results)
        else:
            results["failed"] += 1
            results["errors"].append(image_path)

    logger.info(
        f"Batch processing complete: {results['success']}/{results['total']} successful, "
        f"{results['total_detections']} total detections"
    )

    return results
