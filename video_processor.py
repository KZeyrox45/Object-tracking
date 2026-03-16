"""Video processing module for object detection."""
import cv2
import os
import logging
from typing import Optional, List

from yolo_model import detect_objects, get_detections_count

logger = logging.getLogger(__name__)


def resize_frame(
    frame: cv2.Mat, max_width: int = 800, max_height: int = 600
) -> tuple:
    """
    Resize the video frame to fit within max dimensions.

    Args:
        frame: Input video frame.
        max_width: Maximum width for the resized frame.
        max_height: Maximum height for the resized frame.

    Returns:
        Tuple of (resized frame, scaling factor).
    """
    height, width = frame.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame, scaling_factor


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


def process_video(
    video_path: str,
    model,
    confidence_threshold: float = 0.3,
    max_width: int = 800,
    max_height: int = 600,
    output_dir: Optional[str] = None,
    fps_divisor: int = 2,
    codec: str = "MP4V",
    display: bool = True,
    window_name: str = "Detected Objects",
    fullscreen: bool = True,
) -> Optional[str]:
    """
    Detect objects in a video and save the annotated result.

    Args:
        video_path: Path to the input video.
        model: YOLOv5 model.
        confidence_threshold: Minimum confidence to consider a detection.
        max_width: Maximum width for resized display.
        max_height: Maximum height for resized display.
        output_dir: Directory to save output.
        fps_divisor: Divide FPS by this value for output.
        codec: Video codec for output.
        display: Whether to display during processing.
        window_name: Name of the display window.
        fullscreen: Whether to display in fullscreen.

    Returns:
        Path to saved output file, or None if failed.
    """
    logger.info(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) // fps_divisor
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_dir is None:
        output_dir = "./output/video/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, os.path.basename(video_path).replace(".mp4", "_processed.mp4")
    )

    # Use specified codec
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Setup display if requested
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    total_detections = 0
    processed_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect objects
            results = detect_objects(model, frame)
            total_detections += get_detections_count(results)

            # Annotate the original frame for export
            annotated_frame = annotate_frame(
                frame.copy(), results, model, confidence_threshold
            )

            # Write the processed frame
            out.write(annotated_frame)
            processed_frames += 1

            # Display progress
            if display:
                resized_frame, scaling_factor = resize_frame(frame, max_width, max_height)
                annotated_resized = annotate_frame(
                    resized_frame, results, model, confidence_threshold, scaling_factor
                )
                cv2.imshow(window_name, annotated_resized)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Processing interrupted by user")
                    break

            # Log progress every 10%
            if processed_frames % max(1, total_frames // 10) == 0:
                progress = (processed_frames / total_frames) * 100
                logger.info(f"Progress: {progress:.1f}% ({processed_frames}/{total_frames} frames)")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    logger.info(
        f"Video processing complete: {processed_frames} frames processed, "
        f"{total_detections} total detections. Output: {output_path}"
    )

    return output_path


def process_video_batch(
    video_paths: List[str],
    model,
    confidence_threshold: float = 0.3,
    max_width: int = 800,
    max_height: int = 600,
    output_dir: Optional[str] = None,
    fps_divisor: int = 2,
    codec: str = "MP4V",
    display: bool = False,
    window_name: str = "Detected Objects",
    fullscreen: bool = True,
) -> dict:
    """
    Process multiple videos in batch.

    Args:
        video_paths: List of video paths to process.
        model: YOLOv5 model.
        confidence_threshold: Minimum confidence threshold.
        max_width: Maximum width for resized display.
        max_height: Maximum height for resized display.
        output_dir: Directory to save outputs.
        fps_divisor: Divide FPS by this value.
        codec: Video codec for output.
        display: Whether to display during processing.
        window_name: Name of the display window.
        fullscreen: Whether to display in fullscreen.

    Returns:
        Dictionary with processing statistics.
    """
    logger.info(f"Starting batch processing of {len(video_paths)} videos")

    results = {
        "total": len(video_paths),
        "success": 0,
        "failed": 0,
        "total_detections": 0,
        "outputs": [],
        "errors": [],
    }

    for i, video_path in enumerate(video_paths, 1):
        logger.info(f"Processing video {i}/{len(video_paths)}: {video_path}")

        output_path = process_video(
            video_path=video_path,
            model=model,
            confidence_threshold=confidence_threshold,
            max_width=max_width,
            max_height=max_height,
            output_dir=output_dir,
            fps_divisor=fps_divisor,
            codec=codec,
            display=display,
            window_name=window_name,
            fullscreen=fullscreen,
        )

        if output_path:
            results["success"] += 1
            results["outputs"].append(output_path)
            # Note: total_detections is logged but not accumulated for videos
        else:
            results["failed"] += 1
            results["errors"].append(video_path)

    logger.info(
        f"Batch video processing complete: {results['success']}/{results['total']} successful"
    )

    return results
