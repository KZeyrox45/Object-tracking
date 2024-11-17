import cv2
import os
from yolo_model import load_model, detect_objects

def resize_frame(frame, max_width=800, max_height=600):
    """
    Resize the video frame to fit within the max dimensions while maintaining aspect ratio.
    Args:
        frame: Input video frame.
        max_width: Maximum width for the resized frame.
        max_height: Maximum height for the resized frame.
    Returns:
        Resized frame, scaling factors for width and height.
    """
    height, width = frame.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame, scaling_factor

def annotate_frame(frame, results, model, confidence_threshold, scaling_factor=1.0):
    """
    Annotate the frame with detection results, adjusting for scaling if necessary.
    Args:
        frame: Frame to annotate.
        results: Detection results from the model.
        model: YOLOv5 model.
        confidence_threshold: Minimum confidence to consider a detection.
        scaling_factor: Scaling factor for the frame (used for resized frames).
    Returns:
        Annotated frame.
    """
    for result in results.xyxy[0]:  # Loop through detected objects
        x1, y1, x2, y2, conf, cls = result[:6]
        if conf < confidence_threshold:  # Skip low-confidence detections
            continue

        # Adjust coordinates for scaled frames
        x1, y1, x2, y2 = (int(x1 * scaling_factor), int(y1 * scaling_factor),
                          int(x2 * scaling_factor), int(y2 * scaling_factor))

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

        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (0, 255, 0), -1)  # Background
        cv2.putText(frame, label, (label_x1 + 2, label_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame

def process_video(video_path, model, confidence_threshold=0.3):
    """
    Detect objects in a video frame by frame and export the processed video.
    Args:
        video_path (str): Path to the input video.
        model: YOLOv5 model.
        confidence_threshold (float): Minimum confidence to consider a detection.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}. Ensure the path is correct and the file exists.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) // 2
    output_dir = "./output/video/"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    output_path = os.path.join(output_dir, os.path.basename(video_path).replace(".mp4", "_processed.mp4"))

    # Use MP4V codec for saving the video
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Create a resizable full-screen window
    window_name = "Detected Objects"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a resizable window
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect objects
            results = detect_objects(model, frame)

            # Annotate the original frame for export
            annotated_frame = annotate_frame(frame.copy(), results, model, confidence_threshold)

            # Write the processed frame to the output video
            out.write(annotated_frame)

            # Resize frame for demonstration
            resized_frame, scaling_factor = resize_frame(frame)

            # Annotate the resized frame for clear demonstration
            annotated_resized_frame = annotate_frame(resized_frame, results, model, confidence_threshold, scaling_factor)

            # Display the annotated resized frame
            cv2.imshow(window_name, annotated_resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
    finally:
        # Ensure all resources are released
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Output video saved to {output_path}")
