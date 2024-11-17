import cv2
import os
from yolo_model import load_model, detect_objects

def resize_image(image, max_width=800, max_height=600):
    """
    Resize the image to fit within the max dimensions while maintaining aspect ratio.
    Args:
        image: Input image.
        max_width: Maximum width for the resized image.
        max_height: Maximum height for the resized image.
    Returns:
        Resized image, scaling factors for width and height.
    """
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, scaling_factor

def process_image(image_path, model, confidence_threshold=0.2):
    """
    Detect objects in an image.
    Args:
        image_path (str): Path to the input image.
        model: YOLOv5 model.
        confidence_threshold (float): Minimum confidence to consider a detection.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}.")
        return

    # Detect objects
    results = detect_objects(model, image)

    # Resize the image for display
    resized_image, scaling_factor = resize_image(image)

    # Annotate the resized image with results
    for result in results.xyxy[0]:  # Loop through detected objects
        x1, y1, x2, y2, conf, cls = result[:6]
        if conf < confidence_threshold:  # Skip low-confidence detections
            continue

        # Adjust bounding box coordinates to the resized image
        x1, y1, x2, y2 = int(x1 * scaling_factor), int(y1 * scaling_factor), int(x2 * scaling_factor), int(y2 * scaling_factor)
        label = f"{model.names[int(cls)]}: {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label with background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_x1 = x1
        label_y1 = y1 - 10 if y1 - 10 > 10 else y1 + 10  # Adjust label position
        label_x2 = label_x1 + label_size[0] + 5
        label_y2 = label_y1 + label_size[1] + 5

        cv2.rectangle(resized_image, (label_x1, label_y1), (label_x2, label_y2), (0, 255, 0), -1)  # Background
        cv2.putText(resized_image, label, (label_x1 + 2, label_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Save the output image
    output_dir = "./output/image/"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, resized_image)
    print(f"Output saved to {output_path}")

    # Display the resized image
    window_name = "Detected Objects"
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a resizable window
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
