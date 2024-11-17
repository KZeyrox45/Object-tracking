from yolo_model import load_model
from image_processor import process_image
from video_processor import process_video

def main():
    # Load YOLO model
    model = load_model()

    # Choose between image or video processing
    mode = input("Enter mode (image/video): ").strip().lower()

    if mode == "image":
        sub_path = "./data/image/"
        image_file = input("Enter image file name: ")
        image_path = (sub_path + image_file).strip()
        process_image(image_path, model)
    elif mode == "video":
        sub_path = "./data/video/"
        video_file = input("Enter video file name: ")
        video_path = (sub_path + video_file).strip()
        process_video(video_path, model)
    else:
        print("Invalid mode. Please choose 'image' or 'video'.")

if __name__ == "__main__":
    main()
