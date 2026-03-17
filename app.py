import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import os
import tempfile
import time
import io
from yolo_model import load_model, detect_objects, get_detections_count
from image_processor import annotate_frame as annotate_image
from video_processor import annotate_frame as annotate_video, resize_frame

# Page configuration
st.set_page_config(
    page_title="Object Detection Web App",
    page_icon="🔍",
    layout="wide"
)

# Sidebar
st.sidebar.title("Settings")
model_option = st.sidebar.selectbox(
    "Select YOLOv5 Model",
    ("yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"),
    index=0
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05
)

# Performance settings
st.sidebar.subheader("Performance Settings")
skip_frames = st.sidebar.slider(
    "Frame Skipping",
    min_value=1,
    max_value=10,
    value=2,
    help="Process every N-th frame to speed up video processing."
)

device_info = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
st.sidebar.info(f"Running on: {device_info}")

# Load model
@st.cache_resource
def get_model(model_path):
    # YOLOv5 will automatically download the model if it's not found locally
    return load_model(model_path=model_path)

try:
    model = get_model(model_option)
    st.sidebar.success(f"Model {model_option} loaded!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Main UI
st.title("🔍 Object Detection Web App")
st.markdown("""
Upload an image or video to detect objects in real-time using YOLOv5.
""")

tab1, tab2 = st.tabs(["🖼️ Image Detection", "🎥 Video Detection"])

# Image Detection Tab
with tab1:
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")
    
    if uploaded_image is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(uploaded_image, use_container_width=True)
            
        with col2:
            st.subheader("Detected Objects")
            with st.spinner("Detecting..."):
                # Inference
                results = detect_objects(model, image)
                
                # Annotation
                annotated_image = annotate_image(image.copy(), results, model, confidence_threshold)
                
                # Convert BGR to RGB for Streamlit
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_image_rgb, use_container_width=True)
                
                # Stats
                count = get_detections_count(results)
                st.metric("Total Objects Detected", count)

                # Download button for annotated image
                is_success, buffer = cv2.imencode(".jpg", annotated_image)
                if is_success:
                    st.download_button(
                        label="Download Annotated Image",
                        data=buffer.tobytes(),
                        file_name="annotated_image.jpg",
                        mime="image/jpeg"
                    )

# Video Detection Tab
with tab2:
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"], key="video_uploader")
    
    if uploaded_video is not None:
        if 'processing' not in st.session_state:
            st.session_state.processing = False

        col_ctrl1, col_ctrl2 = st.columns([1, 1])
        with col_ctrl1:
            if not st.session_state.processing:
                if st.button("Start Processing", key="start_btn"):
                    st.session_state.processing = True
                    st.rerun()
            else:
                if st.button("Stop Processing", key="stop_btn"):
                    st.session_state.processing = False
                    st.rerun()

        if st.session_state.processing:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            
            out_tfile_path = tempfile.mktemp(suffix='.mp4')
            
            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Use web-friendly codec
            fourcc = cv2.VideoWriter_fourcc(*'avc1') 
            out = cv2.VideoWriter(out_tfile_path, fourcc, fps, (width, height))
            
            # Progress UI
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Side-by-side display
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Video")
                orig_frame_st = st.empty()
            with col2:
                st.subheader("Processed Stream")
                proc_frame_st = st.empty()
            
            start_time = time.time()
            frame_count = 0
            last_results = None
            
            try:
                while cap.isOpened() and st.session_state.processing:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Performance optimization: skip frames for inference
                    if frame_count % skip_frames == 0 or last_results is None:
                        last_results = detect_objects(model, frame)
                    
                    # Always annotate if we have previous results to prevent flashing
                    if last_results is not None:
                        annotated_frame = annotate_video(frame.copy(), last_results, model, confidence_threshold)
                    else:
                        annotated_frame = frame.copy()
                    
                    # Write to output
                    out.write(annotated_frame)
                    
                    # Display original and processed side-by-side
                    if frame_count % max(1, skip_frames) == 0:
                        orig_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        proc_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        
                        orig_frame_st.image(orig_frame_rgb, use_container_width=True)
                        proc_frame_st.image(proc_frame_rgb, use_container_width=True)
                    
                    # Update progress
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    
                    elapsed_time = time.time() - start_time
                    fps_actual = frame_count / elapsed_time
                    remaining_frames = total_frames - frame_count
                    eta = remaining_frames / fps_actual if fps_actual > 0 else 0
                    
                    status_text.text(f"Processing: {frame_count}/{total_frames} frames | FPS: {fps_actual:.1f} | ETA: {eta:.1f}s")
                    
                cap.release()
                out.release()
                
                if st.session_state.processing:
                    st.success(f"Video processing complete! Total time: {time.time() - start_time:.1f}s")
                    with open(out_tfile_path, "rb") as f:
                        st.download_button(
                            label="Download Annotated Video",
                            data=f,
                            file_name="annotated_video.mp4",
                            mime="video/mp4"
                        )
            except Exception as e:
                st.error(f"Error during video processing: {e}")
            finally:
                if os.path.exists(tfile.name):
                    try:
                        os.unlink(tfile.name)
                    except:
                        pass
                st.session_state.processing = False
        else:
            # Show original video if uploaded but not processing
            st.video(uploaded_video)

# Performance Info Section
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Performance Tips")
st.sidebar.write("""
- **GPU Usage**: The app automatically uses CUDA if a compatible NVIDIA GPU is detected.
- **Model Size**: `yolov5s` (Small) is fastest, while `yolov5x` (X-Large) is most accurate but slower.
- **Frame Skipping**: Increase this to process videos faster by skipping inference on intermediate frames.
- **Resolution**: High-resolution videos (4K) will process significantly slower than 720p or 1080p.
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and YOLOv5")
