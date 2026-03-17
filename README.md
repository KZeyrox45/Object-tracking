# Object Tracking with YOLOv5

A Computer Vision object tracking project built with **YOLOv5** for real-time object detection in images and videos. Features include CLI argument parsing, YAML configuration, batch processing, comprehensive logging, and unit tests.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Object Detection** - Detect 80+ object classes using pre-trained YOLOv5
- **Streamlit UI** - Interactive web interface for easy image and video processing
- **Image Processing** - Process single images or batch process entire directories
- **Video Processing** - Process videos frame-by-frame with annotated output
- **YAML Configuration** - Centralized configuration for all settings
- **CLI Interface** - Full-featured command-line interface with argparse
- **Batch Processing** - Process multiple files efficiently
- **Logging** - Comprehensive logging to file and console
- **Unit Tests** - 33+ tests covering configuration, processing, and CLI

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (for cloning the repository)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Object-tracking
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   # Production dependencies
   pip install -r requirements.txt

   # Development dependencies (optional, for testing)
   pip install -r requirements-dev.txt
   ```

4. **Verify installation:**
   ```bash
   python main.py --help
   ```

## Quick Start

### Run Streamlit Web UI

```bash
streamlit run app.py
```

### Process a Single Image

```bash
python main.py --image cat.jpg
```

### Process a Single Video

```bash
python main.py --video clip.mp4
```

### Batch Process All Images

```bash
python main.py --image --all
```

### Batch Process All Videos (Headless)

```bash
python main.py --video --all --no-display
```

## CLI Commands Reference

### Basic Usage

| Command | Description |
|---------|-------------|
| `python main.py --image <file>` | Process a single image |
| `python main.py --video <file>` | Process a single video |
| `python main.py --image --all` | Process all images in data directory |
| `python main.py --video --all` | Process all videos in data directory |
| `python main.py --image --pattern "*.jpg"` | Process images matching pattern |

### File Selection Options

| Option | Description | Example |
|--------|-------------|---------|
| `--image [FILE]` | Enable image mode (optional filename) | `--image photo.jpg` |
| `--video [FILE]` | Enable video mode (optional filename) | `--video clip.mp4` |
| `--all` | Process all files in input directory | `--image --all` |
| `--pattern` | Glob pattern for file selection | `--pattern "test_*.jpg"` |

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to custom YAML config file | `config/config.yaml` |
| `--confidence` | Confidence threshold (0-1) | 0.2 (image), 0.3 (video) |
| `--model` | Path to custom model weights | `yolov5s.pt` |

### Processing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--no-display` | Disable display window (batch mode) | Display enabled |
| `--no-fullscreen` | Disable fullscreen display | Fullscreen enabled |
| `--output-dir` | Custom output directory | `output/image/` or `output/video/` |
| `--max-width` | Maximum display width | 800 |
| `--max-height` | Maximum display height | 600 |

### Logging Options

| Option | Description | Default |
|--------|-------------|---------|
| `--log-level` | Logging level | `INFO` |
| `--log-file` | Custom log file path | `logs/object_tracking.log` |

**Available log levels:** `DEBUG`, `INFO`, `WARNING`, `ERROR`

## Usage Examples

### Example 1: Process a Specific Image

```bash
python main.py --image ./data/image/sample.jpg
```

### Example 2: Batch Process All JPEG Images

```bash
python main.py --image --pattern "*.jpg"
```

### Example 3: Process Video with Custom Confidence

```bash
python main.py --video traffic.mp4 --confidence 0.5
```

### Example 4: Headless Batch Video Processing

```bash
python main.py --video --all --no-display --log-level DEBUG
```

### Example 5: Use Custom Configuration

```bash
python main.py --image photo.jpg --config custom_config.yaml
```

### Example 6: Custom Output Directory

```bash
python main.py --image --all --output-dir ./results/images
```

### Example 7: Process with Custom Model

```bash
python main.py --video clip.mp4 --model yolov5m.pt
```

## Configuration

### Default Configuration File

Edit `config/config.yaml` to customize default settings:

```yaml
# Model settings
model:
  path: yolov5s.pt
  source: ultralytics/yolov5

# Detection settings
detection:
  confidence_threshold: 0.3
  iou_threshold: 0.45

# Image processing
image:
  input_dir: data/image
  output_dir: output/image
  max_width: 800
  max_height: 600
  confidence_threshold: 0.2

# Video processing
video:
  input_dir: data/video
  output_dir: output/video
  max_width: 800
  max_height: 600
  confidence_threshold: 0.3
  fps_divisor: 2
  codec: MP4V

# Display settings
display:
  fullscreen: true
  window_name: Detected Objects

# Logging settings
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/object_tracking.log
```

### Configuration Priority

Settings are applied in this order (highest priority first):

1. CLI arguments (e.g., `--confidence 0.5`)
2. Custom config file (e.g., `--config custom.yaml`)
3. Default config file (`config/config.yaml`)
4. Built-in defaults

## Project Structure

```
Object-tracking/
├── main.py                      # CLI entry point
├── yolo_model.py                # Model loading and inference
├── image_processor.py           # Image processing functions
├── video_processor.py           # Video processing functions
├── yolov5s.pt                   # Pre-trained YOLOv5 weights
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development dependencies
├── pytest.ini                   # Pytest configuration
├── config/
│   ├── __init__.py              # Config class
│   └── config.yaml              # YAML configuration
├── data/
│   ├── image/                   # Input images
│   └── video/                   # Input videos
├── output/
│   ├── image/                   # Processed image output
│   └── video/                   # Processed video output
├── tests/                       # Unit tests
│   ├── test_config.py
│   ├── test_image_processor.py
│   ├── test_video_processor.py
│   └── test_main.py
└── logs/                        # Log files
    └── object_tracking.log
```

## Testing

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=. --cov-report=html
```

### Run Specific Test File

```bash
pytest tests/test_config.py -v
```

### Run Specific Test

```bash
pytest tests/test_config.py::TestConfig::test_default_config_path -v
```

## Output

### Images
- **Location:** `output/image/`
- **Format:** Same as input (JPEG, PNG, etc.)
- **Content:** Annotated with bounding boxes and labels

### Videos
- **Location:** `output/video/`
- **Format:** MP4 (with `_processed` suffix)
- **Content:** Annotated video with bounding boxes and labels

### Logs
- **Location:** `logs/object_tracking.log`
- **Format:** Timestamped log entries
- **Content:** Processing status, errors, detection counts

## Supported Models

The project supports any YOLOv5 model variant:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov5s.pt` | Small | Fast | Good |
| `yolov5m.pt` | Medium | Medium | Better |
| `yolov5l.pt` | Large | Slow | Best |
| `yolov5x.pt` | Extra Large | Slowest | Highest |

To use a different model:
```bash
python main.py --image photo.jpg --model yolov5m.pt
```

## Troubleshooting

### Model Fails to Load
- Ensure `yolov5s.pt` exists in the project root
- Check internet connection for `torch.hub` download
- Try manually downloading the model weights

### Video Codec Error
- The default `MP4V` codec may not work on all systems
- Try alternative codecs: `XVID`, `MJPG`, `H264`
- Edit `config/config.yaml` to change codec

### OpenCV Display Issues
- Use `--no-display` for headless environments
- On Linux, ensure X11 forwarding is enabled
- On WSL, use an X server like VcXsrv

### No Files Found
- Verify files exist in `data/image/` or `data/video/`
- Check `--pattern` syntax (use quotes: `"*.jpg"`)
- Ensure file extensions match (case-sensitive on Linux)

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
