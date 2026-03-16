"""Configuration management module."""
import os
import yaml
from typing import Any, Dict


class Config:
    """Configuration manager for object tracking."""

    DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

    def __init__(self, config_path: str = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file. Uses default if not provided.
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from YAML file."""
        # Resolve relative path
        if not os.path.isabs(self.config_path):
            self.config_path = os.path.join(os.path.dirname(__file__), "..", self.config_path)

        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "model": {
                "path": "yolov5s.pt",
                "source": "ultralytics/yolov5",
            },
            "detection": {
                "confidence_threshold": 0.3,
                "iou_threshold": 0.45,
            },
            "image": {
                "input_dir": "data/image",
                "output_dir": "output/image",
                "max_width": 800,
                "max_height": 600,
                "confidence_threshold": 0.2,
            },
            "video": {
                "input_dir": "data/video",
                "output_dir": "output/video",
                "max_width": 800,
                "max_height": 600,
                "confidence_threshold": 0.3,
                "fps_divisor": 2,
                "codec": "MP4V",
            },
            "display": {
                "fullscreen": True,
                "window_name": "Detected Objects",
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/object_tracking.log",
            },
        }

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get nested configuration value.

        Args:
            keys: Nested keys to traverse (e.g., 'image', 'output_dir')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    @property
    def model_path(self) -> str:
        return self.get("model", "path", default="yolov5s.pt")

    @property
    def model_source(self) -> str:
        return self.get("model", "source", default="ultralytics/yolov5")

    @property
    def image_input_dir(self) -> str:
        return self.get("image", "input_dir", default="data/image")

    @property
    def image_output_dir(self) -> str:
        return self.get("image", "output_dir", default="output/image")

    @property
    def image_confidence_threshold(self) -> float:
        return self.get("image", "confidence_threshold", default=0.2)

    @property
    def image_max_width(self) -> int:
        return self.get("image", "max_width", default=800)

    @property
    def image_max_height(self) -> int:
        return self.get("image", "max_height", default=600)

    @property
    def video_input_dir(self) -> str:
        return self.get("video", "input_dir", default="data/video")

    @property
    def video_output_dir(self) -> str:
        return self.get("video", "output_dir", default="output/video")

    @property
    def video_confidence_threshold(self) -> float:
        return self.get("video", "confidence_threshold", default=0.3)

    @property
    def video_max_width(self) -> int:
        return self.get("video", "max_width", default=800)

    @property
    def video_max_height(self) -> int:
        return self.get("video", "max_height", default=600)

    @property
    def video_fps_divisor(self) -> int:
        return self.get("video", "fps_divisor", default=2)

    @property
    def video_codec(self) -> str:
        return self.get("video", "codec", default="MP4V")

    @property
    def display_fullscreen(self) -> bool:
        return self.get("display", "fullscreen", default=True)

    @property
    def display_window_name(self) -> str:
        return self.get("display", "window_name", default="Detected Objects")

    @property
    def logging_level(self) -> str:
        return self.get("logging", "level", default="INFO")

    @property
    def logging_format(self) -> str:
        return self.get("logging", "format", default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    @property
    def logging_file(self) -> str:
        return self.get("logging", "file", default="logs/object_tracking.log")
