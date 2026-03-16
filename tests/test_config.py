"""Tests for configuration module."""
import os
import pytest
import yaml
from config import Config


class TestConfig:
    """Test cases for Config class."""

    def test_default_config_path(self):
        """Test default config path is set correctly."""
        config = Config()
        assert "config.yaml" in config.config_path

    def test_load_default_config(self):
        """Test loading default config when file doesn't exist."""
        config = Config("nonexistent_config.yaml")
        assert config.model_path == "yolov5s.pt"
        assert config.model_source == "ultralytics/yolov5"

    def test_get_nested_value(self):
        """Test getting nested configuration values."""
        config = Config("nonexistent_config.yaml")
        value = config.get("model", "path", default="default.pt")
        assert value == "yolov5s.pt"

    def test_get_missing_value_with_default(self):
        """Test getting missing value returns default."""
        config = Config("nonexistent_config.yaml")
        value = config.get("nonexistent", "key", default="fallback")
        assert value == "fallback"

    def test_image_properties(self):
        """Test image-related config properties."""
        config = Config("nonexistent_config.yaml")
        assert config.image_input_dir == "data/image"
        assert config.image_output_dir == "output/image"
        assert config.image_confidence_threshold == 0.2
        assert config.image_max_width == 800
        assert config.image_max_height == 600

    def test_video_properties(self):
        """Test video-related config properties."""
        config = Config("nonexistent_config.yaml")
        assert config.video_input_dir == "data/video"
        assert config.video_output_dir == "output/video"
        assert config.video_confidence_threshold == 0.3
        assert config.video_fps_divisor == 2
        assert config.video_codec == "MP4V"

    def test_logging_properties(self):
        """Test logging-related config properties."""
        config = Config("nonexistent_config.yaml")
        assert config.logging_level == "INFO"
        assert "object_tracking.log" in config.logging_file

    def test_display_properties(self):
        """Test display-related config properties."""
        config = Config("nonexistent_config.yaml")
        assert config.display_fullscreen == True
        assert config.display_window_name == "Detected Objects"


class TestConfigWithFile:
    """Test cases for Config with actual YAML file."""

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create a temporary config file."""
        config_data = {
            "model": {"path": "custom.pt", "source": "custom/repo"},
            "detection": {"confidence_threshold": 0.5},
        }
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        return str(config_file)

    def test_load_custom_config(self, temp_config_file):
        """Test loading configuration from YAML file."""
        config = Config(temp_config_file)
        assert config.model_path == "custom.pt"
        assert config.model_source == "custom/repo"

    def test_custom_config_fallback(self, temp_config_file):
        """Test that missing values fall back to defaults."""
        config = Config(temp_config_file)
        # Custom value
        assert config.model_path == "custom.pt"
        # Default value (not in custom config)
        assert config.image_input_dir == "data/image"
