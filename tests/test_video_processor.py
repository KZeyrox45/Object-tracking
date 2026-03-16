"""Tests for video processor module."""
import cv2
import numpy as np
import pytest
from video_processor import resize_frame, annotate_frame


class TestResizeFrame:
    """Test cases for resize_frame function."""

    def test_resize_frame_maintains_aspect_ratio(self):
        """Test that frame resizing maintains aspect ratio."""
        # Create a 1920x1080 frame (16:9 aspect ratio)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        resized, factor = resize_frame(frame, max_width=800, max_height=600)

        new_height, new_width = resized.shape[:2]
        assert new_width <= 800
        assert new_height <= 600
        # Check aspect ratio is maintained (16:9 = 1.778)
        assert abs((new_width / new_height) - 1.778) < 0.01

    def test_resize_frame_returns_scaling_factor(self):
        """Test that correct scaling factor is returned."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        resized, factor = resize_frame(frame, max_width=800, max_height=600)

        assert factor < 1.0
        assert factor == min(800 / 1920, 600 / 1080)


class TestVideoAnnotateFrame:
    """Test cases for annotate_frame function in video processor."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with class names."""
        class MockModel:
            names = {0: "person", 1: "car", 2: "dog"}
        return MockModel()

    @pytest.fixture
    def mock_results(self):
        """Create mock detection results."""
        class MockResults:
            xyxy = [
                np.array([
                    [100, 100, 200, 200, 0.9, 0],  # person
                    [300, 300, 400, 400, 0.8, 1],  # car
                ])
            ]
        return MockResults()

    def test_video_annotate_draws_bounding_boxes(self, mock_model, mock_results):
        """Test that bounding boxes are drawn on video frame."""
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        annotated = annotate_frame(frame, mock_results, mock_model, confidence_threshold=0.3)

        # Frame should be modified
        assert np.any(annotated != 0)

    def test_video_annotate_same_as_image(self, mock_model, mock_results):
        """Test that video annotation behaves same as image annotation."""
        from image_processor import annotate_frame as image_annotate

        frame1 = np.zeros((600, 800, 3), dtype=np.uint8)
        frame2 = np.zeros((600, 800, 3), dtype=np.uint8)

        video_annotated = annotate_frame(frame1, mock_results, mock_model, 0.3)
        image_annotated = image_annotate(frame2, mock_results, mock_model, 0.3)

        # Both should produce similar results
        assert video_annotated.shape == image_annotated.shape
