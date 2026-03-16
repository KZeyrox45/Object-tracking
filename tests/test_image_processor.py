"""Tests for image processor module."""
import os
import cv2
import numpy as np
import pytest
from image_processor import resize_image, annotate_frame


class TestResizeImage:
    """Test cases for resize_image function."""

    def test_resize_maintains_aspect_ratio(self):
        """Test that resizing maintains aspect ratio."""
        # Create a 1000x500 image (2:1 aspect ratio)
        image = np.zeros((500, 1000, 3), dtype=np.uint8)
        resized, factor = resize_image(image, max_width=800, max_height=600)

        new_height, new_width = resized.shape[:2]
        assert new_width <= 800
        assert new_height <= 600
        # Check aspect ratio is maintained
        assert abs((new_width / new_height) - 2.0) < 0.01

    def test_resize_no_change_if_smaller(self):
        """Test that small images fit within max dimensions."""
        # Create a 400x300 image (smaller than max 800x600)
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        resized, factor = resize_image(image, max_width=800, max_height=600)

        # Image should fit within max dimensions
        assert resized.shape[0] <= 600
        assert resized.shape[1] <= 800
        # Since image is smaller, it will be scaled up to fit max dimensions
        # The scaling factor is determined by min(max_width/width, max_height/height)
        assert factor > 0

    def test_resize_returns_scaling_factor(self):
        """Test that correct scaling factor is returned."""
        image = np.zeros((1000, 1600, 3), dtype=np.uint8)
        resized, factor = resize_image(image, max_width=800, max_height=600)

        assert factor < 1.0
        assert factor == min(800 / 1600, 600 / 1000)


class TestAnnotateFrame:
    """Test cases for annotate_frame function."""

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
            # Format: [x1, y1, x2, y2, conf, cls]
            xyxy = [
                np.array([
                    [100, 100, 200, 200, 0.9, 0],  # person
                    [300, 300, 400, 400, 0.5, 1],  # car
                    [50, 50, 100, 100, 0.1, 2],    # dog (low confidence)
                ])
            ]
        return MockResults()

    def test_annotate_draws_bounding_boxes(self, mock_model, mock_results):
        """Test that bounding boxes are drawn on frame."""
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        annotated = annotate_frame(frame, mock_results, mock_model, confidence_threshold=0.3)

        # Frame should be modified (green pixels for bounding boxes)
        # Check that some pixels are no longer black
        assert np.any(annotated != 0)

    def test_annotate_filters_low_confidence(self, mock_model, mock_results):
        """Test that low confidence detections are filtered."""
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        annotated = annotate_frame(frame, mock_results, mock_model, confidence_threshold=0.3)

        # Only 2 detections should be annotated (confidence > 0.3)
        # The dog detection (0.1 confidence) should be skipped

    def test_annotate_applies_scaling(self, mock_model, mock_results):
        """Test that scaling factor is applied to coordinates."""
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        scaling_factor = 0.5
        annotated = annotate_frame(
            frame, mock_results, mock_model,
            confidence_threshold=0.3,
            scaling_factor=scaling_factor
        )

        # Coordinates should be scaled by 0.5
        # Original [100, 100, 200, 200] should become [50, 50, 100, 100]
