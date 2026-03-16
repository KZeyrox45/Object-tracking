"""Tests for main module argument parsing."""
import pytest
import sys
from unittest.mock import patch
from main import parse_args, get_files_from_pattern


class TestParseArgs:
    """Test cases for argument parsing."""

    def test_image_mode_single_file(self):
        """Test parsing image mode with single file."""
        with patch.object(sys, 'argv', ['main.py', '--image', 'test.jpg']):
            args = parse_args()
            assert args.mode == "image"
            assert args.file == "test.jpg"
            assert args.all == False

    def test_video_mode_single_file(self):
        """Test parsing video mode with single file."""
        with patch.object(sys, 'argv', ['main.py', '--video', 'test.mp4']):
            args = parse_args()
            assert args.mode == "video"
            assert args.file == "test.mp4"

    def test_image_mode_all_flag(self):
        """Test parsing image mode with --all flag."""
        with patch.object(sys, 'argv', ['main.py', '--image', '--all']):
            args = parse_args()
            assert args.mode == "image"
            assert args.all == True
            assert args.file is None

    def test_video_mode_pattern(self):
        """Test parsing video mode with --pattern flag."""
        with patch.object(sys, 'argv', ['main.py', '--video', '--pattern', '*.mp4']):
            args = parse_args()
            assert args.mode == "video"
            assert args.pattern == "*.mp4"
            assert args.file is None

    def test_confidence_override(self):
        """Test parsing confidence threshold override."""
        with patch.object(sys, 'argv', ['main.py', '--image', 'test.jpg', '--confidence', '0.5']):
            args = parse_args()
            assert args.confidence == 0.5

    def test_no_display_flag(self):
        """Test parsing --no-display flag."""
        with patch.object(sys, 'argv', ['main.py', '--image', '--all', '--no-display']):
            args = parse_args()
            assert args.no_display == True

    def test_custom_config(self):
        """Test parsing custom config path."""
        with patch.object(sys, 'argv', ['main.py', '--image', 'test.jpg', '--config', 'custom.yaml']):
            args = parse_args()
            assert args.config == "custom.yaml"

    def test_log_level_override(self):
        """Test parsing log level override."""
        with patch.object(sys, 'argv', ['main.py', '--image', 'test.jpg', '--log-level', 'DEBUG']):
            args = parse_args()
            assert args.log_level == "DEBUG"

    def test_mutually_exclusive_mode(self):
        """Test that image and video modes are mutually exclusive."""
        with patch.object(sys, 'argv', ['main.py', 'test.jpg', 'test.mp4']):
            with pytest.raises(SystemExit):
                parse_args()


class TestGetFilesFromPattern:
    """Test cases for file pattern matching."""

    def test_get_all_files(self, tmp_path):
        """Test getting all files from directory."""
        # Create test files
        (tmp_path / "file1.jpg").touch()
        (tmp_path / "file2.jpg").touch()
        (tmp_path / "file3.png").touch()

        files = get_files_from_pattern(str(tmp_path), "all")
        assert len(files) == 3

    def test_get_pattern_files(self, tmp_path):
        """Test getting files matching glob pattern."""
        # Create test files
        (tmp_path / "file1.jpg").touch()
        (tmp_path / "file2.jpg").touch()
        (tmp_path / "file3.png").touch()

        files = get_files_from_pattern(str(tmp_path), "*.jpg")
        assert len(files) == 2
        assert all(f.endswith(".jpg") for f in files)

    def test_get_single_file(self, tmp_path):
        """Test getting single file by name."""
        # Create test file
        test_file = tmp_path / "specific.jpg"
        test_file.touch()

        files = get_files_from_pattern(str(tmp_path), "specific.jpg")
        assert len(files) == 1
        assert files[0] == str(test_file)

    def test_get_no_files(self, tmp_path):
        """Test getting no files when pattern doesn't match."""
        files = get_files_from_pattern(str(tmp_path), "*.nonexistent")
        assert len(files) == 0
