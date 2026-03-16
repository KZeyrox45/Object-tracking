"""Main entry point for object tracking application."""
import argparse
import logging
import os
import sys
import glob

from config import Config
from yolo_model import load_model
from image_processor import process_image, process_image_batch
from video_processor import process_video, process_video_batch


def setup_logging(config: Config, log_file: str = None) -> None:
    """
    Configure logging for the application.

    Args:
        config: Configuration object.
        log_file: Optional log file path override.
    """
    log_file = log_file or config.logging_file

    # Create logs directory
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.logging_level),
        format=config.logging_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_files_from_pattern(input_dir: str, pattern: str) -> list:
    """
    Get list of files matching pattern in directory.

    Args:
        input_dir: Base input directory.
        pattern: File pattern (filename, glob pattern, or 'all').

    Returns:
        List of file paths.
    """
    if pattern == "all":
        # Get all files in directory
        files = glob.glob(os.path.join(input_dir, "*"))
        return [f for f in files if os.path.isfile(f)]
    elif "*" in pattern or "?" in pattern:
        # Glob pattern
        files = glob.glob(os.path.join(input_dir, pattern))
        return [f for f in files if os.path.isfile(f)]
    else:
        # Single file
        return [os.path.join(input_dir, pattern)]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Object tracking using YOLOv5 for images and videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  python main.py --image cat.jpg

  # Process all images in data directory
  python main.py --image --all

  # Process images matching pattern
  python main.py --image --pattern "*.jpg"

  # Process a video with custom confidence
  python main.py --video clip.mp4 --confidence 0.5

  # Batch process videos without display
  python main.py --video --all --no-display

  # Use custom config file
  python main.py --image photo.jpg --config custom_config.yaml
        """,
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--image",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help="Process image mode (provide filename or use --all/--pattern)",
    )
    mode_group.add_argument(
        "--video",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help="Process video mode (provide filename or use --all/--pattern)",
    )

    # File selection
    file_group = parser.add_argument_group("File selection")
    file_group.add_argument(
        "--all",
        action="store_true",
        help="Process all files in the input directory",
    )
    file_group.add_argument(
        "--pattern",
        type=str,
        help="Glob pattern for file selection (e.g., '*.jpg', 'test_*.mp4')",
    )

    # Configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config YAML file",
    )
    config_group.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confidence threshold (overrides config)",
    )
    config_group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model weights (overrides config)",
    )

    # Processing options
    proc_group = parser.add_argument_group("Processing options")
    proc_group.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display window (useful for batch processing)",
    )
    proc_group.add_argument(
        "--no-fullscreen",
        action="store_true",
        help="Disable fullscreen display",
    )
    proc_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    proc_group.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Maximum display width (overrides config)",
    )
    proc_group.add_argument(
        "--max-height",
        type=int,
        default=None,
        help="Maximum display height (overrides config)",
    )

    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level (overrides config)",
    )
    log_group.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (overrides config)",
    )

    args = parser.parse_args()

    # Validate arguments and determine mode
    if args.image is not None:
        args.mode = "image"
        args.file = args.image if args.image else None
    elif args.video is not None:
        args.mode = "video"
        args.file = args.video if args.video else None
    else:
        parser.error("Must specify either --image or --video")

    # Check for conflicting file selection options
    if args.file and (args.all or args.pattern):
        parser.error("Cannot specify both filename and --all/--pattern")

    return args


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = Config(args.config)

    # Override config with command line arguments
    if args.log_level:
        config._config["logging"] = config._config.get("logging", {})
        config._config["logging"]["level"] = args.log_level

    # Setup logging
    setup_logging(config, args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("Starting object tracking application")
    logger.info(f"Mode: {args.mode}")

    # Load model
    model_path = args.model or config.model_path
    model_source = config.model_source
    model = load_model(model_path, model_source)

    # Determine files to process
    if args.mode == "image":
        input_dir = config.image_input_dir
        output_dir = args.output_dir or config.image_output_dir
        confidence = args.confidence or config.image_confidence_threshold
        max_width = args.max_width or config.image_max_width
        max_height = args.max_height or config.image_max_height
        display = not args.no_display
        fullscreen = not args.no_fullscreen

        # Get files
        if args.all:
            files = get_files_from_pattern(input_dir, "all")
        elif args.pattern:
            files = get_files_from_pattern(input_dir, args.pattern)
        else:
            files = [os.path.join(input_dir, args.file)] if args.file else []

        if not files:
            logger.error(f"No files found to process in {input_dir}")
            sys.exit(1)

        logger.info(f"Found {len(files)} image(s) to process")

        # Process
        if len(files) == 1:
            result = process_image(
                image_path=files[0],
                model=model,
                confidence_threshold=confidence,
                max_width=max_width,
                max_height=max_height,
                output_dir=output_dir,
                display=display,
                window_name=config.display_window_name,
                fullscreen=fullscreen,
            )
            if result:
                logger.info(f"Successfully processed: {files[0]}")
            else:
                logger.error(f"Failed to process: {files[0]}")
                sys.exit(1)
        else:
            results = process_image_batch(
                image_paths=files,
                model=model,
                confidence_threshold=confidence,
                max_width=max_width,
                max_height=max_height,
                output_dir=output_dir,
                display=display,
                window_name=config.display_window_name,
                fullscreen=fullscreen,
            )
            logger.info(
                f"Batch complete: {results['success']}/{results['total']} successful, "
                f"{results['total_detections']} total detections"
            )
            if results["failed"] > 0:
                logger.warning(f"Failed files: {results['errors']}")

    elif args.mode == "video":
        input_dir = config.video_input_dir
        output_dir = args.output_dir or config.video_output_dir
        confidence = args.confidence or config.video_confidence_threshold
        max_width = args.max_width or config.video_max_width
        max_height = args.max_height or config.video_max_height
        fps_divisor = config.video_fps_divisor
        codec = config.video_codec
        display = not args.no_display
        fullscreen = not args.no_fullscreen

        # Get files
        if args.all:
            files = get_files_from_pattern(input_dir, "all")
        elif args.pattern:
            files = get_files_from_pattern(input_dir, args.pattern)
        else:
            files = [os.path.join(input_dir, args.file)] if args.file else []

        if not files:
            logger.error(f"No files found to process in {input_dir}")
            sys.exit(1)

        logger.info(f"Found {len(files)} video(s) to process")

        # Process
        if len(files) == 1:
            result = process_video(
                video_path=files[0],
                model=model,
                confidence_threshold=confidence,
                max_width=max_width,
                max_height=max_height,
                output_dir=output_dir,
                fps_divisor=fps_divisor,
                codec=codec,
                display=display,
                window_name=config.display_window_name,
                fullscreen=fullscreen,
            )
            if result:
                logger.info(f"Successfully processed: {files[0]}")
            else:
                logger.error(f"Failed to process: {files[0]}")
                sys.exit(1)
        else:
            results = process_video_batch(
                video_paths=files,
                model=model,
                confidence_threshold=confidence,
                max_width=max_width,
                max_height=max_height,
                output_dir=output_dir,
                fps_divisor=fps_divisor,
                codec=codec,
                display=display,
                window_name=config.display_window_name,
                fullscreen=fullscreen,
            )
            logger.info(f"Batch complete: {results['success']}/{results['total']} successful")
            if results["failed"] > 0:
                logger.warning(f"Failed files: {results['errors']}")

    logger.info("Application finished")


if __name__ == "__main__":
    main()
