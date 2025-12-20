"""Video frame extraction using ffmpeg."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Generator

from video_search.models import (
    ExtractionConfig,
    FrameData,
    VideoMetadata,
    VideoStatus,
)


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Extract metadata from a video file using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        VideoMetadata with video information.

    Raises:
        FileNotFoundError: If video file doesn't exist.
        RuntimeError: If ffprobe fails.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Generate video ID from path hash
    video_id = hashlib.md5(str(video_path.absolute()).encode()).hexdigest()[:12]

    try:
        # Run ffprobe to get video info
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)

        # Extract video stream info
        video_stream = None
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if video_stream is None:
            raise RuntimeError(f"No video stream found in {video_path}")

        format_info = probe_data.get("format", {})

        # Parse frame rate
        fps = 0.0
        if "r_frame_rate" in video_stream:
            fps_parts = video_stream["r_frame_rate"].split("/")
            if len(fps_parts) == 2 and int(fps_parts[1]) > 0:
                fps = int(fps_parts[0]) / int(fps_parts[1])

        return VideoMetadata(
            video_id=video_id,
            filename=video_path.name,
            filepath=str(video_path.absolute()),
            duration=float(format_info.get("duration", 0)),
            fps=fps,
            width=int(video_stream.get("width", 0)),
            height=int(video_stream.get("height", 0)),
            codec=video_stream.get("codec_name", ""),
            size_bytes=int(format_info.get("size", 0)),
            status=VideoStatus.PENDING,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse ffprobe output: {e}") from e


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    config: ExtractionConfig | None = None,
) -> Generator[FrameData, None, None]:
    """Extract frames from a video at regular intervals.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save extracted frames.
        config: Extraction configuration.

    Yields:
        FrameData for each extracted frame.

    Raises:
        FileNotFoundError: If video file doesn't exist.
        RuntimeError: If ffmpeg fails.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    config = config or ExtractionConfig()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Get video metadata first
    metadata = get_video_metadata(video_path)

    # Create output directory
    video_frames_dir = output_dir / metadata.video_id
    video_frames_dir.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command
    # Extract one frame every N seconds
    output_pattern = str(video_frames_dir / f"frame_%06d.{config.format}")

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{config.interval_seconds}",
    ]

    # Add resize filter if specified
    if config.max_width or config.max_height:
        width = config.max_width or -1
        height = config.max_height or -1
        cmd.extend(
            [
                "-vf",
                f"fps=1/{config.interval_seconds},scale={width}:{height}:force_original_aspect_ratio=decrease",
            ]
        )

    if config.format == "jpg":
        cmd.extend(["-q:v", str(int((100 - config.quality) / 100 * 31))])

    cmd.extend(["-y", output_pattern])

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}") from e

    # Yield frame data for each extracted frame
    frame_files = sorted(video_frames_dir.glob(f"frame_*.{config.format}"))

    for i, frame_path in enumerate(frame_files):
        timestamp = i * config.interval_seconds
        frame_id = f"{metadata.video_id}_{i:06d}"

        yield FrameData(
            frame_id=frame_id,
            video_id=metadata.video_id,
            timestamp=timestamp,
            frame_number=i,
            filepath=str(frame_path.absolute()),
            width=metadata.width,
            height=metadata.height,
        )


def extract_frame_at_timestamp(
    video_path: str | Path,
    timestamp: float,
    output_path: str | Path,
    format: str = "jpg",
) -> Path:
    """Extract a single frame at a specific timestamp.

    Args:
        video_path: Path to the video file.
        timestamp: Timestamp in seconds.
        output_path: Path to save the frame.
        format: Output format (jpg, png).

    Returns:
        Path to the extracted frame.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-ss",
        str(timestamp),
        "-i",
        str(video_path),
        "-vframes",
        "1",
        "-y",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}") from e

    return output_path


def get_frame_count(video_path: str | Path, interval_seconds: float) -> int:
    """Calculate expected number of frames for a video.

    Args:
        video_path: Path to the video file.
        interval_seconds: Interval between frames.

    Returns:
        Expected number of frames.
    """
    metadata = get_video_metadata(video_path)
    return int(metadata.duration / interval_seconds) + 1


class VideoExtractor:
    """Video frame extractor with caching and progress tracking."""

    def __init__(self, config: ExtractionConfig | None = None, frames_dir: str = "./data/frames"):
        """Initialize the extractor.

        Args:
            config: Extraction configuration.
            frames_dir: Directory for extracted frames.
        """
        self.config = config or ExtractionConfig()
        self.frames_dir = Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def get_metadata(self, video_path: str | Path) -> VideoMetadata:
        """Get video metadata.

        Args:
            video_path: Path to video file.

        Returns:
            VideoMetadata.
        """
        return get_video_metadata(video_path)

    def extract(self, video_path: str | Path) -> list[FrameData]:
        """Extract frames from a video.

        Args:
            video_path: Path to video file.

        Returns:
            List of extracted FrameData.
        """
        return list(extract_frames(video_path, self.frames_dir, self.config))

    def extract_at_timestamp(self, video_path: str | Path, timestamp: float) -> FrameData:
        """Extract a frame at a specific timestamp.

        Args:
            video_path: Path to video file.
            timestamp: Timestamp in seconds.

        Returns:
            FrameData for the extracted frame.
        """
        metadata = get_video_metadata(video_path)
        frame_id = f"{metadata.video_id}_{int(timestamp * 1000):09d}"
        output_path = self.frames_dir / metadata.video_id / f"{frame_id}.{self.config.format}"

        extract_frame_at_timestamp(video_path, timestamp, output_path, self.config.format)

        return FrameData(
            frame_id=frame_id,
            video_id=metadata.video_id,
            timestamp=timestamp,
            frame_number=int(timestamp / self.config.interval_seconds),
            filepath=str(output_path.absolute()),
            width=metadata.width,
            height=metadata.height,
        )

    def get_expected_frames(self, video_path: str | Path) -> int:
        """Get expected number of frames.

        Args:
            video_path: Path to video file.

        Returns:
            Expected frame count.
        """
        return get_frame_count(video_path, self.config.interval_seconds)


def create_extractor(
    config: ExtractionConfig | None = None,
    frames_dir: str = "./data/frames",
) -> VideoExtractor:
    """Create a VideoExtractor instance.

    Args:
        config: Extraction configuration.
        frames_dir: Directory for frames.

    Returns:
        VideoExtractor instance.
    """
    return VideoExtractor(config, frames_dir)
