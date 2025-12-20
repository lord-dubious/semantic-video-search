"""Tests for video frame extraction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from video_search.models import ExtractionConfig, VideoStatus
from video_search.extractor import (
    VideoExtractor,
    create_extractor,
    get_video_metadata,
    get_frame_count,
)


class TestGetVideoMetadata:
    """Tests for get_video_metadata function."""

    def test_video_not_found(self):
        """Test error when video doesn't exist."""
        with pytest.raises(FileNotFoundError):
            get_video_metadata("/nonexistent/video.mp4")

    @patch("subprocess.run")
    def test_get_metadata_success(self, mock_run, tmp_path, mock_ffprobe_output):
        """Test successful metadata extraction."""
        # Create a dummy video file
        video_path = tmp_path / "test.mp4"
        video_path.touch()

        mock_run.return_value = MagicMock(
            stdout=json.dumps(mock_ffprobe_output),
            returncode=0,
        )

        metadata = get_video_metadata(video_path)

        assert metadata.filename == "test.mp4"
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.fps == 30.0
        assert metadata.duration == 120.0
        assert metadata.status == VideoStatus.PENDING

    @patch("subprocess.run")
    def test_get_metadata_ffprobe_error(self, mock_run, tmp_path):
        """Test error handling when ffprobe fails."""
        video_path = tmp_path / "test.mp4"
        video_path.touch()

        mock_run.side_effect = Exception("ffprobe not found")

        with pytest.raises(Exception):
            get_video_metadata(video_path)

    @patch("subprocess.run")
    def test_get_metadata_no_video_stream(self, mock_run, tmp_path):
        """Test error when no video stream found."""
        video_path = tmp_path / "test.mp4"
        video_path.touch()

        mock_run.return_value = MagicMock(
            stdout=json.dumps({"streams": [], "format": {}}),
            returncode=0,
        )

        with pytest.raises(RuntimeError, match="No video stream"):
            get_video_metadata(video_path)


class TestGetFrameCount:
    """Tests for get_frame_count function."""

    @patch("video_search.extractor.get_video_metadata")
    def test_frame_count_calculation(self, mock_metadata, sample_video_metadata):
        """Test frame count calculation."""
        mock_metadata.return_value = sample_video_metadata

        count = get_frame_count("/path/to/video.mp4", 2.0)

        # 120 seconds / 2 second interval + 1 = 61 frames
        assert count == 61

    @patch("video_search.extractor.get_video_metadata")
    def test_frame_count_with_short_interval(self, mock_metadata, sample_video_metadata):
        """Test frame count with short interval."""
        mock_metadata.return_value = sample_video_metadata

        count = get_frame_count("/path/to/video.mp4", 0.5)

        # 120 seconds / 0.5 second interval + 1 = 241 frames
        assert count == 241


class TestVideoExtractor:
    """Tests for VideoExtractor class."""

    def test_init_default_config(self):
        """Test extractor initialization with defaults."""
        extractor = VideoExtractor()
        assert extractor.config.interval_seconds == 2.0
        assert extractor.config.format == "jpg"

    def test_init_custom_config(self, sample_extraction_config, tmp_path):
        """Test extractor with custom config."""
        extractor = VideoExtractor(
            config=sample_extraction_config,
            frames_dir=str(tmp_path),
        )
        assert extractor.config.interval_seconds == 2.0
        assert extractor.frames_dir == tmp_path

    @patch("video_search.extractor.get_video_metadata")
    def test_get_metadata(self, mock_get_metadata, sample_video_metadata):
        """Test get_metadata method."""
        mock_get_metadata.return_value = sample_video_metadata
        extractor = VideoExtractor()

        metadata = extractor.get_metadata("/path/to/video.mp4")

        assert metadata.video_id == "abc123def456"
        mock_get_metadata.assert_called_once()

    @patch("video_search.extractor.get_video_metadata")
    def test_get_expected_frames(self, mock_get_metadata, sample_video_metadata):
        """Test get_expected_frames method."""
        mock_get_metadata.return_value = sample_video_metadata
        extractor = VideoExtractor()

        count = extractor.get_expected_frames("/path/to/video.mp4")

        assert count == 61  # 120s / 2s interval + 1


class TestCreateExtractor:
    """Tests for create_extractor factory function."""

    def test_create_extractor_default(self):
        """Test creating extractor with defaults."""
        extractor = create_extractor()
        assert isinstance(extractor, VideoExtractor)

    def test_create_extractor_with_config(self, sample_extraction_config, tmp_path):
        """Test creating extractor with config."""
        extractor = create_extractor(
            config=sample_extraction_config,
            frames_dir=str(tmp_path),
        )
        assert extractor.config == sample_extraction_config


class TestExtractionConfig:
    """Tests for extraction configuration."""

    def test_config_with_resize(self):
        """Test config with resize options."""
        config = ExtractionConfig(
            max_width=640,
            max_height=480,
        )
        assert config.max_width == 640
        assert config.max_height == 480

    def test_config_quality_range(self):
        """Test quality validation."""
        config = ExtractionConfig(quality=1)
        assert config.quality == 1

        config = ExtractionConfig(quality=100)
        assert config.quality == 100
