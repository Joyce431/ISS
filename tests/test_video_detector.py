"""Tests for video deepfake detection"""
import os
import tempfile
import numpy as np
import pytest
import cv2

from app.models.video_detector import VideoDeepfakeDetector


@pytest.fixture(scope="module")
def detector():
    """Shared detector instance"""
    return VideoDeepfakeDetector(device="cpu", max_frames=10)


@pytest.fixture
def sample_video():
    """Generate a simple synthetic test video"""
    path = tempfile.mktemp(suffix=".mp4")
    fps = 10
    duration = 2  # seconds
    width, height = 320, 240
    total_frames = fps * duration
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for i in range(total_frames):
        # Create a frame with moving pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Moving rectangle
        x = int((i / total_frames) * (width - 60))
        cv2.rectangle(frame, (x, 80), (x + 50, 130), (0, 255, 0), -1)
        # Add text
        cv2.putText(frame, f"Frame {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Add some noise
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        writer.write(frame)
    
    writer.release()
    yield path
    
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def short_video():
    """Generate a very short video (2 frames)"""
    path = tempfile.mktemp(suffix=".mp4")
    fps = 5
    width, height = 160, 120
    total_frames = 2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for i in range(total_frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    
    writer.release()
    yield path
    
    if os.path.exists(path):
        os.unlink(path)


class TestVideoDetectorInitialization:
    def test_detector_initializes(self, detector):
        assert detector is not None
        assert detector.max_frames == 10
        assert detector.image_detector is not None

    def test_has_face_cascade(self, detector):
        assert detector.face_cascade is not None


class TestVideoDetection:
    def test_detect_returns_dict(self, detector, sample_video):
        result = detector.detect(sample_video)
        assert isinstance(result, dict)

    def test_has_required_fields(self, detector, sample_video):
        result = detector.detect(sample_video)
        assert "risk_score" in result
        assert "risk_level" in result
        assert "is_deepfake" in result
        assert "detector_type" in result

    def test_detector_type(self, detector, sample_video):
        result = detector.detect(sample_video)
        assert result["detector_type"] == "video_deepfake"

    def test_risk_score_range(self, detector, sample_video):
        result = detector.detect(sample_video)
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_risk_level_valid(self, detector, sample_video):
        result = detector.detect(sample_video)
        assert result["risk_level"] in [
            "MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"
        ]

    def test_has_video_info(self, detector, sample_video):
        result = detector.detect(sample_video)
        assert "video_info" in result
        info = result["video_info"]
        assert "fps" in info
        assert "duration" in info
        assert "sampled_frames" in info

    def test_has_components(self, detector, sample_video):
        result = detector.detect(sample_video)
        assert "components" in result
        comps = result["components"]
        assert "frame_analysis" in comps
        assert "temporal_analysis" in comps
        assert "face_tracking" in comps

    def test_has_recommendations(self, detector, sample_video):
        result = detector.detect(sample_video)
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0

    def test_is_deepfake_boolean(self, detector, sample_video):
        result = detector.detect(sample_video)
        assert isinstance(result["is_deepfake"], bool)

    def test_confidence_range(self, detector, sample_video):
        result = detector.detect(sample_video)
        assert 0.0 <= result["confidence"] <= 1.0


class TestFrameAnalysis:
    def test_frame_analysis_runs(self, detector, sample_video):
        result = detector.detect(sample_video)
        frame_analysis = result["components"]["frame_analysis"]
        assert "avg_score" in frame_analysis
        assert "max_score" in frame_analysis
        assert "frames_analyzed" in frame_analysis

    def test_frames_analyzed_positive(self, detector, sample_video):
        result = detector.detect(sample_video)
        assert result["components"]["frame_analysis"]["frames_analyzed"] > 0

    def test_scores_in_range(self, detector, sample_video):
        result = detector.detect(sample_video)
        fa = result["components"]["frame_analysis"]
        assert 0.0 <= fa["avg_score"] <= 1.0
        assert 0.0 <= fa["max_score"] <= 1.0


class TestTemporalAnalysis:
    def test_temporal_analysis_runs(self, detector, sample_video):
        result = detector.detect(sample_video)
        temporal = result["components"]["temporal_analysis"]
        assert "temporal_score" in temporal
        assert 0.0 <= temporal["temporal_score"] <= 1.0

    def test_temporal_has_findings(self, detector, sample_video):
        result = detector.detect(sample_video)
        temporal = result["components"]["temporal_analysis"]
        assert "findings" in temporal
        assert isinstance(temporal["findings"], list)


class TestFaceTracking:
    def test_face_tracking_runs(self, detector, sample_video):
        result = detector.detect(sample_video)
        face = result["components"]["face_tracking"]
        assert "face_tracking_score" in face

    def test_face_tracking_score_range(self, detector, sample_video):
        result = detector.detect(sample_video)
        face = result["components"]["face_tracking"]
        assert 0.0 <= face["face_tracking_score"] <= 1.0


class TestEdgeCases:
    def test_short_video(self, detector, short_video):
        result = detector.detect(short_video)
        assert isinstance(result, dict)

    def test_nonexistent_file(self, detector):
        result = detector.detect("/nonexistent/video.mp4")
        assert "error" in result or result.get("risk_score", 0) == 0.0

    def test_invalid_file(self, detector, tmp_path):
        fake = tmp_path / "not_video.mp4"
        fake.write_bytes(b"not a video")
        result = detector.detect(str(fake))
        assert isinstance(result, dict)

    def test_empty_file(self, detector, tmp_path):
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        result = detector.detect(str(empty))
        assert isinstance(result, dict)


class TestConsistency:
    def test_same_video_same_result(self, detector, sample_video):
        r1 = detector.detect(sample_video)
        r2 = detector.detect(sample_video)
        # Frame scores may have small variation but should be close
        assert abs(r1["risk_score"] - r2["risk_score"]) < 0.05

    def test_video_not_modified(self, detector, sample_video):
        """Detector must not modify source video"""
        import hashlib
        def hash_file(path):
            with open(path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        
        before = hash_file(sample_video)
        detector.detect(sample_video)
        after = hash_file(sample_video)
        assert before == after


class TestFrameExtraction:
    def test_extract_frames(self, detector, sample_video):
        frames, info = detector._extract_frames(sample_video)
        assert len(frames) > 0
        assert info["sampled_frames"] == len(frames)

    def test_max_frames_respected(self, sample_video):
        detector = VideoDeepfakeDetector(device="cpu", max_frames=5)
        frames, _ = detector._extract_frames(sample_video)
        assert len(frames) <= 5

    def test_frame_shape(self, detector, sample_video):
        frames, _ = detector._extract_frames(sample_video)
        for frame in frames[:3]:
            assert frame.ndim == 3
            assert frame.shape[2] == 3