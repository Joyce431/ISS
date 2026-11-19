"""Tests for audio deepfake detection"""
import os
import tempfile
import numpy as np
import pytest
import soundfile as sf

from app.models.audio_detector import AudioDeepfakeDetector


@pytest.fixture(scope="module")
def detector():
    """Shared detector instance for all tests"""
    return AudioDeepfakeDetector(device="cpu")


@pytest.fixture
def real_audio_file():
    """Generate a synthetic but 'natural-like' audio file"""
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Create a signal with harmonic structure and varying pitch
    fundamental = 150 + 20 * np.sin(2 * np.pi * 0.5 * t)
    signal = np.zeros_like(t)
    for harmonic in range(1, 6):
        signal += (1.0 / harmonic) * np.sin(2 * np.pi * harmonic * fundamental * t)
    
    # Add amplitude modulation and noise
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    signal *= envelope
    signal += 0.01 * np.random.randn(len(signal))
    signal = signal / np.max(np.abs(signal)) * 0.9
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        sf.write(f.name, signal, sr)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def silent_audio_file():
    """Generate a silent audio file"""
    sr = 16000
    signal = np.zeros(sr * 2, dtype=np.float32)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        sf.write(f.name, signal, sr)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def short_audio_file():
    """Generate a very short audio file"""
    sr = 16000
    signal = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(sr * 0.1)))
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        sf.write(f.name, signal, sr)
        yield f.name
    os.unlink(f.name)


class TestAudioDetectorInitialization:
    def test_detector_initializes(self, detector):
        assert detector is not None
        assert detector.sample_rate == 16000

    def test_detector_has_processor_or_fallback(self, detector):
        # Processor may be None if transformers unavailable
        assert hasattr(detector, "processor")
        assert hasattr(detector, "feature_extractor")


class TestAudioDetection:
    def test_detect_returns_dict(self, detector, real_audio_file):
        result = detector.detect(real_audio_file)
        assert isinstance(result, dict)
        assert "risk_score" in result
        assert "risk_level" in result

    def test_risk_score_range(self, detector, real_audio_file):
        result = detector.detect(real_audio_file)
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_risk_level_valid(self, detector, real_audio_file):
        result = detector.detect(real_audio_file)
        assert result["risk_level"] in [
            "MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"
        ]

    def test_detector_type(self, detector, real_audio_file):
        result = detector.detect(real_audio_file)
        assert result["detector_type"] == "audio_deepfake"

    def test_has_components(self, detector, real_audio_file):
        result = detector.detect(real_audio_file)
        assert "components" in result
        comps = result["components"]
        assert "model_analysis" in comps
        assert "spectral_analysis" in comps
        assert "prosody_analysis" in comps

    def test_has_recommendations(self, detector, real_audio_file):
        result = detector.detect(real_audio_file)
        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) > 0

    def test_duration_reported(self, detector, real_audio_file):
        result = detector.detect(real_audio_file)
        assert "duration_seconds" in result
        assert result["duration_seconds"] > 0

    def test_is_deepfake_boolean(self, detector, real_audio_file):
        result = detector.detect(real_audio_file)
        assert isinstance(result["is_deepfake"], bool)

    def test_confidence_range(self, detector, real_audio_file):
        result = detector.detect(real_audio_file)
        assert 0.0 <= result["confidence"] <= 1.0


class TestSpectralAnalysis:
    def test_spectral_analysis_runs(self, detector, real_audio_file):
        result = detector.detect(real_audio_file)
        spectral = result["components"]["spectral_analysis"]
        assert "spectral_score" in spectral
        assert 0.0 <= spectral["spectral_score"] <= 1.0

    def test_spectral_has_findings(self, detector, real_audio_file):
        result = detector.detect(real_audio_file)
        spectral = result["components"]["spectral_analysis"]
        assert "findings" in spectral
        assert isinstance(spectral["findings"], list)


class TestProsodyAnalysis:
    def test_prosody_analysis_runs(self, detector, real_audio_file):
        result = detector.detect(real_audio_file)
        prosody = result["components"]["prosody_analysis"]
        assert "prosody_score" in prosody
        assert 0.0 <= prosody["prosody_score"] <= 1.0

    def test_prosody_has_pitch(self, detector, real_audio_file):
        result = detector.detect(real_audio_file)
        prosody = result["components"]["prosody_analysis"]
        # Pitch may not be present if no voiced segments
        if "pitch_mean" in prosody:
            assert prosody["pitch_mean"] > 0


class TestEdgeCases:
    def test_silent_audio(self, detector, silent_audio_file):
        result = detector.detect(silent_audio_file)
        # Should complete without crashing
        assert "risk_score" in result or "error" in result

    def test_short_audio(self, detector, short_audio_file):
        result = detector.detect(short_audio_file)
        assert isinstance(result, dict)

    def test_nonexistent_file(self, detector):
        result = detector.detect("/nonexistent/audio.wav")
        assert result.get("risk_score", 0) == 0.0 or "error" in result

    def test_invalid_extension(self, detector, tmp_path):
        fake = tmp_path / "not_audio.xyz"
        fake.write_bytes(b"not audio data")
        result = detector.detect(str(fake))
        assert isinstance(result, dict)


class TestConsistency:
    def test_same_file_same_result(self, detector, real_audio_file):
        r1 = detector.detect(real_audio_file)
        r2 = detector.detect(real_audio_file)
        assert abs(r1["risk_score"] - r2["risk_score"]) < 0.001

    def test_file_not_modified(self, detector, real_audio_file):
        """Detector should not modify the source audio file"""
        import hashlib
        def hash_file(path):
            with open(path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        
        before = hash_file(real_audio_file)
        detector.detect(real_audio_file)
        after = hash_file(real_audio_file)
        assert before == after


class TestSupportedFormats:
    @pytest.mark.parametrize("ext", [".wav", ".flac"])
    def test_supported_formats(self, detector, ext, tmp_path):
        sr = 16000
        signal = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr))
        path = tmp_path / f"test{ext}"
        sf.write(str(path), signal, sr)
        result = detector.detect(str(path))
        assert isinstance(result, dict)