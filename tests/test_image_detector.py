"""Test image deepfake detection"""
import pytest
import numpy as np
from PIL import Image
import tempfile
import os
from app.models.image_detector import ImageDeepfakeDetector


@pytest.fixture
def detector():
    return ImageDeepfakeDetector(device="cpu")


@pytest.fixture
def sample_image():
    """Create a test image"""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


def test_image_analysis_runs(detector, sample_image):
    """Test that image analysis completes"""
    result = detector.detect(sample_image)
    assert "risk_score" in result
    assert "risk_level" in result
    assert 0 <= result["risk_score"] <= 1


def test_metadata_analysis(detector, sample_image):
    """Test metadata analysis"""
    result = detector.detect(sample_image)
    assert "metadata_analysis" in result["components"]