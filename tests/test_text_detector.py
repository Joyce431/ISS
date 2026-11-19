"""Test phishing text detection"""
import pytest
from app.models.text_detector import PhishingTextDetector


@pytest.fixture
def detector():
    return PhishingTextDetector(device="cpu")


def test_obvious_phishing_detected(detector):
    """Test that obvious phishing is detected"""
    text = """
    URGENT! Your account will be suspended immediately!
    Click here to verify your password and credit card information:
    http://192.168.1.1/paypal-verify.tk
    """
    result = detector.detect(text)
    assert result["risk_score"] > 0.5
    assert result["risk_level"] in ["MEDIUM", "HIGH", "CRITICAL"]


def test_legitimate_email(detector):
    """Test that legitimate email scores low"""
    text = """
    Hi team,
    
    Just a reminder that our weekly meeting is tomorrow at 2 PM.
    Please bring your progress updates.
    
    Best regards,
    John
    """
    result = detector.detect(text)
    assert result["risk_score"] < 0.7


def test_url_analysis(detector):
    """Test URL analysis component"""
    text = "Click here: http://bit.ly/suspicious-link"
    result = detector.detect(text)
    url_findings = result["components"]["url_analysis"]["findings"]
    assert len(url_findings) > 0