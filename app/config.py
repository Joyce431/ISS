"""DeepGuard AI Configuration"""
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "DeepGuard AI"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    REPORT_DIR: Path = BASE_DIR / "data" / "reports"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    # File limits
    MAX_IMAGE_SIZE_MB: int = 50
    MAX_AUDIO_SIZE_MB: int = 100
    MAX_VIDEO_SIZE_MB: int = 500
    MAX_EMAIL_SIZE_MB: int = 25
    MAX_VIDEO_DURATION_SEC: int = 300  # 5 minutes
    
    # Supported formats
    SUPPORTED_IMAGE_FORMATS: list = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]
    SUPPORTED_AUDIO_FORMATS: list = [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"]
    SUPPORTED_VIDEO_FORMATS: list = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    SUPPORTED_EMAIL_FORMATS: list = [".eml", ".msg", ".txt"]
    
    # Detection thresholds
    PHISHING_THRESHOLD: float = 0.65
    DEEPFAKE_THRESHOLD: float = 0.70
    HIGH_RISK_THRESHOLD: float = 0.85
    
    # Model settings
    TEXT_MODEL_NAME: str = "distilbert-base-uncased"
    IMAGE_MODEL_NAME: str = "efficientnet_b4"
    AUDIO_MODEL_NAME: str = "facebook/wav2vec2-base"
    
    # Device
    DEVICE: str = "cuda"  # or "cpu"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Create directories
# Create directories
for directory in [settings.UPLOAD_DIR, settings.REPORT_DIR, settings.LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)