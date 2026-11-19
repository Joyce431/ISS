"""Logging configuration"""
import logging
import sys
from app.config import settings


def setup_logging():
    """Configure application logging"""
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(settings.LOG_DIR / "deepguard.log")
        ]
    )