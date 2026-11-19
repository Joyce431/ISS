"""Brand Logo Detection using YOLOv8 for detecting copied logos in phishing"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class LogoDetector:
    """
    Detects brand logos in images/emails to identify impersonation.
    Uses YOLOv8 trained on common brand logos.
    """
    
    # Common brands targeted in phishing
    TARGETED_BRANDS = [
        "paypal", "microsoft", "apple", "google", "amazon",
        "netflix", "facebook", "instagram", "whatsapp", "linkedin",
        "dhl", "fedex", "ups", "usps", "bank_of_america",
        "wells_fargo", "chase", "citibank", "hsbc", "barclays"
    ]
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model for logo detection"""
        try:
            from ultralytics import YOLO
            if self.model_path:
                self.model = YOLO(self.model_path)
            else:
                # Use base YOLOv8 (to be fine-tuned on logo dataset)
                self.model = YOLO('yolov8n.pt')
            logger.info("Logo detection model loaded")
        except Exception as e:
            logger.error(f"Failed to load logo model: {e}")
            self.model = None
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """Detect logos in an image"""
        if self.model is None:
            return {"logos_detected": [], "logo_score": 0.0, "model_available": False}
        
        try:
            results = self.model(image_path, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        if conf > 0.5:
                            detections.append({
                                "class": result.names[cls],
                                "confidence": round(conf, 4),
                                "bbox": box.xyxy[0].tolist()
                            })
            
            return {
                "logos_detected": detections,
                "logo_count": len(detections),
                "logo_score": min(len(detections) * 0.1, 0.5),
                "model_available": True
            }
        except Exception as e:
            logger.error(f"Logo detection failed: {e}")
            return {"logos_detected": [], "logo_score": 0.0, "error": str(e)
}