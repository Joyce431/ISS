"""AI-Generated Phishing Email Detector using DistilBERT + Rule Engine"""
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Any
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class PhishingTextDetector:
    """
    Hybrid detector combining:
    1. DistilBERT (base; optionally fine-tuned) for phishing text
    2. Rule-based heuristics for phishing indicators
    3. URL analysis (domain + path + query) for malicious links
    4. Sender / display-name analysis

    Scoring philosophy:
    - The DistilBERT model contributes only through a weighted baseline.
    - It contributes to the "peak" signal ONLY when fine-tuned weights
      are provided (model_path is not None). Otherwise the untrained
      model's neutral ~0.5 output is treated as noise.
    - Strong hits from rules, URLs, or sender analysis dominate the score.
    """

    # Suspicious keywords commonly found in phishing
    URGENCY_KEYWORDS = [
        "urgent", "immediate", "act now", "verify", "suspend", "expire",
        "limited time", "final warning", "account locked", "unusual activity",
        "confirm identity", "click here", "update payment", "wire transfer",
        "confidential", "do not share", "verify immediately", "time-sensitive",
        "as soon as possible", "important notice"
    ]

    CREDENTIAL_KEYWORDS = [
        "password", "login", "username", "credentials", "sign in",
        "bank account", "credit card", "ssn", "social security",
        "pin", "otp", "one-time password", "security code"
    ]

    PAYMENT_KEYWORDS = [
        "invoice", "payment", "transfer", "wire", "bitcoin", "crypto",
        "gift card", "itunes", "western union", "moneygram", "paypal",
        "wire transfer", "bank transfer", "remittance"
    ]

    SUSPICIOUS_TLDS = [
        ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".work",
        ".click", ".link", ".loan", ".download", ".review"
    ]

    SUSPICIOUS_PATH_WORDS = [
        "login", "signin", "verify", "secure", "account",
        "update", "confirm", "password", "billing", "payment",
        "webscr", "authenticate", "authorize", "validate"
    ]

    COMMON_BRANDS = [
        "paypal", "google", "microsoft", "apple", "amazon",
        "facebook", "netflix", "bank", "wellsfargo", "chase",
        "citibank", "hsbc", "linkedin", "instagram", "whatsapp",
        "dhl", "fedex", "ups", "usps"
    ]

    URL_SHORTENERS = [
        "bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly",
        "is.gd", "buff.ly", "rebrand.ly"
    ]

    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load fine-tuned DistilBERT model, or fall back to base model"""
        try:
            if self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path, num_labels=2
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "distilbert-base-uncased"
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased", num_labels=2
                )

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Text model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load text model: {e}")
            self.model = None

    # ------------------------------------------------------------------
    # URL extraction
    # ------------------------------------------------------------------

    def _extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from text"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return re.findall(url_pattern, text, re.IGNORECASE)

    # ------------------------------------------------------------------
    # URL analysis — domain + path + query aware
    # ------------------------------------------------------------------

    def _analyze_urls(self, urls: List[str]) -> Dict[str, Any]:
        """Analyze URLs for phishing indicators (domain + path + query)"""
        findings: List[Dict[str, Any]] = []
        risk_score = 0.0

        for url in urls:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            query = parsed.query.lower()
            full_lower = url.lower()
            url_findings: List[str] = []

            # 1. IP address instead of domain
            if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain):
                url_findings.append("IP address used instead of domain")
                risk_score += 0.30

            # 2. Suspicious TLD anywhere in URL
            for tld in self.SUSPICIOUS_TLDS:
                if (domain.endswith(tld)
                        or tld in path
                        or tld in query
                        or tld in full_lower):
                    url_findings.append(f"Suspicious TLD: {tld}")
                    risk_score += 0.25
                    break

            # 3. @ symbol (credential harvesting / obfuscation)
            if '@' in url:
                url_findings.append("URL contains @ symbol (possible obfuscation)")
                risk_score += 0.30

            # 4. Excessive subdomains
            if domain.count('.') > 3:
                url_findings.append("Excessive subdomains")
                risk_score += 0.15

            # 5. Brand impersonation
            for brand in self.COMMON_BRANDS:
                if brand in full_lower and not domain.endswith(f"{brand}.com"):
                    url_findings.append(f"Possible impersonation of {brand}")
                    risk_score += 0.30
                    break

            # 6. URL shorteners
            if any(s in domain for s in self.URL_SHORTENERS):
                url_findings.append("URL shortener detected (hides destination)")
                risk_score += 0.20

            # 7. Phishing keywords in path
            path_hits = [w for w in self.SUSPICIOUS_PATH_WORDS if w in path]
            if path_hits:
                url_findings.append(
                    f"Phishing keywords in path: {', '.join(path_hits)}"
                )
                risk_score += 0.15 * min(len(path_hits), 3)

            # 8. Non-HTTPS on sensitive-sounding URL
            if parsed.scheme == 'http' and any(
                w in full_lower
                for w in ['login', 'secure', 'bank', 'payment', 'verify']
            ):
                url_findings.append("HTTP (not HTTPS) on sensitive URL")
                risk_score += 0.15

            if url_findings:
                findings.append({"url": url, "indicators": url_findings})

        return {
            "urls_found": len(urls),
            "findings": findings,
            "url_risk_score": min(risk_score, 1.0),
        }

    # ------------------------------------------------------------------
    # Rule-based analysis
    # ------------------------------------------------------------------

    def _rule_based_analysis(self, text: str) -> Dict[str, Any]:
        """Rule-based phishing indicator detection"""
        text_lower = text.lower()

        urgency_hits = [kw for kw in self.URGENCY_KEYWORDS if kw in text_lower]
        credential_hits = [kw for kw in self.CREDENTIAL_KEYWORDS if kw in text_lower]
        payment_hits = [kw for kw in self.PAYMENT_KEYWORDS if kw in text_lower]

        score = 0.0
        score += min(len(urgency_hits) * 0.08, 0.30)
        score += min(len(credential_hits) * 0.12, 0.35)
        score += min(len(payment_hits) * 0.10, 0.25)

        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.30:
            score += 0.10

        punct_ratio = sum(1 for c in text if c in '!!!???') / max(len(text), 1)
        if punct_ratio > 0.02:
            score += 0.10

        return {
            "urgency_keywords": urgency_hits,
            "credential_keywords": credential_hits,
            "payment_keywords": payment_hits,
            "caps_ratio": round(caps_ratio, 3),
            "rule_score": min(score, 1.0),
        }

    # ------------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------------

    def _model_analysis(self, text: str) -> Dict[str, Any]:
        """AI model inference for phishing detection"""
        if self.model is None:
            return {"model_score": 0.5, "model_available": False}

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                phishing_prob = probabilities[0][1].item()

            return {"model_score": phishing_prob, "model_available": True}
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            return {"model_score": 0.5, "model_available": False}

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------

    def detect(self, text: str, subject: str = "", sender: str = "") -> Dict[str, Any]:
        """
        Main detection method.

        Args:
            text:    Email body text
            subject: Email subject line
            sender:  Sender email address

        Returns:
            Detection result with risk score, level and findings
        """
        full_text = f"{subject}\n{text}" if subject else text

        # Run all analysis components
        urls = self._extract_urls(full_text)
        url_analysis = self._analyze_urls(urls)
        rule_analysis = self._rule_based_analysis(full_text)
        model_analysis = self._model_analysis(full_text)

        # Sender analysis
        sender_findings: List[str] = []
        sender_score = 0.0
        if sender:
            if '<' in sender and '>' in sender:
                display_name = sender.split('<')[0].strip()
                if any(
                    kw in display_name.lower()
                    for kw in ['ceo', 'director', 'manager', 'support', 'security']
                ):
                    sender_findings.append(
                        "Authority display name with external email"
                    )
                    sender_score += 0.20

            free_providers = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
            if any(p in sender.lower() for p in free_providers):
                sender_findings.append(
                    "Free email provider used for official communication"
                )
                sender_score += 0.15

        # ---------------------------------------------------------
        # Ensemble scoring
        # ---------------------------------------------------------
        # The DistilBERT model contributes to the weighted baseline
        # always, but only contributes to the "peak" signal when it
        # has been fine-tuned (self.model_path is set). Without
        # fine-tuning, its ~0.5 neutral output is treated as noise.
        # ---------------------------------------------------------
        model_score = float(model_analysis.get("model_score", 0.5))
        rule_score = float(rule_analysis.get("rule_score", 0.0))
        url_score = float(url_analysis.get("url_risk_score", 0.0))
        sender_score_clamped = min(float(sender_score), 1.0)

        weighted = (
            model_score * 0.25
            + rule_score * 0.35
            + url_score * 0.30
            + sender_score_clamped * 0.10
        )

        # Only trust the model for "peak" if fine-tuned weights were loaded
        model_is_trusted = (
            model_analysis.get("model_available", False)
            and self.model_path is not None
        )
        peak_candidates = [rule_score, url_score, sender_score_clamped]
        if model_is_trusted:
            peak_candidates.append(model_score)

        peak = max(peak_candidates) if peak_candidates else 0.0

        final_score = max(weighted, peak * 0.85)

        # Bonus when multiple independent signals agree
        signals_fired = sum(
            1 for s in (rule_score, url_score, sender_score_clamped) if s >= 0.30
        )
        if signals_fired >= 2:
            final_score = min(final_score + 0.08, 1.0)

        final_score = round(final_score, 4)

        # Risk level
        if final_score >= 0.85:
            risk_level = "CRITICAL"
        elif final_score >= 0.70:
            risk_level = "HIGH"
        elif final_score >= 0.50:
            risk_level = "MEDIUM"
        elif final_score >= 0.30:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"

        # Recommendations
        recommendations: List[str] = []
        if final_score >= 0.70:
            recommendations.append("Do NOT click any links or open attachments")
            recommendations.append("Report to IT security immediately")
            recommendations.append("Verify sender through a separate trusted channel")
        elif final_score >= 0.50:
            recommendations.append("Exercise caution before interacting")
            recommendations.append("Verify sender identity independently")
        else:
            recommendations.append("No immediate action required")
            recommendations.append("Stay vigilant for unusual requests")

        return {
            "detector_type": "phishing_text",
            "risk_score": final_score,
            "risk_level": risk_level,
            "is_phishing": final_score >= 0.55,
            "confidence": round(abs(final_score - 0.5) * 2, 4),
            "threshold_used": 0.55,
            "signals_fired": signals_fired,
            "model_trusted": model_is_trusted,
            "components": {
                "model_analysis": model_analysis,
                "rule_analysis": rule_analysis,
                "url_analysis": url_analysis,
                "sender_analysis": {
                    "findings": sender_findings,
                    "score": round(sender_score, 4),
                },
            },
            "recommendations": recommendations,
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
        }