"""API route definitions"""
import os
import uuid
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional, List

from app.config import settings
from app.core.orchestrator import DetectionOrchestrator
from app.core.reporter import ReportGenerator
from app.api.schemas import AnalysisResponse, BatchAnalysisResponse, HealthResponse
from app.utils.encryption import ChatEncryption, ChatStorage

router = APIRouter()
orchestrator = DetectionOrchestrator()
reporter = ReportGenerator()

# E2EE Chat Globals
chat_crypto = ChatEncryption()
chat_storage = ChatStorage()


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        models_loaded={
            "text_detector": orchestrator.text_detector.model is not None,
            "image_detector": orchestrator.image_detector.model is not None,
            "audio_detector": orchestrator.audio_detector.feature_extractor is not None,
            "video_detector": orchestrator.video_detector.image_detector.model is not None,
            "logo_detector": orchestrator.logo_detector.model is not None,
        },
    )


# ---------------------------------------------------------------------
# File / email analysis
# ---------------------------------------------------------------------
@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_file(
    file: UploadFile = File(...),
    analysis_type: Optional[str] = Form(None),
):
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    file_path = settings.UPLOAD_DIR / f"{file_id}{ext}"

    try:
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        result = orchestrator.analyze(str(file_path), analysis_type)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(files: List[UploadFile] = File(...)):
    file_paths = []
    for file in files:
        file_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1]
        file_path = settings.UPLOAD_DIR / f"{file_id}{ext}"
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)
        file_paths.append(str(file_path))
    return orchestrator.analyze_batch(file_paths)


@router.post("/analyze/email/text")
async def analyze_email_text(
    body: str = Form(...),
    subject: str = Form(""),
    sender: str = Form(""),
):
    result = orchestrator.text_detector.detect(
        text=body, subject=subject, sender=sender
    )
    return result


@router.get("/report/{analysis_id}")
async def download_report(analysis_id: str, format: str = "json"):
    return {"message": f"Report {analysis_id} in {format} format"}


# ---------------------------------------------------------------------
# End-to-End Encrypted Chat
# ---------------------------------------------------------------------
@router.post("/chat/send")
async def send_secure_message(
    sender: str = Form(...),
    recipient: str = Form(...),
    message: str = Form(...)
):
    """Encrypts a message and stores it in the recipient's inbox."""
    try:
        encrypted_payload = chat_crypto.encrypt(message)
        chat_storage.add_message(sender, recipient, encrypted_payload)
        return {
            "status": "success",
            "message": f"Encrypted message sent to {recipient}",
            "encrypted_on_wire": encrypted_payload
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/inbox/{user}")
async def get_inbox(user: str):
    """Fetches and decrypts all messages for a specific user."""
    try:
        messages = chat_storage.get_messages_for(user)
        decrypted_messages = []
        
        for m in messages:
            decrypted_messages.append({
                "id": m["id"],
                "time": m["timestamp"],
                "from": m["sender"],
                "encrypted_on_wire": m["payload"],
                "decrypted_text": chat_crypto.decrypt(m["payload"])
            })
            
        return {
            "user": user,
            "message_count": len(decrypted_messages),
            "messages": decrypted_messages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/chat/clear")
async def clear_chat():
    chat_storage.clear()
    return {"status": "cleared"}

@router.get("/chat/key-info")
async def chat_key_info():
    return {
        "algorithm": "Fernet (AES-128-CBC + HMAC-SHA256)",
        "key_preview": chat_crypto.get_key_preview(),
    }