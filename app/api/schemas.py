"""Pydantic schemas for API requests/responses"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class AnalysisRequest(BaseModel):
    analysis_type: Optional[str] = Field(None, description="Override file type detection")
    include_report: bool = Field(True, description="Generate downloadable report")


class AnalysisResponse(BaseModel):
    success: bool
    analysis_id: str
    timestamp: str
    file_info: Dict[str, Any]
    detection_result: Dict[str, Any]
    integrity: Dict[str, Any]
    performance: Dict[str, Any]
    overall_risk: Dict[str, Any]


class BatchAnalysisResponse(BaseModel):
    batch_id: str
    total_files: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: Dict[str, bool]