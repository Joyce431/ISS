"""File integrity verification using SHA-256 hashing"""
import hashlib
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def compute_sha256(file_path: str, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_integrity(file_path: str, original_hash: str = None) -> Dict[str, Any]:
    """
    Verify file integrity by computing and comparing hashes.
    
    Args:
        file_path: Path to the file
        original_hash: Optional original hash to compare against
        
    Returns:
        Dictionary with hash, size, and integrity status
    """
    if not os.path.exists(file_path):
        return {"error": "File not found", "integrity_verified": False}
    
    current_hash = compute_sha256(file_path)
    file_size = os.path.getsize(file_path)
    
    result = {
        "file_path": file_path,
        "sha256": current_hash,
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 4),
        "integrity_verified": True
    }
    
    if original_hash:
        result["original_hash"] = original_hash
        result["hash_match"] = current_hash == original_hash
        result["integrity_verified"] = current_hash == original_hash
        if not result["hash_match"]:
            result["warning"] = "File has been modified since original hash was computed"
    
    return result


def verify_file_unchanged(file_path: str, before_hash: str) -> Dict[str, Any]:
    """Verify file hasn't been corrupted after analysis"""
    after_hash = compute_sha256(file_path)
    after_size = os.path.getsize(file_path)
    
    return {
        "before_hash": before_hash,
        "after_hash": after_hash,
        "hash_unchanged": before_hash == after_hash,
        "file_size_bytes": after_size,
        "integrity_preserved": before_hash == after_hash,
        "message": "File integrity preserved" if before_hash == after_hash 
                   else "WARNING: File may have been modified during analysis"
    }