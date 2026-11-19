"""Test file integrity verification"""
import pytest
import tempfile
import os
from app.core.integrity import compute_sha256, verify_integrity, verify_file_unchanged


def test_hash_consistency():
    """Test that same file produces same hash"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
        f.write(b"Test content for hashing")
        path = f.name
    
    try:
        hash1 = compute_sha256(path)
        hash2 = compute_sha256(path)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex chars
    finally:
        os.unlink(path)


def test_integrity_preservation():
    """Test that file integrity is preserved"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
        f.write(b"Original content")
        path = f.name
    
    try:
        original_hash = compute_sha256(path)
        result = verify_file_unchanged(path, original_hash)
        assert result["integrity_preserved"] is True
        assert result["hash_unchanged"] is True
    finally:
        os.unlink(path)


def test_modified_file_detected():
    """Test that file modification is detected"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
        f.write(b"Original content")
        path = f.name
    
    try:
        original_hash = compute_sha256(path)
        
        # Modify file
        with open(path, 'w') as f:
            f.write("Modified content")
        
        result = verify_file_unchanged(path, original_hash)
        assert result["integrity_preserved"] is False
        assert result["hash_unchanged"] is False
    finally:
        os.unlink(path)