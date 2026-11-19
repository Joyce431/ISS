"""File utility functions for DeepGuard AI"""
import os
import shutil
import mimetypes
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def get_file_extension(file_path: str) -> str:
    """Get lowercase file extension including dot (e.g., '.pdf')"""
    return Path(file_path).suffix.lower()


def get_file_name(file_path: str) -> str:
    """Get file name without path"""
    return Path(file_path).name


def get_file_stem(file_path: str) -> str:
    """Get file name without extension"""
    return Path(file_path).stem


def get_file_size_bytes(file_path: str) -> int:
    """Get file size in bytes"""
    return os.path.getsize(file_path)


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes (rounded to 4 decimals)"""
    return round(os.path.getsize(file_path) / (1024 * 1024), 4)


def get_file_size_human(file_path: str) -> str:
    """Get human-readable file size (e.g., '2.5 MB')"""
    size = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def get_mime_type(file_path: str) -> str:
    """Get MIME type of a file"""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def file_exists(file_path: str) -> bool:
    """Check if file exists and is a file"""
    return os.path.isfile(file_path)


def dir_exists(dir_path: str) -> bool:
    """Check if directory exists"""
    return os.path.isdir(dir_path)


def ensure_directory(dir_path: str) -> str:
    """Create directory if it doesn't exist, return path"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return str(dir_path)


def safe_filename(filename: str, max_length: int = 200) -> str:
    """
    Sanitize filename to prevent path traversal and invalid characters.
    
    Args:
        filename: Original filename
        max_length: Maximum allowed length
        
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '\0']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading dots (hidden files) and spaces
    filename = filename.lstrip('. ')
    
    # Truncate if too long (preserve extension)
    if len(filename) > max_length:
        stem, ext = os.path.splitext(filename)
        filename = stem[:max_length - len(ext)] + ext
    
    # Fallback if empty
    if not filename:
        filename = f"file_{int(datetime.now().timestamp())}"
    
    return filename


def generate_unique_filename(original_name: str, prefix: str = "") -> str:
    """Generate a unique filename using timestamp and UUID"""
    import uuid
    stem = get_file_stem(original_name)
    ext = get_file_extension(original_name)
    unique_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    parts = [p for p in [prefix, timestamp, unique_id, stem] if p]
    return "_".join(parts) + ext


def copy_file_safely(src: str, dst: str) -> str:
    """
    Copy a file safely, ensuring destination directory exists.
    Returns destination path.
    """
    ensure_directory(os.path.dirname(dst))
    shutil.copy2(src, dst)
    logger.info(f"Copied {src} -> {dst}")
    return dst


def move_file_safely(src: str, dst: str) -> str:
    """Move a file safely, ensuring destination directory exists."""
    ensure_directory(os.path.dirname(dst))
    shutil.move(src, dst)
    logger.info(f"Moved {src} -> {dst}")
    return dst


def delete_file_safely(file_path: str) -> bool:
    """Delete a file if it exists, return success status"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to delete {file_path}: {e}")
        return False


def cleanup_directory(dir_path: str, older_than_hours: int = 24) -> int:
    """
    Remove files older than N hours from a directory.
    Returns number of files removed.
    """
    import time
    removed = 0
    cutoff = time.time() - (older_than_hours * 3600)
    
    if not os.path.isdir(dir_path):
        return 0
    
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff:
                os.remove(file_path)
                removed += 1
        except Exception as e:
            logger.error(f"Cleanup failed for {file_path}: {e}")
    
    logger.info(f"Cleaned up {removed} files from {dir_path}")
    return removed


def list_files(directory: str, extensions: Optional[List[str]] = None,
               recursive: bool = False) -> List[str]:
    """
    List files in a directory, optionally filtered by extensions.
    
    Args:
        directory: Directory to search
        extensions: List of extensions to include (e.g., ['.jpg', '.png'])
        recursive: Whether to search subdirectories
        
    Returns:
        List of file paths
    """
    files = []
    
    if not os.path.isdir(directory):
        return files
    
    if recursive:
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if extensions is None or get_file_extension(filename) in extensions:
                    files.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                if extensions is None or get_file_extension(file_path) in extensions:
                    files.append(file_path)
    
    return files


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get comprehensive file information"""
    if not os.path.exists(file_path):
        return {"error": "File not found", "path": file_path}
    
    stat = os.stat(file_path)
    
    return {
        "path": file_path,
        "name": get_file_name(file_path),
        "stem": get_file_stem(file_path),
        "extension": get_file_extension(file_path),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 4),
        "size_human": get_file_size_human(file_path),
        "mime_type": get_mime_type(file_path),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "is_readable": os.access(file_path, os.R_OK)
    }


def is_safe_path(base_dir: str, target_path: str) -> bool:
    """
    Check if target path is safely within base directory.
    Prevents path traversal attacks.
    """
    base = os.path.abspath(base_dir)
    target = os.path.abspath(target_path)
    return target.startswith(base + os.sep) or target == base


def validate_file_extension(file_path: str, allowed: List[str]) -> Tuple[bool, str]:
    """
    Validate file extension against allowlist.
    
    Returns:
        (is_valid, message)
    """
    ext = get_file_extension(file_path)
    if not ext:
        return False, "File has no extension"
    if ext not in [e.lower() for e in allowed]:
        return False, f"Extension '{ext}' not allowed. Allowed: {', '.join(allowed)}"
    return True, "OK"


def read_text_file(file_path: str, encoding: str = "utf-8",
                   max_bytes: Optional[int] = None) -> str:
    """Read a text file safely with encoding fallback"""
    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            if max_bytes:
                return f.read(max_bytes)
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1', errors='replace') as f:
            return f.read(max_bytes) if max_bytes else f.read()


def write_text_file(file_path: str, content: str, encoding: str = "utf-8") -> str:
    """Write text to file, ensuring directory exists"""
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(content)
    return file_path


def get_temp_path(prefix: str = "deepguard_", suffix: str = "") -> str:
    """Generate a temporary file path"""
    import tempfile
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return path