"""
Image utilities for loading, encoding, and processing images.
"""
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image


def load_image(path: Path | str) -> Image.Image:
    """Load an image from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def get_image_size(path: Path | str) -> tuple[int, int]:
    """Get image dimensions (width, height) without loading full image."""
    path = Path(path)
    with Image.open(path) as img:
        return img.size


def image_to_base64(
    image: Image.Image | Path | str,
    format: str = "JPEG",
    quality: int = 85,
    max_size: Optional[tuple[int, int]] = None,
) -> str:
    """
    Convert image to base64 string for API calls.
    
    Args:
        image: PIL Image or path to image
        format: Output format (JPEG, PNG)
        quality: JPEG quality (1-100)
        max_size: Optional (max_width, max_height) to resize
    
    Returns:
        Base64 encoded string
    """
    if isinstance(image, (str, Path)):
        image = load_image(image)

    # Resize if needed
    if max_size:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Convert to bytes
    buffer = BytesIO()
    image.save(buffer, format=format, quality=quality)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


def image_to_data_uri(
    image: Image.Image | Path | str,
    format: str = "JPEG",
    quality: int = 85,
    max_size: Optional[tuple[int, int]] = None,
) -> str:
    """
    Convert image to data URI for API calls.
    
    Returns:
        Data URI string like "data:image/jpeg;base64,..."
    """
    b64 = image_to_base64(image, format, quality, max_size)
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{b64}"


def resize_image_for_api(
    image: Image.Image | Path | str,
    max_dimension: int = 1024,
) -> Image.Image:
    """
    Resize image to fit within max_dimension while preserving aspect ratio.
    Useful for reducing API costs with vision models.
    """
    if isinstance(image, (str, Path)):
        image = load_image(image)

    width, height = image.size
    if max(width, height) <= max_dimension:
        return image

    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def normalize_bbox(
    bbox: list[float],
    image_width: int,
    image_height: int,
    input_normalized: bool = True,
) -> list[int]:
    """
    Convert normalized [0-1] bbox to pixel coordinates, or validate pixel bbox.
    
    Args:
        bbox: [x1, y1, x2, y2] coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
        input_normalized: If True, input is [0-1] normalized; if False, already pixels
    
    Returns:
        Pixel coordinates [x1, y1, x2, y2] clamped to image bounds
    """
    if input_normalized:
        x1 = int(bbox[0] * image_width)
        y1 = int(bbox[1] * image_height)
        x2 = int(bbox[2] * image_width)
        y2 = int(bbox[3] * image_height)
    else:
        x1, y1, x2, y2 = map(int, bbox)

    # Clamp to image bounds
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    # Ensure x1 < x2 and y1 < y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]


def is_normalized_bbox(bbox: list[float]) -> bool:
    """Check if bbox coordinates are in [0, 1] range (normalized)."""
    return all(0.0 <= v <= 1.0 for v in bbox)
