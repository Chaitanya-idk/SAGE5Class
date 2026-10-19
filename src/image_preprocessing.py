"""
Image loading, decoding, aspect-ratio preserving resizing, and validation.
"""

from typing import Optional, Tuple
from pathlib import Path
from PIL import Image, ImageOps
from src.utils import logger

def load_and_preprocess_image(
    image_path: str,
    min_side: int = 224,
    max_side: int = 448,
    max_pixels: int = 200704,
    preserve_aspect_ratio: bool = True,
    target_format: str = "RGB"
) -> Optional[Image.Image]:
    """
    Safely opens an image file, applies EXIF orientation, converts to target format (RGB),
    and resizes while strictly preserving the aspect ratio and respecting pixel budgets.
    
    Returns None if the image file is corrupted or unreadable.
    """
    path = Path(image_path)
    if not path.is_file():
        logger.warning(f"Image file does not exist: {image_path}")
        return None

    try:
        with Image.open(path) as img:
            # Apply EXIF rotation if present
            img = ImageOps.exif_transpose(img)

            # Ensure desired color format
            if img.mode != target_format:
                img = img.convert(target_format)

            width, height = img.size
            if width <= 0 or height <= 0:
                logger.warning(f"Invalid dimensions {img.size} for image: {image_path}")
                return None

            # Aspect-ratio preserving constraint
            if preserve_aspect_ratio:
                # 1. Constrain max_pixels
                current_pixels = width * height
                if current_pixels > max_pixels:
                    scale = (max_pixels / current_pixels) ** 0.5
                    width = int(width * scale)
                    height = int(height * scale)

                # 2. Constrain max_side
                longest = max(width, height)
                if longest > max_side:
                    scale = max_side / longest
                    width = int(width * scale)
                    height = int(height * scale)

                # 3. Constrain min_side (only if both sides would remain within bounds)
                shortest = min(width, height)
                if shortest < min_side:
                    scale = min_side / shortest
                    if max(int(width * scale), int(height * scale)) <= max_side and (int(width * scale) * int(height * scale)) <= max_pixels:
                        width = int(width * scale)
                        height = int(height * scale)

                # Round to multiples of 28 for vision transformer patches (Qwen2.5-VL uses 14x14 or 28x28 patches)
                width = max(28, (width // 28) * 28)
                height = max(28, (height // 28) * 28)

                if (width, height) != img.size:
                    img = img.resize((width, height), resample=Image.Resampling.BICUBIC)
            else:
                img = img.resize((min_side, min_side), resample=Image.Resampling.BICUBIC)

            # Copy image to detach from underlying file descriptor
            return img.copy()

    except Exception as e:
        logger.warning(f"Failed to read/decode image {image_path}: {e}")
        return None

def validate_image(image_path: str) -> Tuple[bool, Optional[str]]:
    """Checks if an image can be opened and verified without crashing."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True, None
    except Exception as e:
        return False, str(e)
