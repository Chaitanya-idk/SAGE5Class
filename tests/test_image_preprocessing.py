"""
Unit tests for image loading, aspect ratio preservation, and preprocessing.
"""

import os
import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from src.image_preprocessing import load_and_preprocess_image, validate_image

class TestImagePreprocessing(unittest.TestCase):
    def setUp(self):
        if not PIL_AVAILABLE:
            self.skipTest("PIL / Pillow not installed in local environment")

    def test_aspect_ratio_preservation(self):
        with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".jpg") as f:
            # Create a 600x300 image (2:1 aspect ratio)
            img = Image.new("RGB", (600, 300), color=(100, 200, 50))
            img.save(f, format="JPEG")
            img_path = f.name

        try:
            processed = load_and_preprocess_image(
                image_path=img_path,
                min_side=224,
                max_side=448,
                max_pixels=200704,
                preserve_aspect_ratio=True
            )
            self.assertIsNotNone(processed)
            w, h = processed.size
            self.assertLessEqual(max(w, h), 448)
            self.assertLessEqual(w * h, 200704)
            # Aspect ratio should still be close to 2:1
            ratio = w / h
            self.assertAlmostEqual(ratio, 2.0, delta=0.2)
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)

    def test_missing_and_corrupt_images(self):
        # Non-existent file
        res = load_and_preprocess_image("non_existent_file_12345.jpg")
        self.assertIsNone(res)

        # Corrupt file
        with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".jpg") as f:
            f.write(b"this is corrupt not a valid jpg header")
            corrupt_path = f.name

        try:
            res_corrupt = load_and_preprocess_image(corrupt_path)
            self.assertIsNone(res_corrupt)
            valid, err = validate_image(corrupt_path)
            self.assertFalse(valid)
        finally:
            if os.path.exists(corrupt_path):
                os.remove(corrupt_path)

if __name__ == "__main__":
    unittest.main()
