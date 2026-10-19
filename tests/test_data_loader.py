"""
Unit tests for data loader, label normalization, and split validation.
"""

import os
import csv
import json
import tempfile
import unittest
from pathlib import Path
from src.data_loader import (
    normalize_label,
    load_class_manifest,
    resolve_disease_label,
    validate_and_filter_records
)

class TestDataLoader(unittest.TestCase):
    def test_normalize_label(self):
        self.assertEqual(normalize_label("Tomato_Leaf_Curl_Virus"), "tomato leaf curl virus")
        self.assertEqual(normalize_label("  Early   Blight  "), "early blight")
        self.assertEqual(normalize_label("Late_Blight"), "late blight")

    def test_resolve_disease_label(self):
        canonical = ["Tomato Leaf Curl Virus", "Early Blight", "Late Blight"]
        
        row1 = {"disease": "early_blight", "canonical_disease": "", "raw_label": ""}
        matched1, err1 = resolve_disease_label(row1, canonical)
        self.assertEqual(matched1, "Early Blight")
        self.assertIsNone(err1)

        row2 = {"disease": "", "canonical_disease": "Late Blight", "raw_label": ""}
        matched2, err2 = resolve_disease_label(row2, canonical)
        self.assertEqual(matched2, "Late Blight")
        self.assertIsNone(err2)

        row3 = {"disease": "unknown_fungus", "canonical_disease": "", "raw_label": ""}
        matched3, err3 = resolve_disease_label(row3, canonical)
        self.assertIsNone(matched3)
        self.assertIsNotNone(err3)

    def test_manifest_loader(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump({
                "Tomato": {"Late Blight": 250, "Early Blight": 250},
                "Rice": ["Blast", "Brown Spot"]
            }, f)
            manifest_path = f.name

        try:
            tomato_classes = load_class_manifest(manifest_path, "Tomato")
            self.assertEqual(sorted(tomato_classes), ["Early Blight", "Late Blight"])

            rice_classes = load_class_manifest(manifest_path, "Rice")
            self.assertEqual(sorted(rice_classes), ["Blast", "Brown Spot"])
        finally:
            os.remove(manifest_path)

if __name__ == "__main__":
    unittest.main()
