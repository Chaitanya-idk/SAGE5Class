"""
Unit tests for configuration loader and normalization.
"""

import os
import unittest
from pathlib import Path
from src.config import load_config, deep_merge, DEFAULT_CONFIG

class TestConfig(unittest.TestCase):
    def test_default_config_loading(self):
        config = load_config()
        self.assertIsNotNone(config)
        self.assertEqual(config["project"]["crop"], "Tomato")
        self.assertEqual(config["training"]["method"], "qlora")
        self.assertEqual(config["lora"]["r"], 16)

    def test_crop_switch_without_code_change(self):
        # Switching crop to Rice
        config = load_config(overrides={"project": {"crop": "Rice"}})
        self.assertEqual(config["project"]["crop"], "Rice")

    def test_deep_merge(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"d": 99, "e": 100}}
        merged = deep_merge(base, override)
        self.assertEqual(merged["a"], 1)
        self.assertEqual(merged["b"]["c"], 2)
        self.assertEqual(merged["b"]["d"], 99)
        self.assertEqual(merged["b"]["e"], 100)

    def test_env_override_model_path(self):
        os.environ["QWEN_MODEL_PATH"] = "/test/custom/model/path"
        config = load_config()
        self.assertEqual(config["model"]["model_path"], "/test/custom/model/path")
        del os.environ["QWEN_MODEL_PATH"]

if __name__ == "__main__":
    unittest.main()
