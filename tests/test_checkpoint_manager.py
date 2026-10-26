"""
Unit tests for CheckpointManager and checkpoint resolution helpers.
"""

import os
import json
import tempfile
import unittest
from pathlib import Path
from src.checkpoint_manager import (
    parse_checkpoint_ref,
    resolve_checkpoint_path,
    get_checkpoint_state,
    CheckpointManager
)

class TestCheckpointManager(unittest.TestCase):
    def test_parse_checkpoint_ref_url(self):
        url = "https://huggingface.co/IDK-Chaitanya/SAGE5Class-cp/tree/main/corn/checkpoints/checkpoint-step-600"
        repo, subfolder = parse_checkpoint_ref(url)
        self.assertEqual(repo, "IDK-Chaitanya/SAGE5Class-cp")
        self.assertEqual(subfolder, "corn/checkpoints/checkpoint-step-600")

    def test_parse_checkpoint_ref_relative(self):
        ref = "soybean/checkpoints/checkpoint-step-600"
        repo, subfolder = parse_checkpoint_ref(ref)
        self.assertEqual(repo, "IDK-Chaitanya/SAGE5Class-cp")
        self.assertEqual(subfolder, "soybean/checkpoints/checkpoint-step-600")

    def test_get_checkpoint_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            state_data = {"step": 600, "epoch": 3}
            with open(tmp_path / "trainer_state.json", "w") as f:
                json.dump(state_data, f)
            
            loaded_state = get_checkpoint_state(tmp_path)
            self.assertEqual(loaded_state.get("step"), 600)
            self.assertEqual(loaded_state.get("epoch"), 3)

    def test_resolve_checkpoint_path_local_existing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "adapter_config.json").touch()
            resolved, source = resolve_checkpoint_path(str(tmp_path))
            self.assertEqual(resolved, tmp_path)
            self.assertIsNone(source)

if __name__ == "__main__":
    unittest.main()
