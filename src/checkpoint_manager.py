"""
Checkpoint management, state serialization, retention pruning, and interruption handling.
"""

import os
import shutil
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
import torch
from src.utils import logger

class CheckpointManager:
    """Manages saving, pruning, and resuming training checkpoints."""
    def __init__(
        self,
        checkpoint_dir: str,
        save_total_limit: int = 3
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = max(1, save_total_limit)

    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        model: Any,
        processor: Any,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> Path:
        """Saves adapter weights, processor, and training state."""
        ckpt_name = f"checkpoint-step-{step}"
        ckpt_path = self.checkpoint_dir / ckpt_name
        ckpt_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to: {ckpt_path}")

        # 1. Save adapter model
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(ckpt_path))

        # 2. Save processor
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(str(ckpt_path))

        # 3. Save optimizer & scheduler & RNG state
        state = {
            "step": step,
            "epoch": epoch,
            "rng_state": torch.get_rng_state().tolist(),
            "cuda_rng_state": torch.cuda.get_rng_state_all().tolist() if torch.cuda.is_available() else None,
            "extra": extra_state or {}
        }
        if optimizer is not None:
            torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), ckpt_path / "scheduler.pt")

        with open(ckpt_path / "trainer_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        # 4. Prune old checkpoints
        self._prune_checkpoints()

        return ckpt_path

    def _prune_checkpoints(self) -> None:
        """Removes older checkpoints keeping only save_total_limit newest."""
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.glob("checkpoint-step-*") if d.is_dir()],
            key=lambda d: os.path.getmtime(d)
        )
        if len(checkpoints) > self.save_total_limit:
            to_remove = checkpoints[:-self.save_total_limit]
            for c in to_remove:
                logger.info(f"Pruning older checkpoint: {c}")
                shutil.rmtree(c, ignore_errors=True)

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Finds the most recent valid checkpoint in the directory."""
        candidates = sorted(
            [d for d in self.checkpoint_dir.glob("checkpoint-step-*") if d.is_dir()],
            key=lambda d: os.path.getmtime(d),
            reverse=True
        )
        for c in candidates:
            if (c / "adapter_config.json").exists() or (c / "trainer_state.json").exists():
                return c
        return None

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: Any,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Loads state dictionary into optimizer, scheduler, and returns trainer state."""
        state_file = checkpoint_path / "trainer_state.json"
        state = {}
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

        if optimizer is not None and (checkpoint_path / "optimizer.pt").exists():
            optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt"))

        if scheduler is not None and (checkpoint_path / "scheduler.pt").exists():
            scheduler.load_state_dict(torch.load(checkpoint_path / "scheduler.pt"))

        logger.info(f"Resumed checkpoint from {checkpoint_path} (step {state.get('step', 0)}, epoch {state.get('epoch', 0)})")
        return state
