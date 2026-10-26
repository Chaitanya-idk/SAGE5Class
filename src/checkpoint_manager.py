"""
Checkpoint management, state serialization, retention pruning, and interruption handling.
"""

import re
import os
import shutil
import json
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None
from src.utils import logger

def parse_checkpoint_ref(ref: str, default_repo: str = "IDK-Chaitanya/SAGE5Class-cp") -> Tuple[str, str]:
    """Extracts repo_id and subfolder from a checkpoint reference, URL, or path."""
    ref = str(ref).strip()
    if ref.startswith("http://") or ref.startswith("https://"):
        m = re.search(r"huggingface\.co/([^/]+/[^/]+)/(?:tree|blob)/[^/]+/(.+)", ref)
        if m:
            return m.group(1), m.group(2).rstrip("/")
        m = re.search(r"huggingface\.co/([^/]+/[^/]+)", ref)
        if m:
            return m.group(1), ""
    if ":" in ref and not Path(ref).exists():
        parts = ref.split(":", 1)
        return parts[0], parts[1].rstrip("/")
    if ref.startswith(f"{default_repo}/"):
        return default_repo, ref[len(default_repo) + 1:].rstrip("/")
    return default_repo, ref.rstrip("/")

def resolve_checkpoint_path(
    checkpoint_ref: Optional[str],
    crop: Optional[str] = None,
    default_repo: str = "IDK-Chaitanya/SAGE5Class-cp",
    cache_dir: str = "/kaggle/working/downloaded_checkpoints",
    token: Optional[str] = None
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Resolves checkpoint path locally or downloads selectively from Hugging Face Hub.
    Returns (local_checkpoint_path, hf_source_path).
    """
    if not checkpoint_ref or str(checkpoint_ref).lower() in ["none", "false"]:
        return None, None

    # Handle "auto" or True by choosing default crop step-600 checkpoint
    if str(checkpoint_ref).lower() in ["auto", "true"]:
        if not crop:
            return None, None
        checkpoint_ref = f"{crop.lower()}/checkpoints/checkpoint-step-600"

    # Check if checkpoint_ref is already an existing local directory with adapter
    local_candidate = Path(checkpoint_ref)
    if local_candidate.exists() and (local_candidate / "adapter_config.json").exists():
        return local_candidate, None

    # Parse HF repo and subfolder
    repo_id, subfolder = parse_checkpoint_ref(checkpoint_ref, default_repo=default_repo)
    hf_source = f"{repo_id}/{subfolder}"

    # Check if already present in cache_dir
    cached_path = Path(cache_dir) / subfolder
    if cached_path.exists() and (cached_path / "adapter_config.json").exists():
        logger.info(f"Using already downloaded checkpoint at: {cached_path}")
        return cached_path, hf_source

    # Download selectively from Hugging Face
    token = token or os.environ.get("HF_TOKEN")
    logger.info(f"Downloading checkpoint '{subfolder}' from Hugging Face repository '{repo_id}'...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subfolder}/*"],
            token=token,
            local_dir=cache_dir,
            resume_download=True
        )
        if cached_path.exists():
            logger.info(f"Checkpoint successfully resolved at: {cached_path}")
            return cached_path, hf_source
    except Exception as e:
        logger.error(f"Error downloading checkpoint from {hf_source}: {e}")

    # Fallback to local_candidate if it exists
    if local_candidate.exists():
        return local_candidate, hf_source

    return None, hf_source

def get_checkpoint_state(checkpoint_path: Optional[Path]) -> Dict[str, Any]:
    """Reads trainer_state.json metadata from a checkpoint directory if present."""
    if not checkpoint_path:
        return {}
    state_file = Path(checkpoint_path) / "trainer_state.json"
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read trainer_state.json from {checkpoint_path}: {e}")
    return {}

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

        # Safe RNG state serialization
        cpu_rng = None
        try:
            cpu_rng = torch.get_rng_state().tolist()
        except Exception:
            pass

        cuda_rng = None
        if torch.cuda.is_available():
            try:
                cuda_rng = [s.tolist() if hasattr(s, "tolist") else list(s) for s in torch.cuda.get_rng_state_all()]
            except Exception:
                pass

        state = {
            "step": step,
            "epoch": epoch,
            "rng_state": cpu_rng,
            "cuda_rng_state": cuda_rng,
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
