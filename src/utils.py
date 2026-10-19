"""
Utility functions for hardware detection, logging, seeding, and Git tracking.
"""

import os
import sys
import random
import logging
import subprocess
from typing import Dict, Any, Optional

def setup_logger(name: str = "sage_trainer", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()

def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def detect_hardware() -> Dict[str, Any]:
    """
    Detects hardware capabilities including CUDA devices, VRAM, BF16 support,
    and library versions.
    """
    hw_info = {
        "cuda_available": False,
        "device_count": 0,
        "device_name": "CPU",
        "total_memory_gb": 0.0,
        "bf16_supported": False,
        "fp16_supported": False,
        "torch_version": None,
        "transformers_version": None,
        "peft_version": None,
        "bitsandbytes_version": None,
    }

    try:
        import torch
        hw_info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            hw_info["cuda_available"] = True
            hw_info["device_count"] = torch.cuda.device_count()
            hw_info["device_name"] = torch.cuda.get_device_name(0)
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            hw_info["total_memory_gb"] = round(total_bytes / (1024 ** 3), 2)
            hw_info["fp16_supported"] = True
            if hasattr(torch.cuda, "is_bf16_supported"):
                hw_info["bf16_supported"] = torch.cuda.is_bf16_supported()
            else:
                hw_info["bf16_supported"] = False
    except ImportError:
        pass

    for pkg in ["transformers", "peft", "bitsandbytes"]:
        try:
            mod = __import__(pkg)
            hw_info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            hw_info[f"{pkg}_version"] = None

    return hw_info

def get_git_info(cwd: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves current Git commit hash and branch, or placeholder if unavailable."""
    info = {"commit_hash": "unversioned", "branch": "unknown", "is_dirty": False}
    try:
        res_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        info["commit_hash"] = res_hash.stdout.strip()
        
        res_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        info["branch"] = res_branch.stdout.strip()

        res_diff = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        info["is_dirty"] = bool(res_diff.stdout.strip())
    except Exception:
        pass
    return info

def get_gpu_memory_usage() -> str:
    """Returns a string describing current GPU VRAM allocated and reserved."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f"{allocated:.2f}GB / {total:.2f}GB (reserved: {reserved:.2f}GB)"
    except Exception:
        pass
    return "N/A"
