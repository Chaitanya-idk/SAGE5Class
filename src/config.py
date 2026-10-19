"""
Configuration loader, validator, and normalizer for SAGE-Qwen-Trainer.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from src.utils import logger, detect_hardware

DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "SAGE-Qwen-Tomato",
        "seed": 42,
        "crop": "Tomato",
        "task": "disease_classification"
    },
    "dataset": {
        "repo_id": "IDK-Chaitanya/SAGE5class",
        "repo_type": "dataset",
        "local_dir": None,
        "metadata_csv": "metadata/all_images_metadata.csv",
        "metadata_jsonl": "metadata/all_images_metadata.jsonl",
        "class_manifest": "metadata/selected_classes.json",
        "split_manifest": "metadata/split_summary.csv",
        "image_root": "images",
        "split_image_root": "split_images",
        "prefer_existing_splits": True,
        "split_names": {"train": "train", "validation": "val", "test": "test"},
        "hf_revision": None,
        "hf_token_env": "HF_TOKEN",
        "download_missing_files": True,
        "validate_image_paths": True,
        "allow_rebuild_splits": False
    },
    "model": {
        "model_path": None,
        "model_path_env": "QWEN_MODEL_PATH",
        "model_name": "Qwen2.5-VL-3B-Instruct",
        "quantization": "4bit_nf4",
        "trust_remote_code": False,
        "use_gradient_checkpointing": True,
        "attention_implementation": "auto"
    },
    "preprocessing": {
        "min_image_side": 224,
        "max_image_side": 448,
        "max_pixels": 200704,
        "image_format": "RGB",
        "interpolation": "bicubic",
        "cache_processed_images": False,
        "preserve_aspect_ratio": True,
        "max_image_count_per_sample": 1
    },
    "training": {
        "method": "qlora",
        "epochs": 3,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
        "logging_steps": 5,
        "eval_steps": 50,
        "save_steps": 50,
        "save_total_limit": 3,
        "bf16": "auto",
        "fp16": "auto",
        "dataloader_num_workers": 2,
        "gradient_checkpointing": True,
        "group_by_length": False,
        "max_seq_length": 256,
        "seed": 42
    },
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": "auto"
    },
    "runtime": {
        "time_budget_hours": 3.5,
        "estimate_runtime_before_training": True,
        "early_stopping": True,
        "early_stopping_patience": 2,
        "checkpoint_on_interrupt": True,
        "resume_from_checkpoint": "auto",
        "max_train_samples": None,
        "max_eval_samples": None,
        "max_test_samples": None
    },
    "evaluation": {
        "primary_metric": "accuracy",
        "report_per_class_metrics": True,
        "generate_confusion_matrix": True,
        "generate_classification_report": True,
        "top_k_accuracy": [1, 3]
    },
    "output": {
        "root_dir": "/kaggle/working/sage_runs",
        "save_adapter": True,
        "save_tokenizer_processor": True,
        "save_training_logs": True,
        "save_predictions": True,
        "save_metrics_json": True,
        "save_metrics_csv": True,
        "save_model_card": True,
        "save_git_commit": True
    }
}

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges override into base dictionary."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_yaml_raw(path: str) -> Dict[str, Any]:
    """Loads a YAML file using PyYAML or basic parser if PyYAML is unavailable."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        import yaml
        with open(path_obj, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Simple fallback parser for key: value dictionaries
        logger.warning("PyYAML is not installed. Using rudimentary fallback YAML loader.")
        result: Dict[str, Any] = {}
        curr_section: Optional[str] = None
        with open(path_obj, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if line.startswith(" ") or line.startswith("\t"):
                    if ":" in stripped and curr_section:
                        k, v = stripped.split(":", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if v.lower() == "true": v = True
                        elif v.lower() == "false": v = False
                        elif v.lower() == "null": v = None
                        elif v.isdigit(): v = int(v)
                        else:
                            try: v = float(v)
                            except ValueError: pass
                        result[curr_section][k] = v
                elif stripped.endswith(":"):
                    curr_section = stripped[:-1].strip()
                    result[curr_section] = {}
        return result

def save_yaml(data: Dict[str, Any], path: str) -> None:
    """Saves dictionary to YAML or JSON formatted file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        with open(path_obj, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Loads, normalizes, and validates the configuration.
    Applies environment variable overrides and hardware auto-detection.
    """
    config = json.loads(json.dumps(DEFAULT_CONFIG))

    if config_path:
        user_cfg = load_yaml_raw(config_path)
        config = deep_merge(config, user_cfg)

    if overrides:
        config = deep_merge(config, overrides)

    # Environment variable overrides
    model_env_var = config["model"].get("model_path_env", "QWEN_MODEL_PATH")
    if os.environ.get(model_env_var):
        config["model"]["model_path"] = os.environ[model_env_var]

    if os.environ.get("SAGE_CROP"):
        config["project"]["crop"] = os.environ["SAGE_CROP"]

    # Hardware detection & normalization
    hw = detect_hardware()
    if config["training"]["bf16"] == "auto":
        config["training"]["bf16"] = bool(hw["bf16_supported"])
    if config["training"]["fp16"] == "auto":
        config["training"]["fp16"] = bool(not hw["bf16_supported"] and hw["cuda_available"])

    # If CPU only, disable half-precision and quantization flags
    if not hw["cuda_available"]:
        config["training"]["bf16"] = False
        config["training"]["fp16"] = False
        if config["model"]["quantization"] in ["4bit_nf4", "8bit"]:
            logger.warning("CUDA not available. Disabling 4-bit quantization for CPU mode.")
            config["model"]["quantization"] = None

    # Validation
    crop = config["project"]["crop"]
    valid_crops = ["Tomato", "Soybean", "Rice", "Wheat", "Corn"]
    if crop not in valid_crops:
        logger.warning(f"Specified crop '{crop}' not in standard list {valid_crops}. Proceeding assuming custom manifest.")

    return config
