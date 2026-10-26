"""
Command-line execution entry point for SAGE Qwen training pipeline.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import logger
from src.train import run_training_pipeline

def main():
    parser = argparse.ArgumentParser(description="Train SAGE Qwen Vision Model.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--crop", type=str, default=None, help="Target crop (e.g. Tomato, Soybean, Rice, Wheat, Corn)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device train batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output root directory")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap training samples for budget testing")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path or HuggingFace checkpoint to resume from")
    args = parser.parse_args()

    overrides = {}
    if args.crop:
        overrides.setdefault("project", {})["crop"] = args.crop
    if args.epochs:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        overrides.setdefault("training", {})["per_device_train_batch_size"] = args.batch_size
    if args.lr:
        overrides.setdefault("training", {})["learning_rate"] = args.lr
    if args.output_dir:
        overrides.setdefault("output", {})["root_dir"] = args.output_dir
    if args.max_train_samples:
        overrides.setdefault("runtime", {})["max_train_samples"] = args.max_train_samples
    if args.resume_from_checkpoint:
        overrides.setdefault("runtime", {})["resume_from_checkpoint"] = args.resume_from_checkpoint

    logger.info("Initializing SAGE training pipeline...")
    run_dir = run_training_pipeline(config_path=args.config, overrides=overrides)
    logger.info(f"Training run completed successfully. Outputs: {run_dir}")

if __name__ == "__main__":
    main()
