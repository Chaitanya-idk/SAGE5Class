"""
Reporting and artifact logging: generates CSVs, JSON summaries, and Model Card.
"""

import os
import csv
import json
import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from src.utils import logger, get_git_info, detect_hardware
from src.config import save_yaml

class RunReporter:
    """Creates a unique run directory and generates comprehensive metrics artifacts."""
    def __init__(self, root_dir: str, crop: str):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(root_dir) / f"{crop}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.crop = crop
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.adapter_dir = self.run_dir / "adapter"
        self.best_adapter_dir = self.run_dir / "best_adapter"
        self.final_adapter_dir = self.run_dir / "final_adapter"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.best_adapter_dir.mkdir(parents=True, exist_ok=True)
        self.final_adapter_dir.mkdir(parents=True, exist_ok=True)

        self.batch_loss_file = self.run_dir / "batch_loss.csv"
        self.epoch_metrics_file = self.run_dir / "epoch_metrics.csv"
        self._init_csv_files()
        logger.info(f"Initialized run directory: {self.run_dir}")

    def _init_csv_files(self) -> None:
        with open(self.batch_loss_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "batch_loss", "running_mean_loss", "learning_rate", "gpu_memory"])

        with open(self.epoch_metrics_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_mean_loss", "val_loss", "val_accuracy", "val_f1_macro", "duration_sec"])

    def log_batch(
        self,
        step: int,
        epoch: int,
        batch_loss: float,
        running_mean_loss: float,
        learning_rate: float,
        gpu_memory: str = "N/A"
    ) -> None:
        with open(self.batch_loss_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([step, epoch, f"{batch_loss:.5f}", f"{running_mean_loss:.5f}", f"{learning_rate:.8f}", gpu_memory])

    def log_epoch(
        self,
        epoch: int,
        train_mean_loss: float,
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        val_f1_macro: Optional[float] = None,
        duration_sec: float = 0.0
    ) -> None:
        with open(self.epoch_metrics_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            val_loss_str = f"{val_loss:.5f}" if val_loss is not None else "N/A"
            val_acc_str = f"{val_accuracy:.4f}" if val_accuracy is not None else "N/A"
            val_f1_str = f"{val_f1_macro:.4f}" if val_f1_macro is not None else "N/A"
            writer.writerow([epoch, f"{train_mean_loss:.5f}", val_loss_str, val_acc_str, val_f1_str, f"{duration_sec:.1f}"])

    def save_predictions_csv(self, predictions: List[Dict[str, Any]]) -> None:
        """Saves individual sample predictions, ground truth, and latencies."""
        pred_path = self.run_dir / "predictions.csv"
        if not predictions:
            return
        keys = ["image_path", "ground_truth", "predicted", "is_correct", "latency_sec", "split"]
        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for p in predictions:
                writer.writerow({k: p.get(k, "") for k in keys})

    def save_classification_report_csv(self, per_class_metrics: Dict[str, Dict[str, Any]]) -> None:
        """Saves precision, recall, f1, support per class to CSV."""
        path = self.run_dir / "classification_report.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class_name", "precision", "recall", "f1_score", "support"])
            for c, m in per_class_metrics.items():
                writer.writerow([c, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}", m["support"]])

    def save_confusion_matrix_csv(self, cm: List[List[int]], classes: List[str]) -> None:
        """Saves raw confusion matrix to CSV."""
        path = self.run_dir / "confusion_matrix.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([""] + classes)
            for i, row in enumerate(cm):
                writer.writerow([classes[i]] + row)

    def save_metadata_and_summary(
        self,
        config: Dict[str, Any],
        dataset_validation: Dict[str, Any],
        test_metrics: Optional[Dict[str, Any]] = None,
        duration_total_sec: float = 0.0
    ) -> None:
        """Saves run_summary.json, config_used.yaml, git_info.json, and model_card.md."""
        # 1. config_used.yaml
        save_yaml(config, str(self.run_dir / "config_used.yaml"))

        # 2. dataset_validation.json
        with open(self.run_dir / "dataset_validation.json", "w", encoding="utf-8") as f:
            json.dump(dataset_validation, f, indent=2)

        # 3. class_mapping.json
        classes = dataset_validation.get("classes", [])
        mapping = {c: i for i, c in enumerate(classes)}
        with open(self.run_dir / "class_mapping.json", "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)

        # 4. git_info.json
        git_info = get_git_info()
        with open(self.run_dir / "git_info.json", "w", encoding="utf-8") as f:
            json.dump(git_info, f, indent=2)

        # 5. run_summary.json
        hw = detect_hardware()
        summary = {
            "crop": self.crop,
            "classes": classes,
            "classes_count": len(classes),
            "total_execution_seconds": duration_total_sec,
            "hardware": hw,
            "git": git_info,
            "test_metrics": test_metrics or {},
            "dataset_counts": dataset_validation.get("split_counts", {})
        }
        with open(self.run_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # 6. model_card.md
        self._generate_model_card(summary, test_metrics)

    def _generate_model_card(self, summary: Dict[str, Any], metrics: Optional[Dict[str, Any]]) -> None:
        mc_path = self.run_dir / "model_card.md"
        acc = f"{metrics.get('accuracy', 0.0):.4f}" if metrics else "N/A"
        f1 = f"{metrics.get('f1_macro', 0.0):.4f}" if metrics else "N/A"
        content = f"""# SAGE Model Card: Qwen2.5-VL {self.crop} Disease Classifier

## Model Overview
- **Base Model:** Qwen/Qwen2.5-VL-3B-Instruct
- **Fine-Tuning Method:** QLoRA (4-bit NF4 Quantization)
- **Target Crop:** {self.crop}
- **Classes Count:** {summary['classes_count']}
- **Task:** Multimodal Supervised Visual Disease Classification
- **Primary Metric (Test Accuracy):** {acc}
- **Macro F1 Score:** {f1}

## Classes
{json.dumps(summary['classes'], indent=2)}

## Hardware & Environment
- **Device:** {summary['hardware']['device_name']} ({summary['hardware']['total_memory_gb']} GB)
- **PyTorch:** {summary['hardware']['torch_version']}
- **Transformers:** {summary['hardware']['transformers_version']}
- **PEFT:** {summary['hardware']['peft_version']}
- **Git Commit:** `{summary['git']['commit_hash']}`

## Intended Use
Autonomous Visual AI Agent for precision agriculture plant pathology detection.
"""
        with open(mc_path, "w", encoding="utf-8") as f:
            f.write(content)
