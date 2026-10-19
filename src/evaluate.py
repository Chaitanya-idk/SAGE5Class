"""
Evaluation pipeline for running generative inference, measuring latencies,
and computing final classification metrics.
"""

import time
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import logger
from src.metrics import (
    parse_model_response,
    compute_classification_metrics,
    plot_and_save_confusion_matrix
)
from src.reporting import RunReporter

def evaluate_model(
    model: Any,
    processor: Any,
    dataset: Any,
    canonical_classes: List[str],
    reporter: Optional[RunReporter] = None,
    split_name: str = "test",
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes greedy generation on the dataset, parses JSON responses,
    measures per-image latency, and outputs classification metrics.
    """
    model.eval()
    y_true: List[str] = []
    y_pred: List[Optional[str]] = []
    predictions_log: List[Dict[str, Any]] = []

    num_samples = len(dataset)
    if max_samples:
        num_samples = min(num_samples, max_samples)

    logger.info(f"Starting evaluation on {split_name} split ({num_samples} samples)...")

    latencies: List[float] = []

    for i in tqdm(range(num_samples), desc=f"Evaluating {split_name}"):
        item = dataset[i]
        image = item["image"]
        ground_truth = item["ground_truth_disease"]
        image_path = item["image_path"]

        if image is None:
            # Corrupted image, record as invalid prediction
            y_true.append(ground_truth)
            y_pred.append(None)
            predictions_log.append({
                "image_path": image_path,
                "ground_truth": ground_truth,
                "predicted": "CORRUPTED_IMAGE",
                "is_correct": False,
                "latency_sec": 0.0,
                "split": split_name
            })
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": item["prompt"]}
                ]
            }
        ]

        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        # Move to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False
            )
        latency = time.perf_counter() - t0
        latencies.append(latency)

        # Trim input prompt tokens from generated output
        input_len = inputs["input_ids"].shape[1]
        output_tokens = generated_ids[0][input_len:]
        output_text = processor.decode(output_tokens, skip_special_tokens=True)

        predicted_class, is_valid_json = parse_model_response(output_text, canonical_classes)

        y_true.append(ground_truth)
        y_pred.append(predicted_class)
        is_correct = (predicted_class == ground_truth)

        predictions_log.append({
            "image_path": image_path,
            "ground_truth": ground_truth,
            "predicted": predicted_class or "UNKNOWN",
            "is_correct": is_correct,
            "latency_sec": round(latency, 4),
            "split": split_name
        })

    metrics = compute_classification_metrics(y_true, y_pred, canonical_classes)
    metrics["mean_latency_sec"] = float(sum(latencies) / len(latencies)) if latencies else 0.0

    logger.info(
        f"Evaluation finished on {split_name}: "
        f"Accuracy={metrics['accuracy']:.4f}, "
        f"F1-Macro={metrics['f1_macro']:.4f}, "
        f"InvalidRate={metrics['invalid_response_rate']:.2%}"
    )

    if reporter:
        reporter.save_predictions_csv(predictions_log)
        reporter.save_classification_report_csv(metrics["per_class"])
        reporter.save_confusion_matrix_csv(metrics["confusion_matrix"], canonical_classes)
        plot_and_save_confusion_matrix(
            cm=metrics["confusion_matrix"],
            class_names=canonical_classes,
            output_png_path=str(reporter.run_dir / f"confusion_matrix_{split_name}.png"),
            title=f"Confusion Matrix ({reporter.crop} - {split_name.capitalize()})"
        )

    return metrics
