"""
Evaluation metrics, robust JSON response parsing, and confusion matrix plotting.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from src.utils import logger
from src.data_loader import normalize_label

def parse_model_response(
    raw_text: str,
    canonical_classes: List[str]
) -> Tuple[Optional[str], bool]:
    """
    Extracts the predicted disease class from model's JSON output.
    Returns (predicted_class_name, is_valid_json).
    """
    clean_text = raw_text.strip()
    
    # Strip markdown code blocks ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
    if match:
        clean_text = match.group(1)
    else:
        # Extract first JSON object
        obj_match = re.search(r"\{[^{}]*\}", clean_text, re.DOTALL)
        if obj_match:
            clean_text = obj_match.group(0)

    try:
        data = json.loads(clean_text)
        disease_candidate = data.get("disease", "")
        if disease_candidate:
            norm_candidate = normalize_label(disease_candidate)
            for c in canonical_classes:
                if normalize_label(c) == norm_candidate:
                    return c, True
            return disease_candidate, False
        return None, False
    except Exception:
        # Fallback exact string match if model output raw text without JSON brackets
        norm_raw = normalize_label(raw_text)
        for c in canonical_classes:
            if normalize_label(c) in norm_raw:
                return c, False
        return None, False

def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[Optional[str]],
    canonical_classes: List[str]
) -> Dict[str, Any]:
    """
    Computes overall accuracy, macro/weighted precision/recall/F1,
    per-class statistics, and confusion matrix.
    """
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report
    )

    # Replace None with "UNKNOWN"
    valid_predictions = [p if p in canonical_classes else "UNKNOWN" for p in y_pred]
    all_eval_classes = canonical_classes + (["UNKNOWN"] if "UNKNOWN" in valid_predictions else [])

    class_to_idx = {c: i for i, c in enumerate(all_eval_classes)}
    true_idx = [class_to_idx[t] for t in y_true]
    pred_idx = [class_to_idx[p] for p in valid_predictions]

    acc = float(accuracy_score(true_idx, pred_idx))
    
    # Precision/Recall/F1 over canonical classes
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        true_idx, pred_idx, labels=list(range(len(canonical_classes))), average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_idx, pred_idx, labels=list(range(len(canonical_classes))), average="weighted", zero_division=0
    )

    # Per-class metrics
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        true_idx, pred_idx, labels=list(range(len(canonical_classes))), average=None, zero_division=0
    )

    per_class_dict = {}
    for i, c in enumerate(canonical_classes):
        per_class_dict[c] = {
            "precision": float(p_per[i]),
            "recall": float(r_per[i]),
            "f1": float(f1_per[i]),
            "support": int(sup_per[i])
        }

    # Confusion matrix
    cm = confusion_matrix(true_idx, pred_idx, labels=list(range(len(canonical_classes))))

    invalid_rate = float(sum(1 for p in y_pred if p is None or p not in canonical_classes) / len(y_pred))

    return {
        "accuracy": acc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "invalid_response_rate": invalid_rate,
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist()
    }

def plot_and_save_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    output_png_path: str,
    title: str = "Confusion Matrix"
) -> None:
    """Renders and saves confusion matrix heatmap using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm_arr = np.array(cm)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_arr,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(title, fontsize=14, pad=12)
        plt.ylabel("True Class", fontsize=12)
        plt.xlabel("Predicted Class", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_png_path, dpi=300)
        plt.close()
        logger.info(f"Saved confusion matrix plot: {output_png_path}")
    except Exception as e:
        logger.warning(f"Could not generate confusion matrix plot: {e}")
