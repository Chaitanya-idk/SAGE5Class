"""
Dataset access, selective downloading from Hugging Face, metadata parsing,
label validation, leakage detection, and PyTorch Dataset creation.
"""

import os
import csv
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    class Dataset:
        pass
from src.utils import logger
from src.image_preprocessing import load_and_preprocess_image

def download_dataset_metadata_and_crop(
    repo_id: str = "IDK-Chaitanya/SAGE5class",
    repo_type: str = "dataset",
    crop: str = "Tomato",
    local_dir: Optional[str] = None,
    revision: Optional[str] = None,
    token_env: str = "HF_TOKEN"
) -> Path:
    """
    Selectively downloads only metadata files and images for the specified crop
    from Hugging Face Hub, avoiding downloading images for other crops.
    """
    token = os.environ.get(token_env, None)
    target_dir = Path(local_dir) if local_dir else Path("./data/SAGE5class")
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        logger.info(f"Downloading metadata and images for crop '{crop}' from {repo_id}...")
        
        # Download metadata first
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
            local_dir=str(target_dir),
            allow_patterns=["metadata/*", ".gitattributes"],
            resume_download=True
        )

        # Download crop-specific images
        patterns = [
            f"images/{crop}/*",
            f"images/{crop.capitalize()}/*",
            f"images/{crop.lower()}/*",
            f"split_images/*/{crop}/*",
            f"split_images/*/{crop.capitalize()}/*",
            f"split_images/*/{crop.lower()}/*"
        ]
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
            local_dir=str(target_dir),
            allow_patterns=patterns,
            resume_download=True
        )
        logger.info(f"Selective download complete in {target_dir}")
    except ImportError:
        logger.warning("huggingface_hub not installed. Assuming dataset is already present locally.")
    except Exception as e:
        logger.error(f"Error during Hugging Face selective download: {e}")
        # Proceed assuming files might already exist locally
    
    return target_dir

def load_class_manifest(manifest_path: str, crop: str) -> List[str]:
    """Loads canonical class names for the chosen crop from selected_classes.json."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Class manifest file not found: {manifest_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Match crop case-insensitively
    crop_classes = None
    for k, v in data.items():
        if k.lower() == crop.lower():
            if isinstance(v, dict):
                crop_classes = list(v.keys())
            elif isinstance(v, list):
                crop_classes = v
            break

    if not crop_classes:
        raise ValueError(f"Crop '{crop}' not found in class manifest {manifest_path}. Available: {list(data.keys())}")

    return sorted(crop_classes)

def normalize_label(label: str) -> str:
    """Normalizes label string by removing underscores, trimming whitespace, and standardizing case."""
    return " ".join(label.replace("_", " ").split()).strip().lower()

def resolve_disease_label(
    row: Dict[str, str],
    canonical_classes: List[str]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolves disease label from row using disease, canonical_disease, or raw_label.
    Returns (matched_canonical_class, error_reason).
    """
    norm_map = {normalize_label(c): c for c in canonical_classes}

    candidates = [
        row.get("disease", ""),
        row.get("canonical_disease", ""),
        row.get("raw_label", "").split("/")[-1] if row.get("raw_label") else ""
    ]

    for cand in candidates:
        if cand:
            norm = normalize_label(cand)
            if norm in norm_map:
                return norm_map[norm], None

    return None, f"Unrecognized label candidates: {candidates}"

def validate_and_filter_records(
    csv_path: str,
    root_dir: Path,
    crop: str,
    canonical_classes: List[str],
    validate_image_files: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parses CSV metadata, filters by crop, resolves file paths, checks split validity,
    detects potential data leakage across splits using hashes, and outputs a validation report.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    records: List[Dict[str, Any]] = []
    seen_hashes_by_split: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    leakage_hashes: Set[str] = set()

    report = {
        "crop": crop,
        "classes": canonical_classes,
        "total_rows_inspected": 0,
        "crop_rows": 0,
        "valid_samples": 0,
        "missing_images": 0,
        "corrupted_images": 0,
        "label_mismatches": 0,
        "leakage_count": 0,
        "split_counts": {"train": {}, "val": {}, "test": {}},
        "missing_splits_per_class": []
    }

    for split in report["split_counts"]:
        for c in canonical_classes:
            report["split_counts"][split][c] = 0

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            report["total_rows_inspected"] += 1
            if row.get("crop", "").strip().lower() != crop.lower():
                continue

            report["crop_rows"] += 1
            disease, err = resolve_disease_label(row, canonical_classes)
            if not disease:
                report["label_mismatches"] += 1
                continue

            raw_img_path = row.get("image", "").strip()
            # Resolve relative to root_dir
            resolved_img_path = root_dir / raw_img_path
            if not resolved_img_path.is_file():
                # Try finding alternative in images/crop/... or split_images/...
                alt_path = root_dir / "images" / crop / Path(raw_img_path).name
                if alt_path.is_file():
                    resolved_img_path = alt_path
                else:
                    report["missing_images"] += 1
                    continue

            # Split normalization
            split = row.get("split", "train").strip().lower()
            if split in ["validation", "valid"]:
                split = "val"
            if split not in ["train", "val", "test"]:
                split = "train"

            # Check image hash for leakage
            img_hash = row.get("sha256", "").strip()
            if not img_hash:
                img_hash = hashlib.sha256(raw_img_path.encode()).hexdigest()

            # Detect leakage across train vs val/test
            if split == "train":
                if img_hash in seen_hashes_by_split["val"] or img_hash in seen_hashes_by_split["test"]:
                    leakage_hashes.add(img_hash)
            elif split in ["val", "test"]:
                if img_hash in seen_hashes_by_split["train"]:
                    leakage_hashes.add(img_hash)

            seen_hashes_by_split[split].add(img_hash)

            record = {
                "image_path": str(resolved_img_path),
                "crop": crop,
                "disease": disease,
                "split": split,
                "sha256": img_hash,
                "metadata": row
            }
            records.append(record)
            report["split_counts"][split][disease] = report["split_counts"][split].get(disease, 0) + 1

    report["leakage_count"] = len(leakage_hashes)
    report["valid_samples"] = len(records)

    # Check absent classes
    for split, counts in report["split_counts"].items():
        for c in canonical_classes:
            if counts.get(c, 0) == 0:
                report["missing_splits_per_class"].append(f"Split '{split}' missing class '{c}'")

    if report["leakage_count"] > 0:
        logger.warning(f"Detected {report['leakage_count']} images appearing across train and eval splits!")

    return records, report

class SAGEDataset(Dataset):
    """
    PyTorch Dataset for SAGE Vision-Language disease classification.
    Formatted specifically for Qwen2.5-VL chat prompt and JSON response.
    """
    def __init__(
        self,
        records: List[Dict[str, Any]],
        canonical_classes: List[str],
        crop: str = "Tomato",
        min_image_side: int = 224,
        max_image_side: int = 448,
        max_pixels: int = 200704
    ):
        self.records = records
        self.canonical_classes = canonical_classes
        self.crop = crop
        self.min_image_side = min_image_side
        self.max_image_side = max_image_side
        self.max_pixels = max_pixels

        classes_list_str = ", ".join(f'"{c}"' for c in self.canonical_classes)
        self.instruction_prompt = (
            f"Identify the disease class visible in this {self.crop.lower()} leaf image. "
            f"Select exactly one class from the following list: [{classes_list_str}]. "
            f"Return the answer in JSON format with 'crop' and 'disease' keys."
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        image_path = item["image_path"]

        image = load_and_preprocess_image(
            image_path=image_path,
            min_side=self.min_image_side,
            max_side=self.max_image_side,
            max_pixels=self.max_pixels
        )

        target_json = json.dumps({
            "crop": item["crop"],
            "disease": item["disease"]
        })

        return {
            "image": image,
            "image_path": image_path,
            "prompt": self.instruction_prompt,
            "target": target_json,
            "ground_truth_disease": item["disease"],
            "crop": item["crop"],
            "split": item["split"]
        }
