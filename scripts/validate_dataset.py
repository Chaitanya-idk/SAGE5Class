"""
Script to validate dataset structure, class manifests, splits, and leakage.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import logger
from src.config import load_config
from src.data_loader import (
    download_dataset_metadata_and_crop,
    load_class_manifest,
    validate_and_filter_records
)

def main():
    parser = argparse.ArgumentParser(description="Validate SAGE dataset for chosen crop.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--crop", type=str, default=None, help="Crop name override (e.g. Tomato, Rice)")
    args = parser.parse_args()

    overrides = {}
    if args.crop:
        overrides["project"] = {"crop": args.crop}

    config = load_config(args.config, overrides)
    crop = config["project"]["crop"]

    logger.info(f"Validating dataset for crop: {crop}...")

    local_root = download_dataset_metadata_and_crop(
        repo_id=config["dataset"]["repo_id"],
        crop=crop,
        local_dir=config["dataset"]["local_dir"]
    )

    manifest_path = local_root / config["dataset"]["class_manifest"]
    classes = load_class_manifest(str(manifest_path), crop)
    logger.info(f"Canonical classes ({len(classes)}): {classes}")

    csv_path = local_root / config["dataset"]["metadata_csv"]
    records, report = validate_and_filter_records(
        csv_path=str(csv_path),
        root_dir=local_root,
        crop=crop,
        canonical_classes=classes
    )

    print("\n" + "=" * 50)
    print("DATASET VALIDATION SUMMARY REPORT")
    print("=" * 50)
    print(f"Crop:                     {crop}")
    print(f"Total Rows Inspected:     {report['total_rows_inspected']}")
    print(f"Crop-Specific Rows:       {report['crop_rows']}")
    print(f"Valid Usable Samples:     {report['valid_samples']}")
    print(f"Missing Image Files:      {report['missing_images']}")
    print(f"Corrupted Images:         {report['corrupted_images']}")
    print(f"Label Mismatches:         {report['label_mismatches']}")
    print(f"Split Cross-Leakage:      {report['leakage_count']}")
    print("\nSamples per Split and Class:")
    for split, counts in report["split_counts"].items():
        total_split = sum(counts.values())
        print(f"  [{split.upper()}] Total: {total_split}")
        for c, count in counts.items():
            print(f"    - {c:25s}: {count}")

    if report["missing_splits_per_class"]:
        print("\nWARNINGS:")
        for w in report["missing_splits_per_class"]:
            print(f"  * {w}")
    print("=" * 50 + "\n")

    if report["valid_samples"] == 0:
        logger.error("Validation failed: 0 usable samples found.")
        sys.exit(1)
    else:
        logger.info("Dataset validation passed successfully.")

if __name__ == "__main__":
    main()
