"""
Multimodal Qwen2.5-VL QLoRA training engine with live tqdm progress,
loss masking on assistant tokens, runtime estimation, and interruption recovery.
"""

import os
import sys
import time
import math
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import logger, set_seed, get_gpu_memory_usage
from src.config import load_config
from src.data_loader import (
    download_dataset_metadata_and_crop,
    load_class_manifest,
    validate_and_filter_records,
    SAGEDataset
)
from src.model_loader import load_processor_and_model
from src.checkpoint_manager import CheckpointManager, resolve_checkpoint_path, get_checkpoint_state
from src.reporting import RunReporter
from src.evaluate import evaluate_model

class QwenMultimodalCollator:
    """
    Collate function for Qwen2.5-VL chat messages that masks out prompt and image
    tokens with label -100 so that loss is computed solely on the assistant's JSON answer.
    """
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        images = []
        labels_list = []

        assistant_tag = "<|im_start|>assistant\n"

        for item in batch:
            image = item["image"]
            prompt = item["prompt"]
            target = item["target"]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                },
                {
                    "role": "assistant",
                    "content": target
                }
            ]

            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(full_text)
            images.append(image)

        # Batch encode with processor
        batch_inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )

        input_ids = batch_inputs["input_ids"]
        labels = input_ids.clone()

        # Mask user prompt and pad tokens with -100
        pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        labels[labels == pad_token_id] = -100

        # For each sample, mask tokens before the assistant response
        for i, text in enumerate(texts):
            if assistant_tag in text:
                parts = text.split(assistant_tag)
                prompt_part = parts[0] + assistant_tag
                # Tokenize prompt portion to find prompt token length
                prompt_tokens = self.processor.tokenizer(prompt_part, add_special_tokens=False)["input_ids"]
                prompt_len = len(prompt_tokens)
                labels[i, :prompt_len] = -100
            else:
                # If assistant tag is missing, mask entire prompt
                labels[i, :] = -100

        batch_inputs["labels"] = labels
        return batch_inputs

def estimate_runtime(
    model: Any,
    train_loader: DataLoader,
    epochs: int,
    grad_accum_steps: int,
    time_budget_hours: float
) -> Dict[str, Any]:
    """Runs a 5-step benchmark to estimate total training runtime."""
    device = next(model.parameters()).device
    model.train()
    
    logger.info("Running runtime benchmark (5 steps)...")
    times = []
    
    iterator = iter(train_loader)
    for _ in range(min(5, len(train_loader))):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        t0 = time.perf_counter()
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
        model.zero_grad()
        times.append(time.perf_counter() - t0)

    avg_step_time = sum(times) / len(times) if times else 1.0
    total_batches = len(train_loader) * epochs
    effective_steps = total_batches // grad_accum_steps
    est_train_seconds = avg_step_time * total_batches
    est_total_seconds = est_train_seconds * 1.15  # +15% for evaluation and checkpointing
    est_hours = est_total_seconds / 3600.0

    logger.info(
        f"Benchmark Result: {avg_step_time:.3f}s per batch. "
        f"Estimated training duration: {est_hours:.2f} hours (Budget: {time_budget_hours}h)."
    )

    if est_hours > time_budget_hours:
        logger.warning(
            f"WARNING: Estimated runtime ({est_hours:.2f}h) exceeds configured budget ({time_budget_hours}h)! "
            f"Consider reducing epochs, reducing max_pixels, or setting runtime.max_train_samples."
        )

    return {
        "avg_batch_time_sec": avg_step_time,
        "estimated_hours": est_hours,
        "within_budget": est_hours <= time_budget_hours
    }

def run_training_pipeline(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> str:
    """Main training execution entry point."""
    start_time = time.time()
    config = load_config(config_path, overrides)
    set_seed(config["project"]["seed"])

    crop = config["project"]["crop"]
    reporter = RunReporter(root_dir=config["output"]["root_dir"], crop=crop)

    # 1. Dataset download & validation
    local_dataset_root = download_dataset_metadata_and_crop(
        repo_id=config["dataset"]["repo_id"],
        crop=crop,
        local_dir=config["dataset"]["local_dir"],
        revision=config["dataset"]["hf_revision"]
    )

    manifest_path = local_dataset_root / config["dataset"]["class_manifest"]
    canonical_classes = load_class_manifest(str(manifest_path), crop)
    logger.info(f"Loaded {len(canonical_classes)} canonical classes for {crop}: {canonical_classes}")

    metadata_csv = local_dataset_root / config["dataset"]["metadata_csv"]
    records, val_report = validate_and_filter_records(
        csv_path=str(metadata_csv),
        root_dir=local_dataset_root,
        crop=crop,
        canonical_classes=canonical_classes
    )

    # Stop if dataset is empty or invalid
    if val_report["valid_samples"] == 0:
        raise RuntimeError(f"No valid samples found for crop '{crop}'. Check dataset metadata.")

    # 2. Datasets & DataLoaders
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]
    test_records = [r for r in records if r["split"] == "test"]

    if config["runtime"]["max_train_samples"]:
        train_records = train_records[:config["runtime"]["max_train_samples"]]
    if config["runtime"]["max_eval_samples"]:
        val_records = val_records[:config["runtime"]["max_eval_samples"]]
    if config["runtime"]["max_test_samples"]:
        test_records = test_records[:config["runtime"]["max_test_samples"]]

    logger.info(f"Split sizes: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")

    train_dataset = SAGEDataset(train_records, canonical_classes, crop)
    val_dataset = SAGEDataset(val_records, canonical_classes, crop)
    test_dataset = SAGEDataset(test_records, canonical_classes, crop)

    # 3. Checkpoint resolution & Pre-training summary
    resume_ref = config["runtime"].get("resume_from_checkpoint")
    resolved_ckpt_path, hf_source_path = resolve_checkpoint_path(
        checkpoint_ref=resume_ref,
        crop=crop,
        default_repo=config["dataset"].get("repo_id", "IDK-Chaitanya/SAGE5Class-cp"),
        token=os.environ.get(config["dataset"].get("hf_token_env", "HF_TOKEN"))
    )

    ckpt_state = get_checkpoint_state(resolved_ckpt_path) if resolved_ckpt_path else {}
    prev_epoch = ckpt_state.get("epoch", 0)
    prev_step = ckpt_state.get("step", 0)
    additional_epochs = int(config["training"]["epochs"])
    total_target_epochs = prev_epoch + additional_epochs
    resume_mode = "adapter_weights_reinit_optimizer" if resolved_ckpt_path else "none"

    lr = float(config["training"]["learning_rate"])

    print("\n" + "=" * 65)
    print("SAGE CONTINUED TRAINING CONFIGURATION")
    print("=" * 65)
    print(f"Crop Name:                  {crop}")
    print(f"Hugging Face Checkpoint:    {hf_source_path or 'None (training from scratch)'}")
    print(f"Actual Local Checkpoint:    {resolved_ckpt_path or 'None'}")
    print(f"Resume Mode:                {resume_mode}")
    print(f"Previous Completed Epoch:   {prev_epoch}")
    print(f"Previous Global Step:       {prev_step}")
    print(f"Requested Additional Epochs:{additional_epochs}")
    print(f"Total Target Epochs:        {total_target_epochs}")
    print(f"Effective Learning Rate:    {lr}")
    print(f"Output Directory:           {reporter.run_dir}")
    print("=" * 65 + "\n")

    # 4. Model & Processor Loading
    processor, model = load_processor_and_model(config, adapter_path=resolved_ckpt_path)
    collator = QwenMultimodalCollator(processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=config["training"].get("dataloader_num_workers", 0)
    )

    # 5. Optimizer, Scheduler, CheckpointManager
    from transformers import get_cosine_schedule_with_warmup
    weight_decay = float(config["training"].get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay
    )

    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    total_continuation_steps = (len(train_loader) // grad_accum_steps) * additional_epochs
    warmup_ratio = float(config["training"].get("warmup_ratio", 0.02))
    warmup_steps = int(total_continuation_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_continuation_steps
    )

    ckpt_manager = CheckpointManager(
        checkpoint_dir=str(reporter.checkpoints_dir),
        save_total_limit=config["training"].get("save_total_limit", 3)
    )

    start_step = prev_step
    start_epoch = prev_epoch + 1
    end_epoch = prev_epoch + additional_epochs

    # 6. Runtime Estimation & Smoke Test
    if config["runtime"].get("estimate_runtime_before_training", True) and prev_epoch == 0:
        estimate_runtime(model, train_loader, additional_epochs, grad_accum_steps, config["runtime"]["time_budget_hours"])

    # 7. Training Loop
    device = next(model.parameters()).device
    model.train()
    global_step = start_step
    running_loss = 0.0
    accumulated_loss = 0.0

    best_val_f1 = -1.0
    early_stop_patience = config["runtime"].get("early_stopping_patience", 2)
    patience_counter = 0

    logger.info("=== Starting Training Loop ===")

    try:
        for epoch in range(start_epoch, end_epoch + 1):
            epoch_loss_total = 0.0
            epoch_batches = 0
            epoch_start_time = time.time()

            pbar = tqdm(
                total=len(train_loader),
                desc=f"Training {crop} | Epoch {epoch}/{end_epoch} (Continuation {epoch - prev_epoch}/{additional_epochs})",
                dynamic_ncols=True
            )

            for step_in_epoch, batch in enumerate(train_loader, 1):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    outputs = model(**batch)
                    loss = outputs.loss

                    if not torch.isfinite(loss):
                        logger.error(f"Loss is {loss.item()} at step {global_step}! Skipping batch.")
                        continue

                    scaled_loss = loss / grad_accum_steps
                    scaled_loss.backward()

                current_loss = loss.item()
                accumulated_loss += current_loss
                epoch_loss_total += current_loss
                epoch_batches += 1

                # Step optimizer when accumulation steps reached
                if step_in_epoch % grad_accum_steps == 0 or step_in_epoch == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["training"].get("max_grad_norm", 1.0)
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    running_loss = accumulated_loss / (step_in_epoch % grad_accum_steps or grad_accum_steps)
                    accumulated_loss = 0.0

                    # Structured logging
                    if global_step % config["training"].get("logging_steps", 5) == 0:
                        reporter.log_batch(
                            step=global_step,
                            epoch=epoch,
                            batch_loss=current_loss,
                            running_mean_loss=running_loss,
                            learning_rate=scheduler.get_last_lr()[0],
                            gpu_memory=get_gpu_memory_usage()
                        )

                    # Checkpoint saving
                    if global_step % config["training"].get("save_steps", 50) == 0:
                        ckpt_manager.save_checkpoint(
                            step=global_step,
                            epoch=epoch,
                            model=model,
                            processor=processor,
                            optimizer=optimizer,
                            scheduler=scheduler
                        )

                # Live progress bar update
                pbar.set_postfix({
                    "loss": f"{current_loss:.3f}",
                    "run_loss": f"{running_loss:.3f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "vram": get_gpu_memory_usage()
                })
                pbar.update(1)

            pbar.close()
            epoch_duration = time.time() - epoch_start_time
            epoch_mean_loss = epoch_loss_total / max(1, epoch_batches)

            # Mid-training validation evaluation
            val_metrics = None
            if len(val_dataset) > 0:
                val_metrics = evaluate_model(
                    model=model,
                    processor=processor,
                    dataset=val_dataset,
                    canonical_classes=canonical_classes,
                    reporter=None,
                    split_name="val"
                )
                val_f1 = val_metrics["f1_macro"]
                val_acc = val_metrics["accuracy"]
                logger.info(f"Epoch {epoch} Summary: TrainLoss={epoch_mean_loss:.4f}, ValAcc={val_acc:.4f}, ValF1={val_f1:.4f}")

                reporter.log_epoch(
                    epoch=epoch,
                    train_mean_loss=epoch_mean_loss,
                    val_loss=None,
                    val_accuracy=val_acc,
                    val_f1_macro=val_f1,
                    duration_sec=epoch_duration
                )

                # Early stopping check
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    # Save best adapter to best_adapter and adapter
                    model.save_pretrained(str(reporter.best_adapter_dir))
                    processor.save_pretrained(str(reporter.best_adapter_dir))
                    model.save_pretrained(str(reporter.adapter_dir))
                    processor.save_pretrained(str(reporter.adapter_dir))
                    logger.info(f"New best validation Macro F1: {best_val_f1:.4f}. Saved best adapter to {reporter.best_adapter_dir}")
                else:
                    patience_counter += 1
                    if config["runtime"].get("early_stopping", True) and patience_counter >= early_stop_patience:
                        logger.info(f"Early stopping triggered after {epoch} epochs.")
                        break
            else:
                reporter.log_epoch(
                    epoch=epoch,
                    train_mean_loss=epoch_mean_loss,
                    duration_sec=epoch_duration
                )

    except (KeyboardInterrupt, SystemExit):
        logger.warning("Training interrupted! Saving emergency checkpoint...")
        ckpt_manager.save_checkpoint(
            step=global_step,
            epoch=epoch,
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler
        )

    # 8. Final Model Saving & Test Evaluation
    logger.info(f"Saving final trained adapter weights to {reporter.final_adapter_dir}...")
    model.save_pretrained(str(reporter.final_adapter_dir))
    processor.save_pretrained(str(reporter.final_adapter_dir))

    test_metrics = None
    if len(test_dataset) > 0:
        logger.info("Running evaluation on held-out test split using best validation adapter...")
        best_adapter_file = reporter.best_adapter_dir / "adapter_model.safetensors"
        if best_adapter_file.exists():
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            best_weights = load_file(str(best_adapter_file))
            set_peft_model_state_dict(model, best_weights)
            logger.info("Restored best validation adapter weights for test set evaluation.")
        test_metrics = evaluate_model(
            model=model,
            processor=processor,
            dataset=test_dataset,
            canonical_classes=canonical_classes,
            reporter=reporter,
            split_name="test"
        )

    total_duration = time.time() - start_time
    reporter.save_metadata_and_summary(
        config=config,
        dataset_validation=val_report,
        test_metrics=test_metrics,
        duration_total_sec=total_duration
    )

    logger.info(f"=== Training and Evaluation complete! Artifacts saved in: {reporter.run_dir} ===")
    return str(reporter.run_dir)
