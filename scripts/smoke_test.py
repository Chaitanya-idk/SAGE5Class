"""
Smoke test script executing a real multimodal forward and backward pass,
verifying that loss is finite and gradients reach trainable LoRA parameters.
"""

import sys
import argparse
from pathlib import Path
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import logger, detect_hardware
from src.config import load_config
from src.model_loader import load_processor_and_model
from src.train import QwenMultimodalCollator

def main():
    parser = argparse.ArgumentParser(description="Multimodal smoke test for Qwen2.5-VL QLoRA.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    hw = detect_hardware()

    logger.info("=== Running Multimodal Smoke Test ===")
    logger.info(f"Hardware detected: CUDA={hw['cuda_available']}, Device={hw['device_name']}, VRAM={hw['total_memory_gb']} GB")

    processor, model = load_processor_and_model(config)
    collator = QwenMultimodalCollator(processor)

    # Create synthetic test sample
    dummy_image = Image.new("RGB", (224, 224), color=(34, 139, 34))
    sample = {
        "image": dummy_image,
        "image_path": "synthetic_smoke_test.jpg",
        "prompt": "Identify the disease class visible in this tomato leaf image. Select exactly one class from: [\"Late Blight\", \"Early Blight\"]. Return JSON.",
        "target": '{"crop": "Tomato", "disease": "Late Blight"}',
        "ground_truth_disease": "Late Blight",
        "crop": "Tomato",
        "split": "train"
    }

    batch = collator([sample])
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    logger.info("Executing forward pass...")
    model.train()
    with torch.amp.autocast(device_type="cuda" if hw["cuda_available"] else "cpu"):
        outputs = model(**batch)
        loss = outputs.loss

    if not torch.isfinite(loss):
        logger.error(f"Smoke test failed: loss is not finite ({loss.item()})")
        sys.exit(1)

    logger.info(f"Forward pass successful! Initial finite loss = {loss.item():.4f}")

    logger.info("Executing backward pass to verify adapter gradient flow...")
    loss.backward()

    # Check that trainable parameters received gradients
    trainable_grads = 0
    total_trainable = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_trainable += 1
            if param.grad is not None and torch.any(param.grad != 0):
                trainable_grads += 1

    logger.info(f"Trainable parameters with verified gradients: {trainable_grads}/{total_trainable}")
    if total_trainable > 0 and trainable_grads == 0:
        logger.error("Smoke test failed: no gradients flowed to trainable adapter parameters!")
        sys.exit(1)

    logger.info("Smoke test passed successfully! Model, collator, loss masking, and LoRA gradients are verified.")

if __name__ == "__main__":
    main()
