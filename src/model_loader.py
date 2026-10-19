"""
Model and processor loading, Kaggle input discovery, 4-bit NF4 quantization,
and QLoRA adapter configuration for Qwen2.5-VL.
"""

import os
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import torch
from src.utils import logger, detect_hardware

def discover_qwen_model_path(
    kaggle_input_root: str = "/kaggle/input",
    expected_substrings: Optional[List[str]] = None
) -> Optional[str]:
    """
    Recursively scans the Kaggle input directory up to 3 levels deep to locate
    a candidate Qwen vision-language model directory containing config.json.
    """
    if expected_substrings is None:
        expected_substrings = ["qwen2.5-vl", "qwen2_5_vl", "qwen-vl", "qwen"]

    root = Path(kaggle_input_root)
    if not root.exists():
        return None

    candidates: List[Tuple[int, Path]] = []
    try:
        # Search for config.json files
        for config_file in root.glob("**/config.json"):
            parent = config_file.parent
            parent_name = str(parent).lower()
            # Score candidate by presence of expected keywords
            score = sum(1 for sub in expected_substrings if sub in parent_name)
            candidates.append((score, parent))
    except Exception as e:
        logger.warning(f"Error while scanning {kaggle_input_root}: {e}")

    if not candidates:
        return None

    # Sort candidates by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_candidate = str(candidates[0][1])
    logger.info(f"Discovered Qwen model directory: {best_candidate}")
    return best_candidate

def resolve_model_path(config: Dict[str, Any]) -> str:
    """
    Resolves the actual model path using config, environment variables,
    or Kaggle input discovery.
    """
    # 1. Config explicit path
    path = config["model"].get("model_path")
    if path and Path(path).exists():
        return str(path)

    # 2. Environment variable
    env_var = config["model"].get("model_path_env", "QWEN_MODEL_PATH")
    if os.environ.get(env_var):
        env_path = os.environ[env_var]
        if Path(env_path).exists():
            return env_path
        logger.warning(f"Path in {env_var} does not exist: {env_path}")

    # 3. Kaggle input discovery
    discovered = discover_qwen_model_path()
    if discovered:
        return discovered

    # 4. Fallback to model name on Hugging Face (or error)
    model_name = config["model"].get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct")
    logger.info(f"Using model identifier: {model_name}")
    return model_name

def load_processor_and_model(
    config: Dict[str, Any]
) -> Tuple[Any, Any]:
    """
    Loads AutoProcessor and Qwen2.5-VL model with 4-bit NF4 QLoRA adapter
    according to configuration and available hardware.
    """
    from transformers import AutoProcessor, AutoConfig
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelClass

    model_path = resolve_model_path(config)
    trust_remote_code = config["model"].get("trust_remote_code", False)

    logger.info(f"Loading processor from: {model_path}")
    processor = None
    min_pix = config["preprocessing"]["min_image_side"] * config["preprocessing"]["min_image_side"]
    max_pix = config["preprocessing"]["max_pixels"]

    # 1. Attempt loading AutoProcessor from local model_path
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            min_pixels=min_pix,
            max_pixels=max_pix
        )
    except Exception as err:
        logger.warning(f"AutoProcessor.from_pretrained failed on local path: {err}")

    # 2. Attempt Qwen2_5_VLProcessor directly
    if processor is None:
        try:
            from transformers import Qwen2_5_VLProcessor
            processor = Qwen2_5_VLProcessor.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                min_pixels=min_pix,
                max_pixels=max_pix
            )
        except Exception as err:
            logger.warning(f"Qwen2_5_VLProcessor failed on local path: {err}")

    # 3. Fallback: Load processor config from Hugging Face hub (weights stay local)
    if processor is None:
        hub_fallback = "Qwen/Qwen2.5-VL-3B-Instruct"
        logger.info(f"Loading processor definitions from Hugging Face hub '{hub_fallback}'...")
        processor = AutoProcessor.from_pretrained(
            hub_fallback,
            trust_remote_code=trust_remote_code,
            min_pixels=min_pix,
            max_pixels=max_pix
        )

    hw = detect_hardware()
    quant_method = config["model"].get("quantization", "4bit_nf4")
    compute_dtype = torch.bfloat16 if config["training"]["bf16"] else torch.float16

    quant_config = None
    if quant_method == "4bit_nf4" and hw["cuda_available"]:
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype
            )
            logger.info("Configured BitsAndBytes 4-bit NF4 quantization.")
        except ImportError:
            logger.warning("bitsandbytes not installed. Proceeding with standard precision.")

    logger.info(f"Loading base model from: {model_path}")
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": compute_dtype if hw["cuda_available"] else torch.float32,
    }

    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    elif hw["cuda_available"]:
        model_kwargs["device_map"] = "auto"

    model = ModelClass.from_pretrained(
        model_path,
        **model_kwargs
    )

    if config["training"].get("gradient_checkpointing", True) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Configure PEFT QLoRA if requested
    if config["training"].get("method", "qlora") == "qlora":
        from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType

        if quant_config is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=config["training"].get("gradient_checkpointing", True)
            )

        target_modules = config["lora"].get("target_modules", "auto")
        if target_modules == "auto" or not target_modules:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        lora_config = LoraConfig(
            r=config["lora"].get("r", 16),
            lora_alpha=config["lora"].get("alpha", 32),
            lora_dropout=config["lora"].get("dropout", 0.05),
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = get_peft_model(model, lora_config)
        trainable_params, total_params = model.get_nb_trainable_parameters()
        pct = 100 * trainable_params / total_params
        logger.info(
            f"LoRA adapter initialized: {trainable_params:,} trainable / {total_params:,} total parameters ({pct:.3f}%)"
        )

    return processor, model
