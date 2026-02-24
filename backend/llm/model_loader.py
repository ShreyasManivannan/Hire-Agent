"""
LLM Model Loader — Handles loading quantized models (4-bit/8-bit) with BitsAndBytes.
Supports Mistral-7B-Instruct or falls back to a smaller model / API-based inference.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global model references
_model = None
_tokenizer = None
_use_api = False


def get_quantization_config(bits: int = 4):
    """Create BitsAndBytes quantization config for 4-bit or 8-bit."""
    try:
        from transformers import BitsAndBytesConfig
        import torch

        if bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Unsupported quantization bits: {bits}")
    except ImportError:
        logger.warning("BitsAndBytes not available, will use unquantized model or API")
        return None


def load_model(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_bits: int = 4,
    force_api: bool = False,
):
    """
    Load a quantized LLM model.
    Falls back to API-based inference if GPU is not available.
    """
    global _model, _tokenizer, _use_api

    if force_api:
        _use_api = True
        logger.info("Using API-based inference mode")
        return None, None

    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning("No CUDA GPU detected. Falling back to API-based inference.")
            _use_api = True
            return None, None

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading {model_name} with {quantization_bits}-bit quantization...")

        quant_config = get_quantization_config(quantization_bits)

        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if quant_config:
            load_kwargs["quantization_config"] = quant_config

        _model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        _model.eval()

        logger.info(f"Model loaded successfully on {_model.device}")
        return _model, _tokenizer

    except Exception as e:
        logger.error(f"Failed to load local model: {e}")
        logger.info("Falling back to API-based inference.")
        _use_api = True
        return None, None


def get_model():
    """Get the loaded model and tokenizer."""
    return _model, _tokenizer


def is_api_mode():
    """Check if using API-based inference."""
    return _use_api
