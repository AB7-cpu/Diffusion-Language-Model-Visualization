"""
Shared checkpoint loading used by both test_engine.py (CLI) and server.py
(FastAPI).
"""

import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


def default_checkpoint_path() -> str:
    local_path = Path(__file__).resolve().parent.parent / "checkpoints" / "LLaDA-8B-Instruct"
    if local_path.is_dir():
        return str(local_path)
    return "GSAI-ML/LLaDA-8B-Instruct"


def load_model(model_name_or_path: str):
    """Load LLaDA-8B-Instruct in 4-bit (bitsandbytes)."""
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    ).eval()
    return model, tokenizer