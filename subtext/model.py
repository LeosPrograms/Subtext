"""Language model loading for Subtext."""

import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def default_model_path() -> str:
    """Resolve the bundled model directory (works inside a PyInstaller bundle)."""
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    # __file__ is inside the subtext/ package; model/ sits one level up
    return os.path.join(os.path.dirname(base), 'model', 'gpt2')


def load(model_path: str | None = None, seed: int = 1234):
    """
    Load tokenizer, model, and choose device.

    Parameters
    ----------
    model_path : str or None
        Path to a local Hugging Face model directory.
        Defaults to the bundled model next to this package.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    tokenizer, model, device
    """
    if model_path is None:
        model_path = default_model_path()

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    local = os.path.isdir(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=local)
    model.to(device)
    model.eval()

    return tokenizer, model, device
