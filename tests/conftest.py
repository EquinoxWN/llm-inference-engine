"""Shared fixtures: a tiny randomly initialised model in float64 so batched and single runs agree."""

from __future__ import annotations

import pytest
import torch

from llm_inference_engine.cli import tiny_model
from llm_inference_engine.model import GPT


@pytest.fixture(scope="session")
def model() -> GPT:
    """2-layer GPT over bytes, seed 0, float64 on CPU."""
    torch.set_num_threads(2)
    return tiny_model(0).to(torch.float64)
