"""Load GPT-2 weights from a Hugging Face checkpoint (safetensors) into the engine's model."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .model import GPT, ModelConfig
from .tokenizer import HFTokenizer

# Default checkpoint, pinned to a commit so every run loads the same bytes.
DEFAULT_MODEL = "openai-community/gpt2"
DEFAULT_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
FILES = ("config.json", "model.safetensors", "tokenizer.json")
# GPT-2 stores these as Conv1D ([in, out]); nn.Linear wants [out, in].
TRANSPOSED = re.compile(r"\.(c_attn|c_proj|c_fc)\.weight$")
IGNORED = re.compile(r"(\.attn\.(bias|masked_bias)$)|(^lm_head\.weight$)")
REPO_ID = re.compile(r"^[\w.-]{1,96}/[\w.-]{1,96}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")


def config_from_hf(cfg: Mapping[str, object]) -> ModelConfig:
    """Build a ModelConfig from a GPT-2 config.json mapping."""
    if cfg.get("model_type") != "gpt2":
        raise ValueError(f"only GPT-2 checkpoints are supported, got {cfg.get('model_type')!r}")

    def num(key: str) -> int:
        """Read one positive integer field."""
        value = cfg.get(key)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"config.json field {key!r} must be a positive integer")
        return value

    eps = cfg.get("layer_norm_epsilon", 1e-5)
    return ModelConfig(
        vocab_size=num("vocab_size"),
        n_positions=num("n_positions"),
        n_embd=num("n_embd"),
        n_layer=num("n_layer"),
        n_head=num("n_head"),
        layer_norm_epsilon=float(eps) if isinstance(eps, int | float) else 1e-5,
    )


def convert_state_dict(hf: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Rename and transpose Hugging Face GPT-2 tensors to this model's layout."""
    out: dict[str, Tensor] = {}
    for key, value in hf.items():
        name = key.removeprefix("transformer.")
        if IGNORED.search(name):
            continue
        out[name] = value.t().contiguous() if TRANSPOSED.search(name) else value
    return out


def load_gpt(
    hf_state: Mapping[str, Tensor], config: ModelConfig, device: str, dtype: torch.dtype
) -> GPT:
    """Create the model and load every weight strictly (no missing or extra tensors)."""
    model = GPT(config)
    model.load_state_dict(convert_state_dict(hf_state), strict=True)
    return model.to(device=device, dtype=dtype).eval()


@dataclass(frozen=True)
class Checkpoint:
    """Local paths of a downloaded checkpoint."""

    repo_id: str
    revision: str
    directory: Path

    def config(self) -> ModelConfig:
        """Parse config.json."""
        text = (self.directory / "config.json").read_text(encoding="utf-8")
        return config_from_hf(json.loads(text))

    def model(self, device: str, dtype: torch.dtype) -> GPT:
        """Load the weights."""
        from safetensors.torch import load_file

        state = load_file(str(self.directory / "model.safetensors"))
        return load_gpt(state, self.config(), device, dtype)

    def tokenizer(self) -> HFTokenizer:
        """Load tokenizer.json."""
        return HFTokenizer(self.directory / "tokenizer.json")


def download(repo_id: str = DEFAULT_MODEL, revision: str = DEFAULT_REVISION) -> Checkpoint:
    """Fetch the three files at a pinned commit into the Hugging Face cache (HF_HOME)."""
    if not REPO_ID.match(repo_id):
        raise ValueError(f"invalid model id {repo_id!r}")
    if not REVISION.match(revision):
        raise ValueError("revision must be a full 40-character commit hash")
    from huggingface_hub import hf_hub_download

    paths = [hf_hub_download(repo_id, f, revision=revision) for f in FILES]
    return Checkpoint(repo_id, revision, Path(paths[0]).parent)
