"""GPT-2 checkpoints load correctly: same logits and greedy tokens as the transformers model."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from llm_inference_engine.engine import Engine, Policy, Request
from llm_inference_engine.sampling import SamplingParams
from llm_inference_engine.weights import config_from_hf, convert_state_dict, download, load_gpt


@pytest.fixture(scope="module")
def reference() -> Any:
    """A random 2-layer transformers GPT-2 (no download)."""
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=300, n_positions=64, n_embd=32, n_layer=2, n_head=4)
    ctor: Any = GPT2LMHeadModel
    return ctor(cfg).double().eval()


def test_loaded_model_matches_transformers_logits(reference: Any) -> None:
    cfg = config_from_hf(reference.config.to_dict())
    ours = load_gpt(reference.state_dict(), cfg, "cpu", torch.float64)
    tokens = torch.randint(0, 300, (3, 20), generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        expected = reference(tokens).logits
    torch.testing.assert_close(ours.forward_full(tokens), expected)


def test_engine_greedy_matches_transformers_generate(reference: Any) -> None:
    cfg = config_from_hf(reference.config.to_dict())
    ours = load_gpt(reference.state_dict(), cfg, "cpu", torch.float64)
    prompts = [[1, 2, 3], [250, 7], [9, 9, 9, 9, 9, 9]]
    eng = Engine(ours, Policy.CONTINUOUS, max_batch=3, max_seq_len=64)
    params = SamplingParams(max_new_tokens=12, stop_at_eos=False)
    got = eng.generate([Request(str(i), p, params) for i, p in enumerate(prompts)])
    for i, p in enumerate(prompts):
        t = torch.tensor([p])
        with torch.inference_mode():
            out = reference.generate(
                t,
                attention_mask=torch.ones_like(t),
                max_new_tokens=12,
                min_new_tokens=12,
                do_sample=False,
                pad_token_id=0,
            )
        assert got[str(i)] == out[0, len(p) :].tolist()


def test_conversion_transposes_conv1d_and_drops_buffers() -> None:
    w = torch.arange(6.0).reshape(2, 3)
    out = convert_state_dict(
        {
            "transformer.h.0.attn.c_attn.weight": w,
            "transformer.h.0.attn.bias": torch.ones(1),
            "lm_head.weight": torch.ones(1),
            "transformer.wte.weight": w,
        }
    )
    assert set(out) == {"h.0.attn.c_attn.weight", "wte.weight"}
    assert torch.equal(out["h.0.attn.c_attn.weight"], w.t())
    assert torch.equal(out["wte.weight"], w)


def test_config_rejects_other_architectures_and_bad_fields() -> None:
    with pytest.raises(ValueError, match="only GPT-2"):
        config_from_hf({"model_type": "llama"})
    with pytest.raises(ValueError, match="n_head"):
        config_from_hf(
            {
                "model_type": "gpt2",
                "vocab_size": 5,
                "n_positions": 8,
                "n_embd": 8,
                "n_layer": 1,
                "n_head": "4",
            }
        )


def test_download_rejects_unpinned_or_malformed_ids() -> None:
    with pytest.raises(ValueError, match="revision"):
        download("openai-community/gpt2", "main")
    with pytest.raises(ValueError, match="model id"):
        download("../../etc", "0" * 40)


def test_download_rejects_a_short_commit_hash() -> None:
    with pytest.raises(ValueError, match="revision"):
        download("openai-community/gpt2", "607a30d")
