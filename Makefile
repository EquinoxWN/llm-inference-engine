.PHONY: setup setup-cuda lint test audit verify bench bench-gpu serve ci

PY ?= python
TORCH_CPU = https://download.pytorch.org/whl/cpu
TORCH_CUDA = https://download.pytorch.org/whl/cu130

# CPU build of PyTorch (as in CI), then the package and dev tools. setuptools comes from PyPI
# first: the PyTorch index only carries an old copy (78.1.0) with known advisories.
setup:
	$(PY) -m pip install --upgrade pip setuptools && $(PY) -m pip install torch --index-url $(TORCH_CPU)
	$(PY) -m pip install -e ".[dev]"

# CUDA 13.0 build of PyTorch for the GPU benchmark.
setup-cuda:
	$(PY) -m pip install --upgrade pip setuptools && $(PY) -m pip install torch --index-url $(TORCH_CUDA)
	$(PY) -m pip install -e ".[dev]"

# ruff, ruff format and mypy --strict.
lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check . && $(PY) -m mypy

# 36 tests on a tiny randomly initialised model (CPU, no download).
test:
	$(PY) -m pytest -q

# Known vulnerabilities in the installed dependencies; torch is audited by its PyPI version
# because the PyTorch index adds a local tag (+cpu, +cu130) that PyPI does not know.
audit:
	$(PY) -m pip_audit --skip-editable --cache-dir .tmp/pip-audit
	$(PY) -c "import torch; print('torch==' + torch.__version__.split('+')[0])" > .tmp/torch-req.txt
	$(PY) -m pip_audit --no-deps -r .tmp/torch-req.txt --cache-dir .tmp/pip-audit

# GPT-2 (pinned revision) against transformers: logits and greedy tokens.
verify:
	lie verify --model openai-community/gpt2 --device cpu

# Naive vs static vs continuous on GPT-2, CPU.
bench:
	lie bench --model openai-community/gpt2 --device cpu --max-batch 16 --requests 32 --output-len 16 128

# The README numbers: GPT-2 in float16 on a CUDA GPU, all requests at t=0 and at 4 requests/s.
bench-gpu:
	lie bench --model openai-community/gpt2 --device cuda --dtype float16 --max-batch 16 --requests 32 --output-len 16 128
	lie bench --model openai-community/gpt2 --device cuda --dtype float16 --max-batch 16 --requests 32 --output-len 16 128 --rate 4

# OpenAI-style API on 127.0.0.1:8000.
serve:
	lie serve --model openai-community/gpt2 --policy continuous

ci: setup lint test
