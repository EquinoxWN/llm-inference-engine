# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub:
**Security > Report a vulnerability** on this repository
(`https://github.com/EquinoxWN/llm-inference-engine/security/advisories/new`). Do not open a public issue.

Include what you found, how to reproduce it, and the impact you expect. You will get an answer
within 7 days. Fixes are released as soon as they are ready, and you are credited unless you ask
not to be.

## Supported versions

This project is pre-1.0. Only the latest commit on `main` receives fixes.

## Scope

- **In scope:** requests that make the server use unbounded memory or time (oversized prompts,
  huge `max_tokens`, unbounded queues), a request that can read or corrupt another request's KV
  cache slot, loading a model from an unpinned or malformed repository id, and anything that lets a
  client crash the scheduling loop for everyone.
- **Out of scope:** the content a model generates; GPT-2 output is not filtered and is not meant
  for end users.

## How this repository protects itself

- The server binds to `127.0.0.1` by default and has no authentication: put it behind your own
  gateway before exposing it.
- Every request body is validated (prompt up to 16,384 characters, `max_tokens` 1-1024,
  `temperature` 0-2, `top_k` 0-1000); prompt plus output must fit the KV-cache slot, and the waiting
  queue is bounded (HTTP 503 when full).
- Model checkpoints are fetched only by a validated repository id and a full 40-character commit
  hash, weights are read with `safetensors` (no pickle), and every tensor must match the model
  (strict load).
- Every GitHub Action is pinned to a full commit SHA, and workflows run with read-only
  permissions and without persisted credentials.
- Dependabot update pull requests are paused (`open-pull-requests-limit: 0`) so the repository
  keeps a single `main` branch; dependencies are updated by hand and checked by the CI audit
  (`pip-audit`, with PyTorch audited by its PyPI version).
