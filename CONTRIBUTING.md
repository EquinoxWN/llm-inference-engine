# Contributing

Thanks for helping. This is a Python project; issues and pull requests are welcome.

## Set up and check your change

Needs Python 3.11+. `make setup` installs the CPU build of PyTorch; `make setup-cuda` the CUDA 13.0
build for the GPU benchmark. Caches, models and virtual environments stay in git-ignored folders
inside the repo.

```bash
make setup    # install dependencies (CPU PyTorch)
make lint     # ruff, ruff format, mypy --strict
make test     # full test suite on a tiny random model, no download
make audit    # known-vulnerability check (as in CI)
make verify   # download GPT-2 (pinned revision) and compare with transformers
```

A pull request is ready when `make lint` and `make test` pass and CI is green.

## Guidelines

- Keep each change focused; explain the problem it solves in the pull request.
- Add or update a test for every behaviour change, and update `docs/results/` when numbers change.
- Scheduler changes must keep `test_continuous_matches_naive_greedy_with_random_arrivals` green:
  batching may change speed, never tokens.
- Significant design changes need an ADR in `docs/adr/` (copy `0001` for the format).
- Keep function comments to one short phrase; let names and tests explain the rest.
- Report security problems privately, as described in [SECURITY.md](SECURITY.md).

By contributing you agree that your work is released under the [MIT License](LICENSE).
