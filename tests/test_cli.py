"""The `lie` command line on the tiny random model (no download)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_inference_engine.cli import main


def test_generate_prints_prompt_and_continuation(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["generate", "hi", "--max-tokens", "4"]) == 0
    assert capsys.readouterr().out.startswith("hi")


def test_bench_prints_table_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "bench.json"
    code = main(
        [
            "bench",
            "--requests",
            "6",
            "--prompt-len",
            "4",
            "8",
            "--output-len",
            "2",
            "6",
            "--max-batch",
            "3",
            "--max-seq-len",
            "64",
            "--json",
            str(out),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "| naive |" in text
    assert "| continuous |" in text
    rows = json.loads(out.read_text("utf-8"))
    assert [r["policy"] for r in rows] == ["naive", "static", "continuous"]


def test_bad_arguments_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["generate", "x", "--max-seq-len", "5000"]) == 2
    assert main(["verify"]) == 2
    assert main(["bench", "--output-len", "1", "600"]) == 2
    assert "error" in capsys.readouterr().err
