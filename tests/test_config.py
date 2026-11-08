"""Spec A: configuration loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from arbiter.config import ConfigError, load_config

MINIMAL = """
db_path: ${ARBITER_DB}
backends:
  base_url: ${ARBITER_BASE_URL}
  embed_model: bge-m3-gguf
  rerank_model: bge-reranker-v2-m3
  chat_model: gemma-4-12B
rags:
  miethaus:
    description: Mietvertraege und Nebenkostenabrechnungen
    corpus_path: /tmp/korpus
    extensions: [".pdf"]
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "arbiter.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def _env(**extra: str) -> dict[str, str]:
    base = {"ARBITER_DB": "/tmp/db", "ARBITER_BASE_URL": "http://127.0.0.1:8080"}
    base.update(extra)
    return base


def test_loads_minimal_config_with_defaults(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, MINIMAL), _env())
    assert cfg.db_path == Path("/tmp/db")
    assert cfg.backends.embed_model == "bge-m3-gguf"
    rag = cfg.rags["miethaus"]
    assert rag.extensions == (".pdf",)
    assert rag.fuse_top_k == 20
    assert rag.rerank_top_n == 5
    assert rag.gate_schwelle == 0.5
    assert rag.chunk_chars == 600
    assert rag.gate_kandidaten == 5


def test_missing_required_field_names_field_and_file(tmp_path: Path) -> None:
    broken = MINIMAL.replace("    description: Mietvertraege und Nebenkostenabrechnungen\n", "")
    with pytest.raises(ConfigError) as err:
        load_config(_write(tmp_path, broken), _env())
    msg = str(err.value)
    assert "description" in msg
    assert "arbiter.yaml" in msg


def test_unset_variable_aborts_with_its_name(tmp_path: Path) -> None:
    env = _env()
    del env["ARBITER_BASE_URL"]
    with pytest.raises(ConfigError) as err:
        load_config(_write(tmp_path, MINIMAL), env)
    assert "ARBITER_BASE_URL" in str(err.value)


def test_empty_variable_is_an_error_not_a_default(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as err:
        load_config(_write(tmp_path, MINIMAL), _env(ARBITER_BASE_URL=""))
    assert "ARBITER_BASE_URL" in str(err.value)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    broken = MINIMAL.replace("    chunk_chars: 600", "") + "    top_k_fancy: 3\n"
    with pytest.raises(ConfigError) as err:
        load_config(_write(tmp_path, broken), _env())
    assert "top_k_fancy" in str(err.value)


def test_model_ids_come_from_config_not_code(tmp_path: Path) -> None:
    other = MINIMAL.replace("gemma-4-12B", "ein-anderes-modell")
    cfg = load_config(_write(tmp_path, other), _env())
    assert cfg.backends.chat_model == "ein-anderes-modell"


def test_real_environment_is_not_needed_when_values_are_literal(tmp_path: Path) -> None:
    literal = MINIMAL.replace("${ARBITER_DB}", "/tmp/db2").replace(
        "${ARBITER_BASE_URL}", "http://localhost:9"
    )
    cfg = load_config(_write(tmp_path, literal), {})
    assert cfg.db_path == Path("/tmp/db2")
    assert os.environ.get("ARBITER_DB") is None or cfg.db_path == Path("/tmp/db2")
