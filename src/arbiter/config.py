"""Configuration loading (spec A).

Values come from one YAML file. ``${VAR}`` placeholders are expanded from the
given environment; an unset or empty variable is an error, never a default.
Unknown keys are rejected so a typo cannot silently disable a setting.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when the configuration is unusable."""


@dataclass(frozen=True)
class Backends:
    base_url: str
    embed_model: str
    rerank_model: str
    chat_model: str
    ocr_dots_model: str = ""
    ocr_paddle_model: str = ""


@dataclass(frozen=True)
class Rag:
    name: str
    description: str
    corpus_path: Path
    extensions: tuple[str, ...]
    top_k_dense: int = 20
    top_k_fts: int = 20
    fuse_top_k: int = 20
    rerank_top_n: int = 5
    gate_schwelle: float = 0.5
    hyde: bool = True
    hyde_max_chars: int = 800
    ocr: bool = False
    ocr_dpi: int = 200
    rerank: bool = True
    chunk_chars: int = 600
    chunk_overlap: int = 100
    hybrid: bool = True
    fts_stem: bool = False
    hyde_gewicht: float = 0.5
    gate_prompt: str = "teil"
    gate_kandidaten: int = 5
    antwort_passagen: int = 3


@dataclass(frozen=True)
class Config:
    db_path: Path
    backends: Backends
    rags: dict[str, Rag]


_TOP_KEYS = {"db_path", "backends", "rags"}
_BACKEND_KEYS = {
    "base_url",
    "embed_model",
    "rerank_model",
    "chat_model",
    "ocr_dots_model",
    "ocr_paddle_model",
}
_RAG_KEYS = {
    "description",
    "corpus_path",
    "extensions",
    "top_k_dense",
    "top_k_fts",
    "fuse_top_k",
    "rerank_top_n",
    "hyde",
    "hyde_max_chars",
    "ocr",
    "ocr_dpi",
    "rerank",
    "chunk_chars",
    "chunk_overlap",
    "hybrid",
    "fts_stem",
    "hyde_gewicht",
    "gate_prompt",
    "gate_kandidaten",
    "antwort_passagen",
}
_REQUIRED_BACKENDS = {
    "base_url",
    "embed_model",
    "rerank_model",
    "chat_model",
}
_REQUIRED_RAG = {"description", "corpus_path", "extensions"}


def load_config(path: Path, environ: Mapping[str, str]) -> Config:
    """Load and validate the configuration file."""
    if not path.is_file():
        raise ConfigError(f"Konfigurationsdatei nicht gefunden: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - message content is yaml's
        raise ConfigError(f"{path.name}: YAML nicht lesbar: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name}: oberste Ebene muss ein Objekt sein")

    data = _expand(raw, path.name, environ)
    _check_keys(data, _TOP_KEYS, path.name, "oberste Ebene")

    for field in ("db_path", "backends", "rags"):
        if field not in data:
            raise ConfigError(f"{path.name}: Pflichtfeld fehlt: {field}")

    backends_raw = data["backends"]
    if not isinstance(backends_raw, dict):
        raise ConfigError(f"{path.name}: 'backends' muss ein Objekt sein")
    _check_keys(backends_raw, _BACKEND_KEYS, path.name, "backends")
    for field in sorted(_REQUIRED_BACKENDS):
        if not backends_raw.get(field):
            raise ConfigError(f"{path.name}: Pflichtfeld fehlt: backends.{field}")

    backends = Backends(
        base_url=str(backends_raw["base_url"]),
        embed_model=str(backends_raw["embed_model"]),
        rerank_model=str(backends_raw["rerank_model"]),
        chat_model=str(backends_raw["chat_model"]),
        ocr_dots_model=str(backends_raw.get("ocr_dots_model", "")),
        ocr_paddle_model=str(backends_raw.get("ocr_paddle_model", "")),
    )

    rags_raw = data["rags"]
    if not isinstance(rags_raw, dict) or not rags_raw:
        raise ConfigError(f"{path.name}: 'rags' muss mindestens einen Eintrag haben")

    rags: dict[str, Rag] = {}
    for name, entry in rags_raw.items():
        rags[str(name)] = _rag(str(name), entry, path.name)

    for rag in rags.values():
        if rag.ocr and not (backends.ocr_dots_model and backends.ocr_paddle_model):
            raise ConfigError(
                f"{path.name}: rag '{rag.name}' hat ocr: true, aber "
                "backends.ocr_dots_model und backends.ocr_paddle_model fehlen"
            )

    return Config(
        db_path=Path(str(data["db_path"])),
        backends=backends,
        rags=rags,
    )


def _rag(name: str, entry: Any, filename: str) -> Rag:
    if not isinstance(entry, dict):
        raise ConfigError(f"{filename}: rag '{name}' muss ein Objekt sein")
    _check_keys(entry, _RAG_KEYS, filename, f"rag '{name}'")
    for field in sorted(_REQUIRED_RAG):
        if field not in entry or entry[field] in (None, "", []):
            raise ConfigError(f"{filename}: Pflichtfeld fehlt: rags.{name}.{field}")

    extensions = entry["extensions"]
    if isinstance(extensions, str):
        extensions = [extensions]
    if not isinstance(extensions, list) or not extensions:
        raise ConfigError(f"{filename}: rags.{name}.extensions muss eine Liste sein")
    norm = tuple(
        (str(ext).lower() if str(ext).startswith(".") else "." + str(ext).lower())
        for ext in extensions
    )

    return Rag(
        name=name,
        description=str(entry["description"]),
        corpus_path=Path(str(entry["corpus_path"])).expanduser(),
        extensions=norm,
        top_k_dense=int(entry.get("top_k_dense", 20)),
        top_k_fts=int(entry.get("top_k_fts", 20)),
        fuse_top_k=int(entry.get("fuse_top_k", 20)),
        rerank_top_n=int(entry.get("rerank_top_n", 5)),
        gate_schwelle=float(entry.get("gate_schwelle", 0.5)),
        hyde=bool(entry.get("hyde", True)),
        hyde_max_chars=int(entry.get("hyde_max_chars", 800)),
        ocr=bool(entry.get("ocr", False)),
        ocr_dpi=int(entry.get("ocr_dpi", 200)),
        rerank=bool(entry.get("rerank", True)),
        chunk_chars=int(entry.get("chunk_chars", 600)),
        chunk_overlap=int(entry.get("chunk_overlap", 100)),
        hybrid=bool(entry.get("hybrid", True)),
        fts_stem=bool(entry.get("fts_stem", False)),
        hyde_gewicht=float(entry.get("hyde_gewicht", 0.5)),
        gate_prompt=str(entry.get("gate_prompt", "teil")),
        gate_kandidaten=int(entry.get("gate_kandidaten", 5)),
        antwort_passagen=int(entry.get("antwort_passagen", 3)),
    )


def _check_keys(data: Mapping[str, Any], allowed: set[str], filename: str, where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"{filename}: unbekannter Schlüssel in {where}: {', '.join(unknown)}")


def _expand(value: Any, filename: str, environ: Mapping[str, str]) -> Any:
    """Expand ``${VAR}`` recursively; an unset or empty variable is an error."""
    if isinstance(value, str):
        return _expand_str(value, filename, environ)
    if isinstance(value, dict):
        return {key: _expand(item, filename, environ) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, filename, environ) for item in value]
    return value


def _expand_str(text: str, filename: str, environ: Mapping[str, str]) -> str:
    out = text
    start = out.find("${")
    while start != -1:
        end = out.find("}", start)
        if end == -1:
            raise ConfigError(f"{filename}: unvollständiger Platzhalter in '{text}'")
        name = out[start + 2 : end]
        if name not in environ:
            raise ConfigError(f"{filename}: Umgebungsvariable {name} ist nicht gesetzt")
        if not environ[name]:
            raise ConfigError(f"{filename}: Umgebungsvariable {name} ist leer — kein Standardwert")
        out = out[:start] + environ[name] + out[end + 1 :]
        start = out.find("${", start + len(environ[name]))
    return out


def env_config_path(environ: Mapping[str, str] | None = None) -> Path:
    """Path of the configuration file, from ``ARBITER_CONFIG``."""
    env = os.environ if environ is None else environ
    value = env.get("ARBITER_CONFIG", "")
    if not value:
        raise ConfigError("ARBITER_CONFIG ist nicht gesetzt")
    return Path(value)
