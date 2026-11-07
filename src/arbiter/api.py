"""HTTP layer (spec §10). A thin client of the library, nothing more.

Routes: ``/health``, ``/rags``, ``/search``, ``/index`` (job with id) and
``/index/{job_id}``. Indexing runs in a thread and reports numbers, because a
real corpus takes minutes and no gateway waits that long.
"""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from arbiter import __version__
from arbiter.config import Config, ConfigError, Rag, load_config
from arbiter.embedding import EmbeddingError
from arbiter.output import suchergebnis_json
from arbiter.pipeline import IndexReport, Pipeline, QuestionTooLong
from arbiter.store import StoreError

EXIT_CONFIG = 1


class SearchRequest(BaseModel):
    rag: str
    frage: str = Field(min_length=1)


class IndexRequest(BaseModel):
    rag: str


class Job(BaseModel):
    job_id: str
    rag: str
    status: str  # "laeuft" | "fertig" | "fehler"
    bericht: dict[str, Any] | None = None
    meldung: str | None = None


def create_app(
    config: Config | None = None,
    factory: Callable[[Config, Rag], Pipeline] | None = None,
) -> FastAPI:
    app = FastAPI(title="ArbiterRAG", version=__version__)
    cfg = config
    jobs: dict[str, Job] = {}
    running: dict[str, str] = {}
    lock = threading.Lock()

    def pipeline_for(rag_name: str) -> Pipeline:
        if cfg is None:
            raise HTTPException(status_code=503, detail="Keine Konfiguration geladen")
        rag = cfg.rags.get(rag_name)
        if rag is None:
            known = ", ".join(sorted(cfg.rags)) or "keine"
            raise HTTPException(
                status_code=404, detail=f"Unbekannter RAG '{rag_name}'. Bekannt: {known}"
            )
        return factory(cfg, rag) if factory else Pipeline(cfg, rag)

    def run_job(job_id: str, rag_name: str) -> None:
        try:
            report = pipeline_for(rag_name).index()
            with lock:
                jobs[job_id] = jobs[job_id].model_copy(
                    update={"status": "fertig", "bericht": _report_dict(report)}
                )
        except Exception as exc:
            with lock:
                jobs[job_id] = jobs[job_id].model_copy(
                    update={"status": "fehler", "meldung": str(exc)}
                )
        finally:
            with lock:
                running.pop(rag_name, None)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/rags")
    def rags() -> list[dict[str, Any]]:
        if cfg is None:
            raise HTTPException(status_code=503, detail="Keine Konfiguration geladen")
        out = []
        for rag in cfg.rags.values():
            try:
                abschnitte = pipeline_for(rag.name).store.count(rag.name)
            except (StoreError, HTTPException):
                abschnitte = 0
            out.append({"name": rag.name, "description": rag.description, "abschnitte": abschnitte})
        return out

    @app.post("/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        pipeline = pipeline_for(request.rag)
        try:
            result = pipeline.search(request.frage)
        except QuestionTooLong as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except StoreError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except EmbeddingError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return suchergebnis_json(result)

    @app.post("/index", status_code=202)
    def index(request: IndexRequest) -> dict[str, str]:
        pipeline_for(request.rag)  # validates the name before a job is created
        with lock:
            if request.rag in running:
                raise HTTPException(
                    status_code=409,
                    detail=f"Für '{request.rag}' läuft bereits ein Index-Lauf "
                    f"({running[request.rag]})",
                )
            job_id = uuid.uuid4().hex[:12]
            jobs[job_id] = Job(job_id=job_id, rag=request.rag, status="laeuft")
            running[request.rag] = job_id
        threading.Thread(target=run_job, args=(job_id, request.rag), daemon=True).start()
        return {"job_id": job_id}

    @app.get("/index/{job_id}")
    def job_status(job_id: str) -> Job:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unbekannter Auftrag '{job_id}'")
        return job

    return app


def _report_dict(report: IndexReport) -> dict[str, Any]:
    return {
        "dateien": report.files,
        "abschnitte": report.chunks,
        "neu_oder_geaendert": report.changed,
        "uebersprungen": report.skipped_files,
        "seiten_ohne_text": report.textless_pages,
        "seiten_per_ocr": report.ocr_pages,
        "warnungen": report.warnings,
    }


def main() -> None:
    """Entry point: ``arbiter-api``. A broken configuration stops the start."""
    try:
        config = load_config(Path(os.environ.get("ARBITER_CONFIG", "arbiter.yaml")), os.environ)
    except ConfigError as exc:
        raise SystemExit(f"Konfiguration unbrauchbar: {exc}") from exc

    host = os.environ.get("ARBITER_BIND", "127.0.0.1")
    port = int(os.environ.get("ARBITER_PORT", "8091"))
    uvicorn.run(create_app(config), host=host, port=port, log_level="info")


app = create_app()
