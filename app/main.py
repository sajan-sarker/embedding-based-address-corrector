"""FastAPI service for the embedding-based address corrector.

Run from the project root:
    uvicorn app.main:app --port 8001
"""

import json
import logging
import queue
import threading
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src import AddressCorrector, Trace, get_settings
from src.log_buffer import memory_handler, setup_logging

from .schemas import CorrectRequest, IngestRequest

logger = logging.getLogger("api")
settings = get_settings()


class Engine:
    """Holds the corrector, which loads in a background thread so the API (and its logs) is up immediately."""

    def __init__(self):
        self.status = "loading"
        self.error: str | None = None
        self.corrector: AddressCorrector | None = None
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.ready_at: str | None = None
        self.startup_trace = Trace()

    def load(self) -> None:
        try:
            self.corrector = AddressCorrector.from_settings(settings, self.startup_trace)
            self.status = "ready"
            self.ready_at = datetime.now().isoformat(timespec="seconds")
            self.startup_trace.log("startup", "Address corrector is ready", level="success")
        except Exception as exc:
            self.status = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Failed to initialise the address corrector")
            self.startup_trace.log("startup", f"Startup failed: {self.error}", level="error")

    def require(self) -> AddressCorrector:
        if self.status == "error":
            raise HTTPException(503, f"Address corrector failed to start: {self.error}")
        if self.corrector is None:
            raise HTTPException(503, "Address corrector is still loading the embedding model, try again shortly")
        return self.corrector


engine = Engine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting address corrector API (model=%s, chroma=%s)", settings.model_name, settings.chroma_path)
    threading.Thread(target=engine.load, name="engine-loader", daemon=True).start()
    yield


app = FastAPI(
    title="Embedding-based Address Corrector",
    description="Corrects district, thana and post office names in Bangla addresses using BGE-M3 "
                "embeddings, ChromaDB vector search (thana and post office search filtered by the corrected "
                "district) and character n-gram sliding re-ranking (60% sliding, 40% similarity).",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    corrector = engine.corrector
    return {
        "status": engine.status,
        "error": engine.error,
        "model": settings.model_name,
        "device": corrector.embedder.device if corrector else None,
        "dimension": corrector.embedder.dimension if corrector else None,
        "chroma_path": str(settings.chroma_path),
        "defaults": {"n_results": settings.n_results, "padding": settings.padding,
                     "sliding_weight": settings.sliding_weight, "similarity_weight": settings.similarity_weight},
        "started_at": engine.started_at,
        "ready_at": engine.ready_at,
        "collections": corrector.store.stats() if corrector else [],
    }


@app.get("/api/startup-logs")
def startup_logs():
    return {"status": engine.status, "logs": engine.startup_trace.events}


@app.get("/api/locations")
def locations():
    # `records` carries per-record metadata, e.g. each thana's / post office's district (used to filter their search)
    return [
        {
            "collection": spec.collection,
            "label": spec.label,
            "title": spec.title,
            "filter_by": spec.filter_by,
            "values": spec.unique_values,
            "records": [{"name": r.name, **dict(r.metadata)} for r in spec.unique_records],
        }
        for spec in engine.require().fields
    ]


@app.post("/api/correct")
def correct(request: CorrectRequest):
    corrector = engine.require()
    return corrector.correct(request.address, n_results=request.n_results, padding=request.padding)


@app.post("/api/correct/stream")
def correct_stream(request: CorrectRequest):
    """Same as /api/correct, but streams every log event as NDJSON while the pipeline runs.

    Lines are {"type": "log", "event": {...}}, then one {"type": "result", "result": {...}}
    or {"type": "error", "detail": "..."}.
    """
    corrector = engine.require()
    events: queue.Queue = queue.Queue()

    def run():
        try:
            trace = Trace(sink=lambda event: events.put({"type": "log", "event": event}))
            result = corrector.correct(request.address, n_results=request.n_results,
                                       padding=request.padding, trace=trace)
            events.put({"type": "result", "result": result})
        except Exception as exc:
            logger.exception("Correction failed")
            events.put({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while (item := events.get()) is not None:
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/ingest")
def ingest(request: IngestRequest):
    corrector = engine.require()
    result = corrector.reingest(Trace(), force=request.force)
    result["collections_status"] = corrector.store.stats()
    return result


@app.get("/api/logs")
def server_logs(after_id: int = Query(0, ge=0), limit: int = Query(500, ge=1, le=2000)):
    return {"last_id": memory_handler.last_id, "logs": memory_handler.since(after_id, limit)}
