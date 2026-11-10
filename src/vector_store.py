import hashlib
import re
import shutil
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import chromadb
import numpy as np

from .embedding import Embedder
from .locations import FieldSpec
from .trace import Trace

# Chroma keeps each collection's HNSW index in a folder named after its segment id
SEGMENT_DIR = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class LocationStore:
    """One cosine-space ChromaDB collection per address field ("জেলা: ঢাকা", "থানা: ...", ...)."""

    def __init__(self, path: Path, embedder: Embedder, fields: tuple[FieldSpec, ...]):
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.client = chromadb.PersistentClient(path=str(path))
        self.embedder = embedder
        self.fields = fields
        # Re-ingestion empties collections, so queries must not run while it happens
        self.lock = threading.RLock()

    def _fingerprint(self, spec: FieldSpec) -> str:
        # Changes whenever the reference names, their metadata, the document format or the model change
        payload = "\n".join([
            self.embedder.model_name,
            *(f"{spec.document(r.name)}\t{r.metadata}" for r in spec.unique_records),
        ])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _get(self, name: str):
        try:
            return self.client.get_collection(name)
        except Exception:
            return None

    def status(self, spec: FieldSpec) -> tuple[bool, str]:
        collection = self._get(spec.collection)
        if collection is None:
            return False, "collection is missing"
        if (collection.metadata or {}).get("fingerprint") != self._fingerprint(spec):
            return False, "reference data or model changed since last ingestion"
        if collection.count() != len(spec.unique_records):
            return False, "record count does not match reference data"
        return True, "up to date"

    def _reset_collection(self, spec: FieldSpec):
        # Empty the collection in place rather than dropping it: on Windows a dropped collection's
        # segment folder is still open and gets left behind in chroma_db
        metadata = {"label": spec.label, "model": self.embedder.model_name, "fingerprint": self._fingerprint(spec)}
        collection = self._get(spec.collection)
        if collection is None:
            return self.client.create_collection(name=spec.collection, metadata={"hnsw:space": "cosine", **metadata})
        ids = collection.get(include=[])["ids"]
        batch = self.client.get_max_batch_size()
        for start in range(0, len(ids), batch):
            collection.delete(ids=ids[start:start + batch])
        collection.modify(metadata=metadata)  # hnsw:space can't change after creation, the collection keeps cosine
        return collection

    def prune_orphan_segments(self, trace: Trace) -> list[str]:
        # Remove segment folders that chroma.sqlite3 no longer references (left over from dropped collections)
        try:
            uri = (self.path / "chroma.sqlite3").as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as conn:
                live = {row[0] for row in conn.execute("SELECT id FROM segments")}
        except sqlite3.Error as exc:
            trace.log("ingest", f"Skipped pruning orphaned segment folders: {exc}", level="warning")
            return []

        removed = []
        for folder in self.path.iterdir():
            if folder.is_dir() and SEGMENT_DIR.match(folder.name) and folder.name not in live:
                shutil.rmtree(folder, ignore_errors=True)
                if not folder.exists():
                    removed.append(folder.name)
        if removed:
            trace.log("ingest", f"Removed {len(removed)} orphaned segment folder(s) from {self.path}",
                      data={"folders": removed})
        return removed

    def drop_stale_collections(self, trace: Trace) -> list[str]:
        # Collections no field uses any more (e.g. the old "city"); their folders are pruned on the next start
        wanted = {spec.collection for spec in self.fields}
        stale = [c.name for c in self.client.list_collections() if c.name not in wanted]
        for name in stale:
            self.client.delete_collection(name)
        if stale:
            trace.log("ingest", f"Dropped {len(stale)} collection(s) no field uses any more: {', '.join(stale)}",
                      level="warning", data={"collections": stale})
        return stale

    def ingest(self, spec: FieldSpec, trace: Trace) -> int:
        records = spec.unique_records
        texts = [spec.document(record.name) for record in records]
        with trace.timed(f"ingest_embed_{spec.collection}") as timer:
            embeddings = self.embedder.encode(texts)
        trace.log(
            "ingest",
            f"[{spec.title}] Encoded {len(texts)} documents in {timer['ms']} ms",
            data={"collection": spec.collection, "examples": texts[:5], "dimension": int(embeddings.shape[1])},
        )

        ids = [f"{spec.collection}_{i}" for i in range(len(texts))]
        metadatas = [{"name": record.name, **dict(record.metadata)} for record in records]
        batch = self.client.get_max_batch_size()  # thousands of post offices can exceed one upsert
        with self.lock:
            collection = self._reset_collection(spec)
            for start in range(0, len(texts), batch):
                end = start + batch
                collection.upsert(
                    ids=ids[start:end],
                    documents=texts[start:end],
                    embeddings=embeddings[start:end].tolist(),
                    metadatas=metadatas[start:end],
                )
            count = collection.count()
        trace.log("ingest", f"[{spec.title}] Collection '{spec.collection}' now holds {count} records",
                  level="success", data={"collection": spec.collection, "count": count})
        return count

    def ensure_ingested(self, trace: Trace, force: bool = False) -> dict:
        self.prune_orphan_segments(trace)
        self.drop_stale_collections(trace)
        summary = {}
        for spec in self.fields:
            current, reason = self.status(spec)
            if current and not force:
                trace.log("ingest", f"[{spec.title}] Skipping, collection is {reason}",
                          data={"collection": spec.collection})
                summary[spec.collection] = {"ingested": False, "count": self._get(spec.collection).count()}
                continue
            trace.log("ingest", f"[{spec.title}] Ingesting: {'forced re-ingestion' if force else reason}")
            summary[spec.collection] = {"ingested": True, "count": self.ingest(spec, trace)}
        return summary

    def count(self, spec: FieldSpec, where: dict | None = None) -> int:
        # Records a query with this metadata filter would search
        with self.lock:
            collection = self.client.get_collection(spec.collection)
            return len(collection.get(where=where, include=[])["ids"]) if where else collection.count()

    def query(self, spec: FieldSpec, embedding: np.ndarray, n_results: int, where: dict | None = None) -> list[dict]:
        # `where` is a metadata filter, e.g. {"district": "ঢাকা"}: only matching records are searched
        with self.lock:
            collection = self.client.get_collection(spec.collection)
            # With a filter, Chroma returns at most the matching records (none when nothing matches)
            result = collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=max(1, min(n_results, collection.count())),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        return [
            {
                "rank": rank,
                "id": record_id,
                "name": metadata["name"],
                "metadata": {key: value for key, value in metadata.items() if key != "name"},
                "text": document,
                "distance": max(float(distance), 0.0),  # cosine distance can come back as -0.0
                "similarity": 1.0 - max(float(distance), 0.0),
            }
            for rank, (record_id, document, metadata, distance) in enumerate(
                zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]),
                start=1,
            )
        ]

    def stats(self) -> list[dict]:
        rows = []
        for spec in self.fields:
            collection = self._get(spec.collection)
            current, reason = self.status(spec)
            rows.append({
                "collection": spec.collection,
                "label": spec.label,
                "title": spec.title,
                "count": collection.count() if collection else 0,
                "reference_count": len(spec.unique_records),
                "filter_by": spec.filter_by,
                "up_to_date": current,
                "status": reason,
            })
        return rows
