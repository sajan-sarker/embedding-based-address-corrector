"""End-to-end address correction: extract -> embed -> vector search -> sliding re-rank -> compose.

Fields are corrected in order (district, thana, post office). Each value comes from its label
("থানা: ..."); without one, district and thana are taken by position: the district is the last
comma-separated part, and once it is removed the thana is the last remaining part (city corporation /
pourashava parts skipped), as in NID addresses "..., ডাকঘর: X - ১২০৭, <thana>, <city corp>, <district>".
A field with neither falls back to the address left after removing the parts the other fields took. District is searched over all districts;
thana and post office have `filter_by="district"`, so they only search the records whose district
metadata matches the corrected district. For every field the candidates are ranked by
sliding_weight * sliding n-gram score + similarity_weight * vector similarity (60% / 40% by default).
"""

import unicodedata

from .config import Settings
from .embedding import Embedder
from .extraction import clean_text, extract_field, last_free_segment, split_segments
from .locations import FieldSpec, load_fields
from .rerank import rerank_candidates
from .text_units import char_units
from .trace import Trace
from .vector_store import LocationStore


class AddressCorrector:
    def __init__(self, embedder: Embedder, store: LocationStore, settings: Settings):
        self.embedder = embedder
        self.store = store
        self.settings = settings
        self.fields = store.fields

    @classmethod
    def from_settings(cls, settings: Settings, trace: Trace | None = None) -> "AddressCorrector":
        trace = trace or Trace()
        trace.log("startup", f"Loading embedding model '{settings.model_name}'{f' on {settings.device}' if settings.device else ''}")
        with trace.timed("load_model") as timer:
            embedder = Embedder(settings.model_name, settings.device, settings.model_path)
        trace.log("startup", f"Model loaded from {embedder.source} on {embedder.device} in {timer['ms'] / 1000:.1f} s (dimension {embedder.dimension})", level="success")

        fields = load_fields(settings.data_path)
        trace.log("startup", f"Loaded reference locations from {settings.data_path}: " + ", ".join(f"{len(f.unique_records)} {f.title.lower()}" for f in fields), data={f.collection: len(f.unique_records) for f in fields})

        trace.log("startup", f"Opening ChromaDB at {settings.chroma_path}")
        store = LocationStore(settings.chroma_path, embedder, fields)
        store.ensure_ingested(trace)
        return cls(embedder, store, settings)

    def reingest(self, trace: Trace | None = None, force: bool = True) -> dict:
        trace = trace or Trace()
        with trace.timed("total"):
            summary = self.store.ensure_ingested(trace, force=force)
        trace.log("done", f"Ingestion finished in {trace.timings['total']} ms", level="success")
        return {"collections": summary, "timings_ms": trace.timings, "logs": trace.events}

    def correct(self, address: str, n_results: int | None = None, padding: int | None = None,
                trace: Trace | None = None) -> dict:
        trace = trace or Trace()
        n_results = n_results or self.settings.n_results
        padding = self.settings.padding if padding is None else padding

        address = unicodedata.normalize("NFC", address).strip()
        trace.log("input", f"Received address ({len(address)} characters)", data={"address": address, "n_results": n_results, "padding": padding})

        with trace.timed("tokenize") as timer:
            address_units = char_units(address)
        trace.log("tokenize", f"Split address into {len(address_units)} Bangla character units (non-Bangla characters dropped) in {timer['ms']} ms", data={"preview": address_units[:40]})

        # 1. Pull each value out of the address: "label: value" first, else by position (district = last
        # part, thana = last part once the district is removed); query in the stored documents' format
        plans = []
        with trace.timed("extract"):
            segments = split_segments(address)
            labeled = {spec.collection: extract_field(address, spec.label) for spec in self.fields}
            used = [extracted for extracted in labeled.values() if extracted]
            for spec in self.fields:
                extracted, source = labeled[spec.collection], "label"
                if extracted is None and spec.positional:
                    extracted, source = last_free_segment(segments, used), "position"
                    if extracted:
                        used.append(extracted)  # removed from the address for the fields after this one
                if extracted:
                    trace.log("extract", f"[{spec.title}] Found '{extracted.value}' by {source}" + (f" ('{spec.label}:')" if source == "label" else " (last remaining part of the address)"), data={"value": extracted.value, "span": [extracted.start, extracted.end], "source": source})
                plans.append([spec, extracted, source])

            # Fields with neither a label nor a position search the address minus the parts already taken
            remaining = ", ".join(s.value for s in segments if not any(s.overlaps(u) for u in used)) or clean_text(address)
            for plan in plans:
                spec, extracted, _ = plan
                if extracted:
                    plan.append(spec.document(extracted.value))
                else:
                    plan[2] = "remaining"
                    plan.append(remaining)
                    trace.log("extract", f"[{spec.title}] No '{spec.label}:' label in the address; falling back to the remaining address as query", level="warning", data={"remaining": remaining})

        # 2. Embed every query in one batch (the queries don't depend on each other, only the search filters do)
        with trace.timed("embed") as timer:
            embeddings = self.embedder.encode([query_text for *_, query_text in plans])
        trace.log("embed", f"Encoded {len(plans)} queries with {self.embedder.model_name} in {timer['ms']} ms", data={"queries": [query_text for *_, query_text in plans], "dimension": int(embeddings.shape[1])})

        fields_out = []
        by_collection: dict[str, dict] = {}
        for (spec, extracted, source, query_text), embedding in zip(plans, embeddings):
            # 3. Vector search, filtered by an already corrected field when the spec asks for it
            where, filter_fallback = self._filter_for(spec, by_collection, trace)
            total = self.store.count(spec)
            with trace.timed("vector_search") as timer:
                searched = self.store.count(spec, where)
                if where and not searched:
                    filter_fallback = f"no records match {where}"
                    trace.log("filter", f"[{spec.title}] No records with {where}; searching all {total} records", level="warning")
                    where, searched = None, total
                candidates = self.store.query(spec, embedding, n_results, where=where)
            nearest = candidates[0] if candidates else None
            trace.log(
                "vector_search",
                f"[{spec.title}] Searched {searched} of {total} records"
                + (f" ({', '.join(f'{k}={v}' for k, v in where.items())})" if where else "")
                + f", retrieved {len(candidates)} candidates in {timer['ms']} ms"
                + (f"; nearest '{nearest['name']}' (distance {nearest['distance']:.4f})" if nearest else ""),
                data={"where": where, "searched": searched, "total": total, "embedding": {"dimension": int(embeddings.shape[1])}, "top_5": [{"name": c["name"], "distance": round(c["distance"], 4)} for c in candidates[:5]]},
            )

            # 4. Character n-gram sliding of the candidate names over the extracted value
            # (the remaining address when there is no value)
            target_units = char_units(extracted.value if extracted else query_text)
            with trace.timed("rerank") as timer:
                ranked = rerank_candidates(target_units, candidates, padding,
                                           self.settings.sliding_weight, self.settings.similarity_weight)
            best = ranked[0] if ranked else None
            trace.log(
                "rerank",
                f"[{spec.title}] Re-ranked {len(ranked)} candidates in {timer['ms']} ms"
                + (f"; best '{best['name']}' (score {best['score']:.3f} = {self.settings.sliding_weight:g} x sliding {best['sliding_score']:.3f}"
                   f" + {self.settings.similarity_weight:g} x similarity {best['similarity']:.3f})" if best else ""),
                data={"top_5": [{"name": r["name"], "score": round(r["score"], 3), "sliding_score": round(r["sliding_score"], 3), "similarity": round(r["similarity"], 3), "vector_rank": r["vector_rank"]} for r in ranked[:5]]},
            )

            winner_changed = bool(best and nearest and best["name"] != nearest["name"])
            if winner_changed:
                trace.log("rerank", f"[{spec.title}] Re-ranking overrode the vector search: '{nearest['name']}' (vector #1) -> '{best['name']}' (was vector #{best['vector_rank']})", level="warning")

            changed = bool(best and extracted and extracted.value != best["name"])
            if not best:
                trace.log("decision", f"[{spec.title}] No candidates, field left unchanged", level="error")
            elif not extracted:
                trace.log("decision", f"[{spec.title}] Best guess '{best['name']}' (not substituted, field not found in the address)", level="warning")
            elif changed:
                trace.log("decision", f"[{spec.title}] Corrected '{extracted.value}' -> '{best['name']}'", level="success")
            else:
                trace.log("decision", f"[{spec.title}] '{extracted.value}' is already correct", level="success")

            field_out = {
                "collection": spec.collection,
                "label": spec.label,
                "title": spec.title,
                "extracted": extracted.value if extracted else None,
                "span": [extracted.start, extracted.end] if extracted else None,
                "query_text": query_text,
                "used_fallback": extracted is None,
                "source": source,  # where the value came from: "label", "position" or "remaining" (fallback)
                "filter": where,  # metadata filter applied to the vector search, e.g. {"district": "ঢাকা"}
                "filter_fallback": filter_fallback,  # why a wanted filter was not applied
                "searched": searched,
                "total": total,
                "corrected": best["name"] if best else None,
                "changed": changed,
                "score": best["score"] if best else None,
                "sliding_score": best["sliding_score"] if best else None,
                "similarity": best["similarity"] if best else None,
                "distance": best["distance"] if best else None,
                "vector_top": nearest["name"] if nearest else None,
                "rerank_changed_winner": winner_changed,
                "vector_candidates": candidates,
                "reranked": ranked,
            }
            fields_out.append(field_out)
            by_collection[spec.collection] = field_out

        # 5. Rebuild the address with the corrected values in place
        corrected_address, segments = compose_address(address, fields_out)
        trace.log("compose", f"Applied {sum(f['changed'] for f in fields_out)} correction(s) to the address",
                  data={"corrected_address": corrected_address})

        trace.timings["total"] = trace.elapsed_ms
        trace.log("done", f"Finished in {trace.timings['total']} ms", level="success")

        return {
            "input_address": address,
            "corrected_address": corrected_address,
            "segments": segments,
            "corrections": {f["collection"]: f["corrected"] for f in fields_out},
            "fields": fields_out,
            "address_units": address_units,
            "params": {"n_results": n_results, "padding": padding, "model": self.embedder.model_name,
                       "sliding_weight": self.settings.sliding_weight, "similarity_weight": self.settings.similarity_weight},
            "timings_ms": trace.timings,
            "logs": trace.events,
        }

    @staticmethod
    def _filter_for(spec: FieldSpec, by_collection: dict[str, dict], trace: Trace) -> tuple[dict | None, str | None]:
        # Returns (metadata filter, reason the wanted filter was skipped)
        if not spec.filter_by:
            return None, None
        parent = by_collection.get(spec.filter_by)
        if parent is None or parent["corrected"] is None:
            reason = f"{spec.filter_by} was not corrected"
        else:
            trace.log("filter", f"[{spec.title}] Filtering by {spec.filter_by} = '{parent['corrected']}'",
                      data={"where": {spec.filter_by: parent["corrected"]}})
            return {spec.filter_by: parent["corrected"]}, None
        trace.log("filter", f"[{spec.title}] Not filtering by {spec.filter_by} ({reason}); searching all records",
                  level="warning")
        return None, reason


def compose_address(address: str, fields: list[dict]) -> tuple[str, list[dict]]:
    # Split the address into plain text and field segments so the UI can highlight substitutions
    spans = sorted((f for f in fields if f["span"] and f["corrected"]), key=lambda f: f["span"][0])
    segments, cursor = [], 0
    for field in spans:
        start, end = field["span"]
        if start < cursor:  # overlapping spans should not happen, but never duplicate text
            continue
        if start > cursor:
            segments.append({"text": address[cursor:start]})
        segments.append({
            "text": field["corrected"],
            "original": field["extracted"],
            "collection": field["collection"],
            "changed": field["changed"],
        })
        cursor = end
    if cursor < len(address):
        segments.append({"text": address[cursor:]})
    return "".join(segment["text"] for segment in segments), segments
