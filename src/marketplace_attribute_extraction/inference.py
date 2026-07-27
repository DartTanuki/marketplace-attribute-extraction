"""End-to-end marketplace attribute extraction."""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fasttext
import pandas as pd
import torch
from gliner2 import GLiNER2

from .normalization import normalize_id, normalize_query_for_fasttext
from .shortlist import GENERIC_ATTRIBUTES, select_attribute_shortlist


@dataclass(frozen=True)
class ArtifactPaths:
    fasttext_model: Path
    category_attributes: Path
    attribute_candidates: Path | None = None
    category_meta: Path | None = None

    @classmethod
    def from_directory(cls, directory: str | Path) -> "ArtifactPaths":
        directory = Path(directory)
        return cls(
            fasttext_model=directory / "subject_classifier_fasttext.bin",
            category_attributes=directory / "category_attributes_by_subject_id.pkl",
            attribute_candidates=directory / "gliner_attribute_candidates.pkl",
            category_meta=directory / "category_meta.pkl",
        )


class ProductAttributeExtractor:
    """fastText category classifier + dynamic shortlist + base GLiNER2."""

    def __init__(
        self,
        fasttext_model: Any,
        gliner_extractor: Any,
        category_attributes: dict[str, list[str]],
        *,
        attribute_priority_lookup: dict[str, dict[str, dict[str, float]]] | None = None,
        category_meta: dict[str, dict[str, Any]] | None = None,
        shortlist_size: int | None = 5,
        gliner_threshold: float = 0.2,
        category_top_k: int = 3,
    ) -> None:
        self.fasttext_model = fasttext_model
        self.gliner_extractor = gliner_extractor
        self.category_attributes = {
            str(normalize_id(category_id)): list(dict.fromkeys(attributes))
            for category_id, attributes in category_attributes.items()
        }
        self.attribute_priority_lookup = attribute_priority_lookup or {}
        self.category_meta = category_meta or {}
        self.shortlist_size = shortlist_size
        self.gliner_threshold = float(gliner_threshold)
        self.category_top_k = int(category_top_k)

    @classmethod
    def from_artifacts(
        cls,
        artifacts: ArtifactPaths | str | Path,
        *,
        base_model: str = "fastino/gliner2-multi-v1",
        device: str | None = None,
        shortlist_size: int | None = 5,
        gliner_threshold: float = 0.2,
        category_top_k: int = 3,
    ) -> "ProductAttributeExtractor":
        paths = (
            artifacts
            if isinstance(artifacts, ArtifactPaths)
            else ArtifactPaths.from_directory(artifacts)
        )

        required = [paths.fasttext_model, paths.category_attributes]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required artifacts: " + ", ".join(missing))

        fasttext_model = fasttext.load_model(str(paths.fasttext_model))
        with paths.category_attributes.open("rb") as file:
            category_attributes = pickle.load(file)

        category_meta: dict[str, dict[str, Any]] = {}
        if paths.category_meta and paths.category_meta.exists():
            with paths.category_meta.open("rb") as file:
                category_meta = pickle.load(file)

        priority_lookup: dict[str, dict[str, dict[str, float]]] = {}
        if paths.attribute_candidates and paths.attribute_candidates.exists():
            frame = pd.read_pickle(paths.attribute_candidates).copy()
            for row in frame.itertuples(index=False):
                category_id = str(normalize_id(row.category_id))
                attribute_name = str(row.attribute_name)
                priority_lookup.setdefault(category_id, {})[attribute_name] = {
                    "rank": int(getattr(row, "attribute_rank", 10_000)),
                    "score": float(getattr(row, "score", 0.0)),
                    "coverage": float(getattr(row, "coverage", 0.0)),
                }

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        gliner_extractor = GLiNER2.from_pretrained(
            base_model,
            map_location=device,
        )
        return cls(
            fasttext_model=fasttext_model,
            gliner_extractor=gliner_extractor,
            category_attributes=category_attributes,
            attribute_priority_lookup=priority_lookup,
            category_meta=category_meta,
            shortlist_size=shortlist_size,
            gliner_threshold=gliner_threshold,
            category_top_k=category_top_k,
        )

    def predict_category(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        normalized_query = normalize_query_for_fasttext(query)
        labels, probabilities = self.fasttext_model.predict(
            normalized_query,
            k=k or self.category_top_k,
        )
        return [
            {
                "rank": rank,
                "category_id": normalize_id(label.removeprefix("__label__")),
                "confidence": float(probability),
            }
            for rank, (label, probability) in enumerate(
                zip(labels, probabilities), start=1
            )
        ]

    @staticmethod
    def _flatten_facts(facts: Any) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        if not isinstance(facts, dict):
            return flattened
        for attribute_name, entities in facts.items():
            if not isinstance(entities, list):
                entities = [entities]
            for entity in entities:
                if isinstance(entity, dict):
                    flattened.append(
                        {
                            "attribute": attribute_name,
                            "value": entity.get("text"),
                            "confidence": entity.get("confidence"),
                            "start": entity.get("start"),
                            "end": entity.get("end"),
                        }
                    )
                else:
                    flattened.append(
                        {
                            "attribute": attribute_name,
                            "value": entity,
                            "confidence": None,
                            "start": None,
                            "end": None,
                        }
                    )
        return flattened

    @staticmethod
    def _synchronize_cuda() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def predict(
        self,
        query: str,
        *,
        shortlist_size: int | None = None,
        gliner_threshold: float | None = None,
        category_top_k: int | None = None,
    ) -> dict[str, Any]:
        if query is None or not str(query).strip():
            raise ValueError("query must not be empty")

        shortlist_size = self.shortlist_size if shortlist_size is None else shortlist_size
        gliner_threshold = (
            self.gliner_threshold if gliner_threshold is None else float(gliner_threshold)
        )
        category_top_k = (
            self.category_top_k if category_top_k is None else int(category_top_k)
        )

        self._synchronize_cuda()
        started_at = time.perf_counter()

        category_predictions = self.predict_category(query, k=category_top_k)
        top_prediction = category_predictions[0]
        category_id = str(top_prediction["category_id"])

        all_attributes = self.category_attributes.get(category_id)
        used_fallback_schema = not bool(all_attributes)
        if not all_attributes:
            all_attributes = list(GENERIC_ATTRIBUTES)

        attributes, ranking = select_attribute_shortlist(
            query=query,
            category_id=category_id,
            attributes=all_attributes,
            shortlist_size=shortlist_size,
            attribute_priority_lookup=self.attribute_priority_lookup,
        )

        raw_result = self.gliner_extractor.extract_entities(
            str(query),
            attributes,
            threshold=gliner_threshold,
            include_confidence=True,
            include_spans=True,
        )
        facts = raw_result.get("entities", raw_result) if isinstance(raw_result, dict) else {}

        self._synchronize_cuda()
        latency_ms = (time.perf_counter() - started_at) * 1_000

        category_name = (
            self.category_meta.get(category_id, {}).get("category_name")
            if isinstance(self.category_meta, dict)
            else None
        )

        return {
            "query": str(query),
            "normalized_query": normalize_query_for_fasttext(query),
            "category_id": category_id,
            "category_name": category_name,
            "category_confidence": top_prediction["confidence"],
            "category_top_k": category_predictions,
            "used_fallback_schema": used_fallback_schema,
            "all_candidate_attributes": all_attributes,
            "candidate_attributes": attributes,
            "n_all_candidate_attributes": len(all_attributes),
            "n_candidate_attributes": len(attributes),
            "shortlist_size": shortlist_size,
            "shortlist_ranking": ranking,
            "facts": facts,
            "flat_facts": self._flatten_facts(facts),
            "latency_ms": latency_ms,
        }
