"""Batch NER taggers for dgiLIT.

The HuggingFace pipelines are lazy-loaded and called in batches. Inputs are
sanitized so missing abstracts, NaN values, and non-string objects do not crash
transformers token-classification pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Iterable, Any
from tqdm.auto import tqdm

from .models import EntityMention, EntityType


@dataclass(frozen=True)
class TaggerConfig:
    batch_size: int = 16
    include_drugs: bool = True
    include_genes: bool = True
    include_diseases: bool = False
    aggregation_strategy: str = "first"
    device: int = -1
    num_workers: int = 0


class BioBertEntityTagger:
    """Lazy-loaded BioBERT entity tagger compatible with old novel.py models."""

    MODEL_BY_TYPE: dict[EntityType, str] = {
        "gene": "alvaroalon2/biobert_genetic_ner",
        "drug": "alvaroalon2/biobert_chemical_ner",
        "disease": "alvaroalon2/biobert_diseases_ner",
    }

    def __init__(self, config: TaggerConfig | None = None) -> None:
        self.config = config or TaggerConfig()
        self._pipelines = {}

    def tag_texts(self, texts: list[Any]) -> list[list[EntityMention]]:
        """Return entity mentions for each input text, preserving input order.

        Blank or invalid inputs return an empty mention list. This avoids the
        common transformers error: "sentence must be an untokenized string".
        """
        clean_texts = [_coerce_text(text) for text in texts]
        mentions_by_text: list[list[EntityMention]] = [[] for _ in clean_texts]

        # Only send non-empty strings into HuggingFace. Keep original positions.
        valid_items = [(idx, text) for idx, text in enumerate(clean_texts) if text.strip()]
        if not valid_items:
            return mentions_by_text

        for entity_type in self._enabled_types():
            pipe = self._get_pipeline(entity_type)
            for offset in tqdm(
                range(0, len(texts), self.config.batch_size),
                total=(len(texts) + self.config.batch_size - 1) // self.config.batch_size,
                desc="Tagging text batches",
            ):                
                batch_items = valid_items[offset : offset + self.config.batch_size]
                batch_indices = [idx for idx, _ in batch_items]
                batch_texts = [text for _, text in batch_items]

                raw_batch = pipe(
                    batch_texts,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers,
                )

                # transformers returns list[dict] for one string, list[list[dict]] for many strings.
                if batch_texts and raw_batch and isinstance(raw_batch[0], dict):
                    raw_batch = [raw_batch]

                for original_idx, raw_mentions in zip(batch_indices, raw_batch):
                    mentions_by_text[original_idx].extend(
                        self._convert_mentions(raw_mentions, entity_type)
                    )
        return mentions_by_text

    def tag_text(self, text: str) -> list[EntityMention]:
        return self.tag_texts([text])[0]

    def _enabled_types(self) -> Iterable[EntityType]:
        if self.config.include_drugs:
            yield "drug"
        if self.config.include_genes:
            yield "gene"
        if self.config.include_diseases:
            yield "disease"

    def _get_pipeline(self, entity_type: EntityType):
        if entity_type not in self._pipelines:
            from transformers import pipeline

            self._pipelines[entity_type] = pipeline(
                "token-classification",
                model=self.MODEL_BY_TYPE[entity_type],
                aggregation_strategy=self.config.aggregation_strategy,
                device=self.config.device,
            )
        return self._pipelines[entity_type]

    @staticmethod
    def _convert_mentions(raw_mentions: list[dict], entity_type: EntityType) -> list[EntityMention]:
        converted: list[EntityMention] = []
        for raw in raw_mentions or []:
            entity_group = str(raw.get("entity_group", ""))
            if entity_group == "0":
                continue
            word = raw.get("word") or raw.get("text")
            if not word:
                continue
            converted.append(
                EntityMention(
                    text=str(word),
                    entity_type=entity_type,
                    start=int(raw.get("start", 0)),
                    end=int(raw.get("end", 0)),
                    score=raw.get("score"),
                    source=f"biobert:{BioBertEntityTagger.MODEL_BY_TYPE[entity_type]}",
                )
            )
        return converted


def _coerce_text(value: Any) -> str:
    """Convert dataframe values into safe text for token-classification."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        try:
            if isnan(value):
                return ""
        except TypeError:
            pass
    # pandas.NA / numpy.nan often behave badly in boolean contexts.
    try:
        import pandas as pd  # type: ignore

        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)
