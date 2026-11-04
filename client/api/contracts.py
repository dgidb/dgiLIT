"""Validated HTTP responses; raw saved records remain lossless JSON objects."""
from typing import Any, Literal
from pydantic import BaseModel


class Health(BaseModel):
    status: Literal["ok"]
    access: Literal["read_only"]
    dataset: str


class CohortCounts(BaseModel):
    cohort: Literal["pilot", "beyond_pilot"]
    positive_contexts: int
    supporting_articles: int


class Summary(BaseModel):
    read_at: str
    dataset: str
    cohorts: list[CohortCounts]
    positive_contexts: int
    supporting_articles: int
    metadata: dict[str, Any]


class ContextRecord(BaseModel):
    task_id: str
    pmcid: str
    pmid: str | None
    title: str | None
    publication_year: str | None
    coverage: str
    article_status: str
    drug: str
    drug_id: str
    gene: str
    gene_id: str
    updated_at: str
    evidence: str | None
    evidence_verified: bool | None
    interaction_type: str | None
    directionality: str | None
    section_label: str | None
    section_type: str | None
    cohort: Literal["pilot", "beyond_pilot"]


class ContextPage(BaseModel):
    items: list[ContextRecord]
    total: int
    page: int
    page_size: int
    read_at: str


class PairCounts(BaseModel):
    status: str
    contexts: int
    positive: int
    negative: int


class ContextDetail(BaseModel):
    task: dict[str, Any]
    entities: list[dict[str, Any]]
    related_contexts: list[PairCounts]
    metadata: dict[str, Any]
    read_at: str
    dataset: str
    prompt: dict[str, Any]
    links: dict[str, str]


class SourceResult(BaseModel):
    state: Literal["verified", "unavailable", "hash_mismatch", "retrieval_error"]
    origin: str
    context: str | None
    read_at: str
    message: str
    section_hash: str | None = None
    context_hash: str | None = None
