from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator
from wags_llm.prompts import BasePromptTemplate
from .references import REF_DIR, assemble_system_prompt, load_vocabulary


class InteractionClassificationResult(BaseModel):
    """LLM result for classifying one candidate drug-gene pair."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    drug: str
    gene: str
    interaction: bool
    evidence: str | None = None
    interaction_type: str | None = None
    directionality: str | None = None
    error_message: str | None = None


class RunResult:
    temperature: float
    run_idx: int
    pass


class _V3Result(InteractionClassificationResult):
    interaction: bool = Field(strict=True)

    @model_validator(mode="after")
    def check_decision(self):
        if self.error_message is not None:
            raise ValueError("Classification reports an error")
        if not self.interaction:
            if self.interaction_type is not None or self.directionality is not None:
                raise ValueError("Negative decisions require null type and directionality")
        else:
            if not self.evidence or not self.evidence.strip():
                raise ValueError("Positive decisions require evidence")
            if self.interaction_type is None or self.directionality is None:
                raise ValueError("Positive decisions require type and directionality")
            expected = load_vocabulary()[self.interaction_type]["directionality"]
            if expected != "unclear" and self.directionality != expected:
                raise ValueError("Interaction type and directionality disagree")
        return self


def classification_result_model(version: str) -> type[InteractionClassificationResult]:
    """Select the strict v3 contract without changing legacy parsing."""
    if version != "v3":
        return InteractionClassificationResult
    labels = tuple(sorted(load_vocabulary()))
    return create_model(
        "InteractionClassificationResultV3", __base__=_V3Result,
        interaction_type=(Literal[labels] | None, ...),
        directionality=(Literal["activating", "inhibiting", "unclear"] | None, ...),
    )


def validate_classification(value, *, version: str, context: str,
                            candidate_drug: str, candidate_gene: str):
    """Validate a result against its source; shared by pipelines and direct runners."""
    if isinstance(value, BaseModel):
        value = value.model_dump()
    result = classification_result_model(version).model_validate(value)
    if version == "v3" and (result.drug != candidate_drug or result.gene != candidate_gene):
        raise ValueError("Classification changed candidate identity")
    evidence = " ".join((result.evidence or "").split()).strip('\"“”')
    verified = bool(evidence) and evidence.casefold() in " ".join(context.split()).casefold()
    if result.interaction and not verified:
        raise ValueError("Positive classification lacks a verifiable direct evidence quote")
    if version == "v3" and result.evidence and not verified:
        raise ValueError("Evidence is not a direct quote from the context")
    return result


PROMPT_NAME = "dgilit_interaction_classification"


@dataclass
class InteractionClassificationPrompt(BasePromptTemplate):
    """Prompt for classifying whether one drug-gene pair is supported by text."""

    name = PROMPT_NAME
    PROMPT_DIR = Path(__file__).parent / "dgilit_prompts"
    REF_DIR = REF_DIR

    def __init__(self, version: str):
        self.version = version

    @property
    def prompt_path(self) -> Path:
        return self.PROMPT_DIR / f"{self.version}.text"

    def build_system_prompt(self) -> str:
        if not self.prompt_path.exists():
            available_versions = sorted(
                p.stem for p in self.PROMPT_DIR.glob("*.text")
            )

            raise FileNotFoundError(
                f"Prompt version '{self.version}' not found. "
                f"Available versions: {available_versions}"
            )

        return assemble_system_prompt(self.version, self.PROMPT_DIR, self.REF_DIR)

    def build_user_prompt(self, payload: Mapping[str, Any]) -> str:
        return (
            "Candidate Drug:\n"
            f"{payload['candidate_drug']}\n\n"
            "Candidate Gene:\n"
            f"{payload['candidate_gene']}\n\n"
            "Context:\n"
            f"{payload['context']}\n"
        )

    def build_payload(
        self,
        context: str,
        candidate_drug: str,
        candidate_gene: str,
    ) -> dict[str, Any]:
        return {
            "context": context,
            "candidate_drug": candidate_drug,
            "candidate_gene": candidate_gene,
        }
