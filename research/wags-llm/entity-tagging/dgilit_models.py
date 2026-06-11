from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from wags_llm.prompts import BasePromptTemplate


class ExtractedInteraction(BaseModel):
    """One extracted drug-gene interaction from a text block."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    drug: str
    gene: str
    interaction: bool = True
    evidence: str | None = None
    interaction_type: str | None = None
    directionality: str | None = None


class InteractionExtractionResult(BaseModel):
    """LLM result for extracting zero, one, or many interactions from context."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    interactions: list[ExtractedInteraction] = Field(default_factory=list)
    error_message: str | None = None


class RunResult:
    temperature: float
    run_idx: int
    pass


PROMPT_NAME = "dgilit_interaction_identification"


@dataclass
class InteractionExtractionPrompt(BasePromptTemplate):
    """Prompt for identifying drug-gene interactions from biomedical text."""

    name = PROMPT_NAME
    PROMPT_DIR = Path(__file__).parent / "dgilit_prompts"

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

        return self.prompt_path.read_text()

    def build_user_prompt(self, payload: Mapping[str, Any]) -> str:
        candidate_drugs = payload.get("candidate_drugs", [])
        candidate_genes = payload.get("candidate_genes", [])

        return (
            "Candidate drugs:\n"
            f"{self._format_candidates(candidate_drugs)}\n\n"
            "Candidate genes:\n"
            f"{self._format_candidates(candidate_genes)}\n\n"
            "Context:\n"
            f"{payload['context']}\n"
        )

    def build_payload(
        self,
        context: str,
        candidate_drugs: list[str] | None = None,
        candidate_genes: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "context": context,
            "candidate_drugs": candidate_drugs or [],
            "candidate_genes": candidate_genes or [],
        }

    @staticmethod
    def _format_candidates(candidates: list[str]) -> str:
        if not candidates:
            return "- None identified"

        return "\n".join(f"- {candidate}" for candidate in candidates)