from pydantic import BaseModel, ConfigDict
from wags_llm.prompts import BasePromptTemplate
from pathlib import Path
from typing import Any
from collections.abc import Mapping
from dataclasses import dataclass

# from dgilit_models import InteractionPredictionResult
# from dgilit_models import InteractionPredictionPrompt

class InteractionExtractionResult(BaseModel):
    """Model for LLM result in extracting an interaction from a block of text (context)"""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    drug: str | None = None
    gene: str | None = None
    interaction: bool | None = None
    error_message: str | None = None


class RunResult:
    temperature: float
    run_idx: int
    pass


PROMPT_NAME = 'dgilit_interaction_identification'

@dataclass
class InteractionExtractionPrompt(BasePromptTemplate):
    """Prompt for identifying a drug-gene interaction from a piece of text in a published PMID"""
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

    def build_user_prompt(self, payload: Mapping[str, Any],) -> str:
        """
        Build user prompt for a single block of text to identify interactions
        
        :param payload: 
        :returns: User prompt text
        """
        return (
            # f"Drug Name: {payload['drug_name']}\n"
            # f"Gene Name: {payload['gene_name']}\n"
            f"Context: {payload['context']}\n"
        )
    def build_payload(self, context: str) -> dict:
        return {
            # "drug_name": drug_name,
            # "gene_name": gene_name,
            "context": context
        }


    pass


