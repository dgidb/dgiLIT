"""Prompt helpers for wiring pre-tagged entities into WAGS LLM payloads."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dgilit_wags.tagging.models import InteractionExtractionInput, TaggedTextBlock


class InteractionExtractionResult(BaseModel):
    """Minimal result model matching the current wags.ipynb prototype."""

    model_config = ConfigDict(extra="forbid")

    drug: str | None = None
    gene: str | None = None
    interaction: bool | None = None
    evidence: str | None = None
    error_message: str | None = None


class InteractionExtractionPrompt(BaseModel):
    """WAGS-compatible prompt object with pre-tagged entity support."""

    name: str = "dgilit_interaction_extraction"
    version: str = "v1"
    include_diseases: bool = False

    def build_payload_from_tagged_block(self, block: TaggedTextBlock) -> dict[str, str]:
        return self.build_payload(InteractionExtractionInput.from_tagged_block(block))

    def build_payload(self, extraction_input: InteractionExtractionInput | str, **kwargs) -> dict[str, str]:
        """Build payload for StructuredTaskRunner.

        Accepts either the new InteractionExtractionInput or a raw context string
        for backward compatibility with the current notebook.
        """
        if isinstance(extraction_input, str):
            extraction_input = InteractionExtractionInput(context=extraction_input, **kwargs)

        entity_block = TaggedTextBlock(
            context=extraction_input.context,
            pmid=extraction_input.pmid,
            block_id=extraction_input.block_id,
            entities=extraction_input.tagged_entities,
        ).prompt_entity_block(include_diseases=self.include_diseases)

        instruction = (
            "You are a scientific biocurator extracting drug-gene interactions for DGIdb. "
            "Use the pre-identified entities as candidates, but only report an interaction when "
            "the context directly supports it. Do not invent drugs, genes, or evidence. "
            "Return JSON matching the requested response schema."
        )

        return {
            "instruction": instruction,
            "pmid": extraction_input.pmid or "",
            "block_id": extraction_input.block_id or "",
            "preidentified_entities": entity_block,
            "context": extraction_input.context,
        }
