"""WAGS-backed interaction extraction helpers for the evaluation notebook."""

import json
import logging
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict
from utils import clean_value, collapse_ws
from wags_llm.cache import InMemoryCache
from wags_llm.client import BedrockClaudeJsonClient
from wags_llm.prompts import BasePromptTemplate, PromptRegistry
from wags_llm.services import StructuredTaskRunner

logger = logging.getLogger(__name__)


class ExtractedInteraction(BaseModel):
    """One extracted inhibitor or activator interaction."""

    model_config = ConfigDict(extra="forbid")

    pmid: str
    drug_name: str
    gene_name: str
    interaction_type: Literal["inhibitor", "activator"]


class InteractionExtractionResponse(BaseModel):
    """Structured response containing extracted interactions."""

    model_config = ConfigDict(extra="forbid")

    interactions: list[ExtractedInteraction]


def make_prompt_template(prompt_name: str, prompt_version: str) -> BasePromptTemplate:
    """Build a WAGS prompt template for PubMed extraction tasks."""

    class Prompt(BasePromptTemplate):
        name = prompt_name
        version = prompt_version

        def build_system_prompt(self) -> str:
            return "Extract inhibitor or activator drug-gene interactions from the supplied PubMed title and abstract. Return only JSON matching the schema."

        def build_user_prompt(self, payload: Mapping[str, Any]) -> str:
            return f"""
Return {{"interactions": []}} if no explicitly supported inhibitor or activator drug-gene interaction is present.
Each interaction must include pmid, drug_name, gene_name, and interaction_type, where interaction_type is exactly "inhibitor" or "activator".
Do not infer interactions from pathway context alone.

PMID: {clean_value(payload.get("pmid"))}
Article title: {collapse_ws(payload.get("article_title", ""))}

Abstract:
{str(payload.get("abstract") or "").strip()}
""".strip()

    return Prompt()


def make_prompt_registry(prompt_name: str, prompt_version: str) -> PromptRegistry:
    """Register and return the evaluation prompt template."""
    registry = PromptRegistry()
    registry.register(make_prompt_template(prompt_name, prompt_version))
    return registry


def initialize_wags_client(
    model_id: str,
    bedrock_region: str,
    aws_profile_name: str,
    max_tokens: int,
    temperature: float,
) -> BedrockClaudeJsonClient:
    """Create the Bedrock-backed WAGS JSON client."""
    return BedrockClaudeJsonClient(
        model_id=model_id,
        region_name=bedrock_region,
        profile_name=aws_profile_name,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def make_wags_runner(
    client: BedrockClaudeJsonClient,
    prompt_registry: PromptRegistry,
    cache_max_entries: int,
) -> StructuredTaskRunner:
    """Create a structured WAGS task runner with an in-memory cache."""
    cache = InMemoryCache(max_entries=cache_max_entries)
    return StructuredTaskRunner(
        client=client, prompt_registry=prompt_registry, cache=cache
    )


def execute_wags_task(
    runner: StructuredTaskRunner,
    payload: Mapping[str, Any],
    prompt_name: str,
    prompt_version: str,
) -> InteractionExtractionResponse:
    """Execute one WAGS extraction task and validate the response model."""
    result = runner.execute(
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        payload=payload,
        response_model=InteractionExtractionResponse,
    )
    return InteractionExtractionResponse.model_validate(result)


def interaction_rows(
    response: InteractionExtractionResponse, allowed_pmids: Iterable[str]
) -> list[dict[str, str]]:
    """Convert a structured WAGS response into filtered interaction rows."""
    allowed = {str(pmid).strip() for pmid in allowed_pmids}
    rows = []
    for item in response.interactions:
        row = item.model_dump()
        row = {
            "pmid": clean_value(row["pmid"]),
            "drug_name": clean_value(row["drug_name"]),
            "gene_name": clean_value(row["gene_name"]),
            "interaction_type": clean_value(row["interaction_type"]).lower(),
        }
        if row["pmid"] in allowed and row["drug_name"] and row["gene_name"]:
            rows.append(row)
    return rows


def load_predictions(path: str | Path) -> dict[str, Any]:
    """Load saved prediction records from JSON if present."""
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def save_predictions(predictions: dict[str, Any], path: str | Path) -> Path:
    """Persist prediction records as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    return path


def prediction_status_frame(predictions: dict[str, Any]) -> pd.DataFrame:
    """Summarize prediction success, interaction counts, and errors."""
    return pd.DataFrame(
        [
            {
                "pmid": pmid,
                "successful": isinstance(payload, dict)
                and not payload.get("error")
                and "interactions" in payload,
                "parsed_interactions": len(payload.get("interactions", []))
                if isinstance(payload, dict)
                else 0,
                "error": payload.get("error", "")
                if isinstance(payload, dict)
                else "Invalid prediction record",
            }
            for pmid, payload in sorted(predictions.items())
        ]
    )


def run_wags_predictions(
    payloads: list[dict[str, str]],
    predictions_path: str | Path,
    enable_llm_calls: bool,
    model_key: str,
    model_id: str,
    prompt_name: str,
    prompt_version: str,
    aws_profile_name: str,
    bedrock_region: str,
    max_tokens: int,
    temperature: float,
    cache_max_entries: int = 500,
    overwrite: bool = False,
    stop_on_error: bool = False,
    request_sleep_seconds: float = 0.25,
    allowed_pmids: Iterable[str] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Run or resume WAGS predictions for task payloads."""
    predictions = load_predictions(predictions_path)
    selected_pmids = (
        [payload["pmid"] for payload in payloads]
        if allowed_pmids is None
        else allowed_pmids
    )
    allowed = {clean_value(pmid) for pmid in selected_pmids}

    if not enable_llm_calls:
        logger.info("ENABLE_LLM_CALLS is False. No Bedrock calls were made.")
        return predictions

    client = initialize_wags_client(
        model_id, bedrock_region, aws_profile_name, max_tokens, temperature
    )
    registry = make_prompt_registry(prompt_name, prompt_version)
    runner = make_wags_runner(client, registry, cache_max_entries)

    for index, payload in enumerate(payloads, start=1):
        pmid = str(payload["pmid"])
        if (
            pmid in predictions
            and "interactions" in predictions[pmid]
            and not overwrite
        ):
            if progress:
                logger.info(
                    "Skipping cached WAGS prediction %s of %s", index, len(payloads)
                )
            continue
        if progress:
            logger.info("Running WAGS prediction %s of %s", index, len(payloads))
        try:
            parsed = execute_wags_task(runner, payload, prompt_name, prompt_version)
            predictions[pmid] = {
                "model_key": model_key,
                "model_id": model_id,
                "prompt_name": prompt_name,
                "prompt_version": prompt_version,
                "interactions": interaction_rows(parsed, allowed),
            }
        except Exception as exc:
            predictions[pmid] = {
                "model_key": model_key,
                "model_id": model_id,
                "prompt_name": prompt_name,
                "prompt_version": prompt_version,
                "error": str(exc),
            }
            logger.exception("WAGS prediction failed for PMID %s", pmid)
            if stop_on_error:
                raise
        save_predictions(predictions, predictions_path)
        if request_sleep_seconds:
            time.sleep(request_sleep_seconds)
    return predictions


def raw_audit_wags_response(
    payload: Mapping[str, Any],
    client: BedrockClaudeJsonClient,
    prompt_registry: PromptRegistry,
    prompt_name: str,
    prompt_version: str,
) -> dict[str, Any]:
    """Return raw and parsed WAGS output for one audit payload."""
    prompt = prompt_registry.get(prompt_name, prompt_version)
    response = client.invoke_json(
        system_prompt=prompt.build_system_prompt(),
        user_prompt=prompt.build_user_prompt(payload),
        json_schema=InteractionExtractionResponse.model_json_schema(),
    )
    parsed = InteractionExtractionResponse.model_validate(response.parsed_json)
    return {
        "pmid": str(payload.get("pmid", "")),
        "raw_response": response.raw_text,
        "parsed_json": response.parsed_json,
        "interactions": interaction_rows(parsed, {str(payload.get("pmid", ""))}),
    }
