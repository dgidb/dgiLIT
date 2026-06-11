"""Example notebook usage for efficient pre-tagging before WAGS LLM calls."""

from __future__ import annotations

from tqdm.notebook import tqdm

from dgilit_wags.llm import InteractionExtractionPrompt, InteractionExtractionResult
from dgilit_wags.tagging import (
    BioBertEntityTagger,
    EntityPreTaggingService,
    SQLiteNormalizerCache,
    TaggerConfig,
    ViccNormalizer,
)


def build_pretagging_service(cache_path: str = ".dgilit_normalizer_cache.sqlite") -> EntityPreTaggingService:
    tagger = BioBertEntityTagger(
        TaggerConfig(
            batch_size=16,
            include_drugs=True,
            include_genes=True,
            include_diseases=False,
            device=-1,  # set to 0 on a CUDA machine
        )
    )
    normalizer = ViccNormalizer(cache=SQLiteNormalizerCache(cache_path))
    return EntityPreTaggingService(tagger=tagger, normalizer=normalizer, keep_ungrounded=True)


def pretag_dataframe(df, context_col: str = "context", pmid_col: str = "pmid"):
    """Works with pandas or polars after converting selected columns to lists."""
    contexts = df[context_col].to_list()
    pmids = df[pmid_col].cast(str).to_list() if pmid_col in df.columns else [None] * len(contexts)
    block_ids = [str(i) for i in range(len(contexts))]

    service = build_pretagging_service()
    return service.tag_blocks(contexts=contexts, pmids=pmids, block_ids=block_ids)


def extract_interactions_from_tagged_blocks(task_runner, tagged_blocks, prompt_version: str = "v1"):
    prompt = InteractionExtractionPrompt(version=prompt_version)
    results = []
    for block in tqdm(tagged_blocks, desc="LLM extraction"):
        if not block.should_run_llm:
            results.append(
                {
                    "pmid": block.pmid,
                    "block_id": block.block_id,
                    "drug": None,
                    "gene": None,
                    "interaction": None,
                    "skipped": True,
                    "skip_reason": block.skip_reason,
                    "num_drug_candidates": len(block.grounded_drugs),
                    "num_gene_candidates": len(block.grounded_genes),
                }
            )
            continue

        payload = prompt.build_payload_from_tagged_block(block)
        try:
            task_result = task_runner.execute(
                prompt_name=prompt.name,
                prompt_version=prompt.version,
                payload=payload,
                response_model=InteractionExtractionResult,
            )
            results.append(
                {
                    "pmid": block.pmid,
                    "block_id": block.block_id,
                    "drug": task_result.drug,
                    "gene": task_result.gene,
                    "interaction": task_result.interaction,
                    "evidence": task_result.evidence,
                    "skipped": False,
                    "skip_reason": None,
                    "num_drug_candidates": len(block.grounded_drugs),
                    "num_gene_candidates": len(block.grounded_genes),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "pmid": block.pmid,
                    "block_id": block.block_id,
                    "drug": None,
                    "gene": None,
                    "interaction": None,
                    "skipped": False,
                    "skip_reason": None,
                    "error_message": str(exc),
                    "num_drug_candidates": len(block.grounded_drugs),
                    "num_gene_candidates": len(block.grounded_genes),
                }
            )
    return results
