"""Fetch, cache, and filter DGIdb publication-backed interactions."""

import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from utils import ALLOWED_TYPES, clean_value, find_col, read_cached_table, trueish

logger = logging.getLogger(__name__)

GRAPHQL_QUERY = """
query getInteractionsByGene($names: [String!]) {
  genes(names: $names) {
    nodes {
      interactions {
        drug { name }
        gene { name }
        interactionTypes { type }
        publications { pmid }
        interactionClaims { publications { pmid } }
      }
    }
  }
}
"""


def standardize_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Reduce a DGIdb summary table to unique drug-gene names."""
    drug_col = find_col(
        raw, ["drug_name", "drug", "drug_claim_primary_name", "drug_claim_name"]
    )
    gene_col = find_col(raw, ["gene_name", "gene", "gene_claim_name"])
    df = pd.DataFrame(
        {
            "drug_name": raw[drug_col].map(clean_value),
            "gene_name": raw[gene_col].map(clean_value),
        }
    )
    return (
        df[(df["drug_name"] != "") & (df["gene_name"] != "")]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def graphql_post(
    session: requests.Session, graphql_url: str, query: str, variables: dict[str, Any]
) -> dict[str, Any]:
    """POST a DGIdb GraphQL request and return the decoded payload."""
    response = session.post(
        graphql_url,
        json={"query": query, "variables": variables},
        headers={
            "User-Agent": "dgilit-pmid-eval",
            "dgidb-client-name": "dgilit-pmid-eval",
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"], indent=2)[:4000])
    return payload


def publication_pmids(publications: list[dict[str, Any]] | None) -> set[str]:
    """Extract non-empty PMID strings from DGIdb publication records."""
    return {
        clean_value(pub.get("pmid"))
        for pub in publications or []
        if clean_value(pub.get("pmid"))
    }


def interaction_types_from(
    items: list[dict[str, Any]] | None,
    mechanistic_types: Iterable[str] = tuple(ALLOWED_TYPES),
) -> list[str]:
    """Return allowed mechanistic interaction types from DGIdb records."""
    allowed = {str(x).lower() for x in mechanistic_types}
    return sorted(
        {clean_value(item.get("type")).lower() for item in items or []} & allowed
    )


def rows_from_interaction(
    interaction: dict[str, Any], mechanistic_types: Iterable[str] = tuple(ALLOWED_TYPES)
) -> list[dict[str, Any]]:
    """Expand one DGIdb interaction into publication-level rows."""
    gene = clean_value((interaction.get("gene") or {}).get("name"))
    drug = clean_value((interaction.get("drug") or {}).get("name"))
    if not gene or not drug:
        return []

    pmids = publication_pmids(interaction.get("publications"))
    for claim in interaction.get("interactionClaims") or []:
        pmids.update(publication_pmids(claim.get("publications")))
    if not pmids:
        return []

    types = interaction_types_from(
        interaction.get("interactionTypes"), mechanistic_types
    )
    return [
        {
            "pmid": pmid,
            "drug_name": drug,
            "gene_name": gene,
            "interaction_type": interaction_type,
            "interaction_type_pmid_specific": len(pmids) == 1 and bool(types),
        }
        for pmid in sorted(pmids)
        for interaction_type in (types or [""])
    ]


def fetch_publication_interactions(
    genes: list[str],
    graphql_url: str,
    graphql_query: str | None = None,
    batch_size: int = 25,
    sleep_seconds: float = 0.05,
    max_genes: int | None = None,
    mechanistic_types: Iterable[str] = tuple(ALLOWED_TYPES),
    progress: bool = False,
) -> pd.DataFrame:
    """Fetch publication-level interaction rows for DGIdb genes."""
    genes = [g for g in dict.fromkeys(clean_value(g) for g in genes) if g]
    genes = genes[:max_genes] if max_genes is not None else genes
    rows: list[dict[str, Any]] = []

    with requests.Session() as session:
        for start in range(0, len(genes), batch_size):
            if progress:
                logger.info(
                    "Fetching DGIdb genes %s-%s of %s",
                    start + 1,
                    min(start + batch_size, len(genes)),
                    len(genes),
                )
            payload = graphql_post(
                session,
                graphql_url,
                graphql_query or GRAPHQL_QUERY,
                {"names": genes[start : start + batch_size]},
            )
            for node in payload.get("data", {}).get("genes", {}).get("nodes") or []:
                for interaction in node.get("interactions") or []:
                    rows.extend(rows_from_interaction(interaction, mechanistic_types))
            if sleep_seconds:
                time.sleep(sleep_seconds)

    columns = [
        "pmid",
        "drug_name",
        "gene_name",
        "interaction_type",
        "interaction_type_pmid_specific",
    ]
    return pd.DataFrame(rows, columns=columns).drop_duplicates().reset_index(drop=True)


def load_cached_dgidb_source(
    dgidb_url: str,
    summary_cache_path: str | Path,
    source_cache_path: str | Path,
    graphql_url: str,
    refresh: bool = False,
    use_cache: bool = True,
    graphql_batch_size: int = 25,
    graphql_sleep_seconds: float = 0.05,
    max_graphql_genes: int | None = None,
    mechanistic_types: Iterable[str] = tuple(ALLOWED_TYPES),
    progress: bool = False,
) -> pd.DataFrame:
    """Load cached DGIdb source rows or rebuild them from DGIdb inputs."""
    source_cache_path = Path(source_cache_path)
    if use_cache and source_cache_path.exists() and not refresh:
        return pd.read_csv(source_cache_path, dtype=str, keep_default_na=False)

    summary = standardize_summary(
        read_cached_table(
            dgidb_url, summary_cache_path, refresh=refresh, use_cache=use_cache
        )
    )
    source = fetch_publication_interactions(
        sorted(summary["gene_name"].unique()),
        graphql_url=graphql_url,
        batch_size=graphql_batch_size,
        sleep_seconds=graphql_sleep_seconds,
        max_genes=max_graphql_genes,
        mechanistic_types=mechanistic_types,
        progress=progress,
    )
    if use_cache:
        source_cache_path.parent.mkdir(parents=True, exist_ok=True)
        source.to_csv(source_cache_path, index=False)
    return source


def filter_eligible_interactions(
    source_df: pd.DataFrame,
    require_pmid_specific_interaction_type: bool = True,
    mechanistic_types: Iterable[str] = tuple(ALLOWED_TYPES),
) -> pd.DataFrame:
    """Keep eligible mechanistic interactions for answer-sheet construction."""
    allowed = {str(x).lower() for x in mechanistic_types}
    df = source_df.fillna("").copy()
    mask = df["interaction_type"].str.lower().isin(allowed)
    if require_pmid_specific_interaction_type:
        mask &= df["interaction_type_pmid_specific"].map(trueish)
    return df[mask].reset_index(drop=True)
