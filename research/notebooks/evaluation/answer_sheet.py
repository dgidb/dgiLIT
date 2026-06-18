"""Build deterministic answer sheets and task payloads for evaluation."""

import random
from collections.abc import Iterable

import pandas as pd
from utils import ALLOWED_TYPES, INTERACTION_COLUMNS, trueish


def simple_random_sample_pmids(df: pd.DataFrame, n: int, seed: int) -> list[str]:
    """Sample unique PMIDs from an eligible interaction frame."""
    pmids = sorted(df["pmid"].astype(str).unique())
    if len(pmids) < n:
        message = f"Only {len(pmids)} eligible PMIDs are available; cannot sample {n}."
        raise ValueError(message)
    return sorted(random.Random(seed).sample(pmids, n))  # noqa: S311


def build_answer_sheet(
    eligible_df: pd.DataFrame,
    n_pmids: int,
    seed: int,
    require_pmid_specific_interaction_type: bool = True,
    mechanistic_types: Iterable[str] = tuple(ALLOWED_TYPES),
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Create answer-sheet interactions for a deterministic PMID sample."""
    selected_pmids = simple_random_sample_pmids(eligible_df, n_pmids, seed)
    allowed = {str(x).lower() for x in mechanistic_types}

    df = eligible_df[eligible_df["pmid"].astype(str).isin(selected_pmids)].copy()
    if require_pmid_specific_interaction_type:
        df = df[df["interaction_type_pmid_specific"].map(trueish)]
    df = df[df["interaction_type"].isin(allowed)]
    df = (
        df.drop_duplicates()
        .sort_values(["pmid", "interaction_type", "gene_name", "drug_name"])
        .reset_index(drop=True)
    )

    if df["pmid"].nunique() != n_pmids:
        message = "Answer sheet does not contain exactly the requested number of PMIDs."
        raise ValueError(message)

    return df[INTERACTION_COLUMNS].copy(), df, selected_pmids


def task_pmids_from_abstracts(
    selected_pmids: list[str], abstracts_df: pd.DataFrame
) -> pd.DataFrame:
    """Return selected PubMed records in task order."""
    order = {str(pmid).strip(): i for i, pmid in enumerate(selected_pmids)}
    df = abstracts_df[abstracts_df["pmid"].astype(str).isin(order)].copy()
    missing = set(order) - set(df["pmid"].astype(str))
    if missing:
        message = f"Missing PubMed abstracts for selected PMID(s): {sorted(missing)}"
        raise ValueError(message)
    df["_order"] = df["pmid"].astype(str).map(order)
    return df.sort_values("_order")[["pmid", "article_title", "abstract"]].reset_index(
        drop=True
    )


def task_payloads_from_pmids(task_pmids_df: pd.DataFrame) -> list[dict[str, str]]:
    """Convert PubMed task rows into WAGS prompt payloads."""
    return [
        {
            "pmid": str(r.pmid),
            "article_title": str(r.article_title or ""),
            "abstract": str(r.abstract or ""),
        }
        for r in task_pmids_df.itertuples(index=False)
    ]
