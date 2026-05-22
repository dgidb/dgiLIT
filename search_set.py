"""Generate gene-specific search sets from DGIdb data.

This module filters DGIdb interaction data for a given gene and outputs a ranked CSV file for downstream operations.
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parent / "research"

def generate_search_set(gene: str) -> None:
    """Filter the DGIdb dataset datafile for input gene and save for downstream operations
    :param gene: The desired search gene
    :type gene: str
    """
    # Filter DGIdb Dataset for input gene
    file_path = next((RESEARCH_DIR / "data" / "dgidb").glob("data-*.csv"))
    df = pd.read_csv(file_path)

    tdf = df[df["interaction_score"].notnull()]
    tdf = tdf[tdf["gene_symbol"] == gene].sort_values(by="interaction_score", ascending=False).reset_index(drop=True)
    tdf.rename(columns={"gene_symbol": "Gene", "concept_name": "Drug"}, inplace=True)

    # Save
    (RESEARCH_DIR / "search").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_filename = f"{timestamp}_{gene}_clin_score.csv"

    out_path = RESEARCH_DIR / "search" / out_filename
    tdf.to_csv(out_path, index=False)
