"""Generate gene-specific search sets from DGIdb data.

This module filters DGIdb interaction data for a given gene and outputs a ranked CSV file for downstream operations.
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def generate_search_set(gene: str) -> None:
    """Filter the DGIdb dataset datafile for input gene and save for downstream operations
    :param gene: The desired search gene
    :type gene: str
    """
    # Filter DGIdb Dataset for input gene
    data_path = Path("data") / "dgidb" / "data-1727966158790.csv"
    df = pd.read_csv(data_path)

    tdf = df[df["interaction_score"].notnull()]
    tdf = tdf[tdf["gene_symbol"] == gene].sort_values(by="interaction_score", ascending=False).reset_index(drop=True)
    tdf.rename(columns={"gene_symbol": "Gene", "concept_name": "Drug"}, inplace=True)

    # Save
    Path('search').mkdir(parents=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_filename = f"{timestamp}_{gene}_clin_score.csv"

    out_path = Path("search") / out_filename
    tdf.to_csv(out_path, index=False)
