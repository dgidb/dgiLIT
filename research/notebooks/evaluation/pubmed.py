"""Fetch, parse, cache, and filter PubMed abstracts for evaluation."""

import logging
import time
from pathlib import Path

import pandas as pd
import requests
from defusedxml import ElementTree
from utils import clean_value, collapse_ws

logger = logging.getLogger(__name__)


def element_text(element: object | None) -> str:
    """Return collapsed text content for an XML element."""
    return "" if element is None else collapse_ws("".join(element.itertext()))


def parse_pubmed_abstracts(xml_text: str) -> pd.DataFrame:
    """Parse PubMed efetch XML into PMID, title, and abstract rows."""
    rows = []
    for article in ElementTree.fromstring(xml_text).findall(".//PubmedArticle"):
        parts = []
        for item in article.findall(".//Abstract/AbstractText"):
            text = element_text(item)
            label = clean_value(
                item.attrib.get("Label") or item.attrib.get("NlmCategory") or ""
            )
            if text:
                parts.append(f"{label}: {text}" if label else text)
        rows.append(
            {
                "pmid": clean_value(
                    element_text(article.find(".//MedlineCitation/PMID"))
                ),
                "article_title": element_text(article.find(".//ArticleTitle")),
                "abstract": "\n".join(parts).strip(),
            }
        )
    return pd.DataFrame(
        rows, columns=["pmid", "article_title", "abstract"]
    ).drop_duplicates("pmid")


def fetch_pubmed_abstract_batch(
    pmids: list[str],
    efetch_url: str,
    ncbi_tool: str = "dgilit-pmid-eval",
    ncbi_email: str = "",
) -> pd.DataFrame:
    """Fetch one PubMed abstract batch from NCBI efetch."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": ncbi_tool,
    }
    if ncbi_email:
        params["email"] = ncbi_email
    response = requests.get(
        efetch_url, params=params, headers={"User-Agent": ncbi_tool}, timeout=120
    )
    response.raise_for_status()
    return parse_pubmed_abstracts(response.text)


def fetch_pubmed_abstracts(
    pmids: list[str],
    cache_path: str | Path,
    efetch_url: str,
    batch_size: int = 100,
    sleep_seconds: float = 0.34,
    refresh: bool = False,
    use_cache: bool = True,
    ncbi_tool: str = "dgilit-pmid-eval",
    ncbi_email: str = "",
    progress: bool = False,
) -> pd.DataFrame:
    """Load cached abstracts and fetch missing PMID records."""
    pmids = [str(pmid).strip() for pmid in dict.fromkeys(pmids) if str(pmid).strip()]
    columns = ["pmid", "article_title", "abstract"]
    cache_path = Path(cache_path)

    cached = pd.DataFrame(columns=columns)
    if use_cache and cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, dtype=str, keep_default_na=False)
        cached = cached.reindex(columns=columns, fill_value="").drop_duplicates("pmid")

    missing = [pmid for pmid in pmids if pmid not in set(cached["pmid"].astype(str))]
    fetched = []
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        if progress:
            logger.info(
                "Fetching PubMed abstracts %s-%s of %s missing PMIDs",
                start + 1,
                min(start + batch_size, len(missing)),
                len(missing),
            )
        frame = fetch_pubmed_abstract_batch(batch, efetch_url, ncbi_tool, ncbi_email)
        found = set(frame["pmid"].astype(str)) if not frame.empty else set()
        blanks = [
            {"pmid": pmid, "article_title": "", "abstract": ""}
            for pmid in batch
            if pmid not in found
        ]
        fetched.append(
            pd.concat([frame, pd.DataFrame(blanks)], ignore_index=True)[columns]
        )
        if sleep_seconds:
            time.sleep(sleep_seconds)

    if fetched:
        cached = (
            pd.concat([cached, *fetched], ignore_index=True)[columns]
            .fillna("")
            .drop_duplicates("pmid", keep="last")
        )
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cached.to_csv(cache_path, index=False)

    return cached[cached["pmid"].astype(str).isin(pmids)].reset_index(drop=True)


def filter_interactions_with_abstracts(
    interactions_df: pd.DataFrame,
    abstracts_df: pd.DataFrame,
    min_pmids: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Keep interactions whose PMID has a non-empty PubMed abstract."""
    before = interactions_df["pmid"].astype(str).nunique()
    abstracts = abstracts_df[
        abstracts_df["abstract"].map(lambda x: collapse_ws(x) != "")
    ][["pmid", "article_title", "abstract"]].copy()
    interactions = (
        interactions_df[
            interactions_df["pmid"].astype(str).isin(set(abstracts["pmid"].astype(str)))
        ]
        .copy()
        .reset_index(drop=True)
    )
    after = interactions["pmid"].astype(str).nunique()

    if interactions.empty:
        message = "No eligible interactions with PubMed abstracts are available."
        raise ValueError(message)
    if min_pmids is not None and after < min_pmids:
        message = (
            f"Only {after} eligible PMID(s) have PubMed abstracts; "
            f"cannot sample {min_pmids}."
        )
        raise ValueError(message)

    return (
        interactions,
        abstracts.reset_index(drop=True),
        {
            "pmids_before_abstract_filter": before,
            "pmids_removed_without_abstract": before - after,
            "pmids_after_abstract_filter": after,
            "eligible_interaction_rows_after_filter": len(interactions),
        },
    )
