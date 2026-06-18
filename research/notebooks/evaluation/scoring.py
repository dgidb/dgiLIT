"""Scoring helpers for extracted drug-gene interaction predictions."""

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from utils import ALLOWED_TYPES, clean_value, norm_gene, norm_text

logger = logging.getLogger(__name__)


def predictions_to_frame(predictions: dict[str, Any]) -> pd.DataFrame:
    """Convert saved prediction records into an interaction DataFrame."""
    rows = []
    for pmid, payload in predictions.items():
        rows.extend(
            [
                {
                    "pmid": str(item.get("pmid") or pmid).strip(),
                    "drug_name": clean_value(item.get("drug_name")),
                    "gene_name": clean_value(item.get("gene_name")),
                    "interaction_type": clean_value(
                        item.get("interaction_type")
                    ).lower(),
                }
                for item in (
                    payload.get("interactions", []) if isinstance(payload, dict) else []
                )
            ]
        )
    return pd.DataFrame(
        rows, columns=["pmid", "drug_name", "gene_name", "interaction_type"]
    ).drop_duplicates()


def standardize_interactions(
    df: pd.DataFrame, mechanistic_types: Iterable[str] = tuple(ALLOWED_TYPES)
) -> pd.DataFrame:
    """Create normalized matching keys for interaction rows."""
    df = df.copy().fillna("")
    for col in ["pmid", "drug_name", "gene_name", "interaction_type"]:
        if col not in df:
            df[col] = ""
    df["pmid_key"] = df["pmid"].astype(str).str.strip()
    df["drug_key"] = df["drug_name"].map(norm_text)
    df["gene_key"] = df["gene_name"].map(norm_gene)
    df["type_key"] = df["interaction_type"].map(lambda x: clean_value(x).lower())
    complete = (df[["pmid_key", "drug_key", "gene_key"]] != "").all(axis=1)
    return df[
        complete & df["type_key"].isin({str(x).lower() for x in mechanistic_types})
    ]


def _tuples(df: pd.DataFrame, cols: list[str]) -> set[tuple[Any, ...]]:
    return (
        set(map(tuple, df[cols].drop_duplicates().to_numpy()))
        if not df.empty
        else set()
    )


def _metrics(
    expected_df: pd.DataFrame, predicted_df: pd.DataFrame, cols: list[str], level: str
) -> dict[str, Any]:
    expected = _tuples(expected_df, cols)
    predicted = _tuples(predicted_df, cols)
    tp = expected & predicted
    precision = len(tp) / len(predicted) if predicted else 0.0
    recall = len(tp) / len(expected) if expected else 0.0
    return {
        "level": level,
        "expected": len(expected),
        "predicted": len(predicted),
        "true_positives": len(tp),
        "false_positives": len(predicted - expected),
        "false_negatives": len(expected - predicted),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
    }


def score_interaction_frames(
    expected_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    mechanistic_types: Iterable[str] = tuple(ALLOWED_TYPES),
) -> pd.DataFrame:
    """Score expected and predicted interaction frames at pair and typed levels."""
    expected = standardize_interactions(expected_df, mechanistic_types)
    predicted = standardize_interactions(predicted_df, mechanistic_types)
    return pd.DataFrame(
        [
            _metrics(expected, predicted, ["pmid_key", "drug_key", "gene_key"], "pair"),
            _metrics(
                expected,
                predicted,
                ["pmid_key", "drug_key", "gene_key", "type_key"],
                "typed",
            ),
        ]
    )


def is_successful_prediction_record(payload: dict[str, Any]) -> bool:
    """Return whether a saved prediction record contains parsed interactions."""
    return (
        isinstance(payload, dict)
        and not payload.get("error")
        and "interactions" in payload
    )


def score_prediction_dict(
    answer_sheet: pd.DataFrame,
    predictions: dict[str, Any],
    mechanistic_types: Iterable[str] = tuple(ALLOWED_TYPES),
) -> pd.DataFrame:
    """Score an in-memory prediction dictionary against the answer sheet."""
    scored_pmids = {
        str(pmid)
        for pmid, payload in predictions.items()
        if is_successful_prediction_record(payload)
    }
    columns = [
        "level",
        "expected",
        "predicted",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
        "scored_pmids",
        "sampled_answer_pmids",
        "prediction_records",
        "successful_prediction_records",
        "error_records",
    ]
    if not scored_pmids:
        logger.warning(
            "No successful prediction records are available; extraction metrics were not computed."
        )
        return pd.DataFrame(columns=columns)

    expected = answer_sheet[answer_sheet["pmid"].astype(str).isin(scored_pmids)]
    predicted = {
        pmid: payload
        for pmid, payload in predictions.items()
        if str(pmid) in scored_pmids
    }
    metrics = score_interaction_frames(
        expected, predictions_to_frame(predicted), mechanistic_types
    )
    metrics["scored_pmids"] = len(scored_pmids)
    metrics["sampled_answer_pmids"] = answer_sheet["pmid"].astype(str).nunique()
    metrics["prediction_records"] = len(predictions)
    metrics["successful_prediction_records"] = len(scored_pmids)
    metrics["error_records"] = sum(
        1 for p in predictions.values() if isinstance(p, dict) and p.get("error")
    )
    return metrics


def score_predictions(
    answer_sheet: pd.DataFrame,
    predictions_path: str | Path,
    metrics_path: str | Path | None = None,
    mechanistic_types: Iterable[str] = tuple(ALLOWED_TYPES),
) -> pd.DataFrame:
    """Load predictions from disk, score them, and optionally write metrics."""
    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        logger.warning("Prediction file does not exist yet: %s", predictions_path)
        return score_prediction_dict(answer_sheet, {}, mechanistic_types)

    metrics = score_prediction_dict(
        answer_sheet, json.loads(predictions_path.read_text()), mechanistic_types
    )
    if metrics_path is not None and not metrics.empty:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(metrics_path, index=False)
    return metrics
