"""Streaming Parquet snapshots with an atomic latest-snapshot pointer."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid


def export_run(store, min_free_bytes):
    import polars as pl
    generation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    directory = store.output / "exports" / generation
    directory.mkdir(parents=True)
    # Full JSON payloads are retained alongside fields used for routine filtering.
    active = " JOIN articles a ON a.pmcid=t.pmcid AND a.text_hash=t.article_hash "
    aggregates = '''SELECT t.pmcid,a.pmid,t.drug_id,t.gene_id,min(t.drug) drug,min(t.gene) gene,
        count(*) context_count,sum(t.status='done') completed_contexts,
        sum(t.status!='done') unresolved_contexts,
        sum(CASE WHEN t.status='done' AND json_extract(t.result,'$.interaction')=1 THEN 1 ELSE 0 END) positive_contexts,
        json_group_array(CASE WHEN t.status='done' AND json_extract(t.result,'$.interaction')=1 THEN t.id END) supporting_task_ids,
        a.text_hash,a.status article_status,min(t.config_hash) config_hash,
        min(json_extract(t.result,'$.model_id')) model_id,
        min(json_extract(t.result,'$.prompt_version')) prompt_version
        FROM tasks t ''' + active + " GROUP BY t.pmcid,t.drug_id,t.gene_id"
    # Preparation can fail before all section tasks exist. Do not call a pair
    # negative until the article has completed processing as well as its tasks.
    paired = '''SELECT *,CASE WHEN positive_contexts>0 THEN 1
        WHEN unresolved_contexts=0 AND article_status IN ('complete','complete_partial') THEN 0 END interaction,
        CASE WHEN positive_contexts>0 THEN 'positive'
        WHEN unresolved_contexts=0 AND article_status IN ('complete','complete_partial') THEN 'negative'
        ELSE 'unresolved' END pair_status
        FROM (''' + aggregates + ")"
    queries = {
        "articles": "SELECT * FROM articles ORDER BY pmcid",
        "partitions": "SELECT * FROM partitions ORDER BY start,end",
        "sections": "SELECT s.* FROM sections s JOIN articles a ON s.pmcid=a.pmcid AND s.article_hash=a.text_hash ORDER BY s.id",
        "entities": '''SELECT e.id,e.pmcid,a.pmid,e.section_id,
            json_extract(e.data,'$.entity_type') entity_type,json_extract(e.data,'$.text') mention,
            json_extract(e.data,'$.start') start,json_extract(e.data,'$.end') end,
            json_extract(e.data,'$.concept.concept_id') concept_id,
            json_extract(e.data,'$.concept.concept_label') concept_label,e.data
            FROM entities e JOIN sections s ON e.section_id=s.id
            JOIN articles a ON s.pmcid=a.pmcid AND s.article_hash=a.text_hash ORDER BY e.id''',
        "classifications": '''SELECT t.*,a.pmid,json_extract(t.result,'$.interaction') interaction,
            json_extract(t.result,'$.evidence') evidence,json_extract(t.result,'$.interaction_type') interaction_type,
            json_extract(t.result,'$.directionality') directionality,
            json_extract(t.result,'$.model_id') model_id,json_extract(t.result,'$.prompt_version') prompt_version
            FROM tasks t ''' + active + " ORDER BY t.id",
        "article_pairs": paired,
        "positive_interactions": "SELECT * FROM (" + paired + ") WHERE positive_contexts>0",
        "failures": "SELECT * FROM failures ORDER BY id",
    }
    integers = {"cohort_year", "pilot", "section_count", "expected", "received", "section_index", "context_start", "context_end",
                "attempts", "start", "end", "context_count", "completed_contexts", "unresolved_contexts", "positive_contexts", "resolved"}
    manifest = {"generation": generation, "config": store.meta("config"),
                "execution_fingerprint": store.meta("execution_fingerprint"), "status": store.status(), "datasets": {}}
    for name, query in queries.items():
        cursor = store.db.execute(query)
        columns = [d[0] for d in cursor.description]
        schema = {column: (pl.Boolean if column == "interaction" else pl.Float64 if column == "elapsed_seconds"
                           else pl.Int64 if column in integers and name != "partitions" or name == "failures" and column == "id"
                           else pl.Int64 if name == "partitions" and column in {"expected", "received"} else pl.String)
                  for column in columns}
        destination = directory / name
        destination.mkdir()
        count, shard = 0, 0
        while True:
            store.check_space(min_free_bytes)
            batch = cursor.fetchmany(10000)
            if not batch and shard:
                break
            rows = [dict(row) for row in batch]
            for row in rows:
                if "supporting_task_ids" in row:
                    row["supporting_task_ids"] = json.dumps([x for x in json.loads(row["supporting_task_ids"]) if x is not None])
            frame = pl.from_dicts(rows, schema=schema, strict=False) if rows else pl.DataFrame(schema=schema)
            temporary = destination / f"part-{shard:06d}.tmp"
            frame.write_parquet(temporary, compression="zstd")
            temporary.replace(temporary.with_suffix(".parquet"))
            count += len(rows)
            shard += 1
            if not batch:
                break
        manifest["datasets"][name] = {"rows": count, "parts": shard}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pointer = store.output / "exports/latest.json"
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(json.dumps({"generation": generation, "manifest": str(directory / 'manifest.json')}, indent=2) + "\n")
    temporary.replace(pointer)
    return manifest
