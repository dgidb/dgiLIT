"""Resumable, section-level PMC interaction extraction.

The notebook and CLI use this API; no corpus-sized dataframe is constructed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import time

from .dgilit_models import InteractionClassificationPrompt, InteractionClassificationResult
from .pmc import PMCArticle, PMCClient
from .pmc_store import Store, ResourceLimit, digest, now

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data/pmc_last10years_2016-11-03_2026-11-03"


@dataclass(frozen=True)
class PipelineConfig:
    start: str = "2016-11-03"
    end: str = "2026-11-03"
    model_id: str = "mlx-community/Qwen3-14B-4bit"
    prompt_version: str = "v2"
    max_tokens: int = 1000
    ner_overlap: int = 64
    context_overlap: int = 256
    schema_version: int = 1

    def __post_init__(self):
        if date.fromisoformat(self.start) > date.fromisoformat(self.end):
            raise ValueError("Start date must not follow end date")
        if self.max_tokens <= 0 or min(self.ner_overlap, self.context_overlap) < 0:
            raise ValueError("Invalid token budget/overlap")


def _model_revision(model_id):
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(model_id, "config.json")
    return Path(path).parent.name if isinstance(path, str) else "uncached"


class PMCPipeline:
    def __init__(self, output=DEFAULT_OUTPUT, config=None, client=None, pretagger=None,
                 llm=None, min_free_bytes=10 * 1024**3, cache_bytes=10 * 1024**3):
        self.output = Path(output)
        # Opening status/export with no config reuses the immutable saved configuration.
        if config is None and (self.output / "state.sqlite").exists():
            import sqlite3
            with sqlite3.connect(self.output / "state.sqlite") as db:
                row = db.execute("SELECT value FROM meta WHERE key='config'").fetchone()
                config = PipelineConfig(**json.loads(row[0])) if row else None
        self.config = config or PipelineConfig()
        self.store = Store(self.output, asdict(self.config))
        self.client = client or PMCClient()
        self.pretagger, self.llm = pretagger, llm
        self.min_free_bytes, self.cache_bytes = min_free_bytes, cache_bytes
        self.prompt = InteractionClassificationPrompt(self.config.prompt_version)
        self.schema = InteractionClassificationResult.model_json_schema()

    def discover(self):
        with self.store.writer_lock():
            self._discover()
        return self.store.write_status(force=True)

    def _discover(self):
        if self.store.meta("discovery_complete"):
            return
        self.store.set_meta("discovery_started_at", self.store.meta("discovery_started_at", now()))
        start, end = date.fromisoformat(self.config.start), date.fromisoformat(self.config.end)
        self.store.set_meta("query", self.client.query(start, end))
        cursor = start
        while cursor <= end:
            stop = min(date(cursor.year, 12, 31), end)
            self._discover_partition(cursor, stop)
            cursor = stop + timedelta(days=1)
        self.store.set_meta("discovery_complete", True)
        self.store.set_meta("discovery_finished_at", now())

    def _discover_partition(self, start, end, uid_bounds=None):
        self.store.check_space(self.min_free_bytes)
        key = f"{start}:{end}" + (f":{uid_bounds[0]}:{uid_bounds[1]}" if uid_bounds else "")
        db = self.store.db
        old = db.execute("SELECT status FROM partitions WHERE id=?", (key,)).fetchone()
        if old and old[0] in {"complete", "split"}:
            return
        query = self.client.query(start, end)
        if uid_bounds:
            query = f"({query}) AND ({uid_bounds[0]}:{uid_bounds[1]}[uid])"
        def search(**kwargs):
            if uid_bounds:
                kwargs["query"] = query
            return self.client.search(start, end, **kwargs)
        with db:
            db.execute("INSERT OR REPLACE INTO partitions(id,start,end,query,status,discovered_at) VALUES (?,?,?,?,?,?)",
                       (key, str(start), str(end), query, "searching", now()))
        try:
            # Obtain the count and IDs together. Count-only responses and later
            # ID requests may see different live/cached database versions. In
            # particular, retmax=an earlier count can omit newly indexed IDs.
            # Every leaf fits below 9,000 IDs, so no second page is needed.
            for attempt in range(3):
                count, identifiers = search(size=9000)
                unique_count = len(set(identifiers))
                with db:
                    db.execute("UPDATE partitions SET expected=?,received=? WHERE id=?", (count, unique_count, key))
                if count > 9000 or (len(identifiers) == count and unique_count == count):
                    break
                if attempt == 2:
                    raise ValueError(
                        f"Incomplete PMC search response for {query}: expected {count} IDs, "
                        f"received {len(identifiers)} ({unique_count} unique) after 3 attempts"
                    )
                time.sleep(2 ** attempt)
            if count > 9000:
                if start < end and uid_bounds is None:
                    middle = start + (end - start) // 2
                    self._discover_partition(start, middle)
                    self._discover_partition(middle + timedelta(days=1), end)
                elif uid_bounds is None:
                    upper = 10**9
                    while True:
                        # Prove the bound has no excluded IDs. Comparing counts
                        # from two different queries recreates the live-count race.
                        outside = f"({query}) NOT (1:{upper}[uid])"
                        excluded, _ = self.client.search(start, end, query=outside, size=9000)
                        if excluded == 0:
                            break
                        upper *= 10
                        if upper > 10**18:
                            raise ValueError("Unable to establish a UID range covering the entire high-volume date")
                    self._discover_partition(start, end, (1, upper))
                else:
                    low, high = uid_bounds
                    if low == high:
                        raise ValueError("Multiple records for a single PMC UID")
                    middle = (low + high) // 2
                    self._discover_partition(start, end, (low, middle))
                    self._discover_partition(start, end, (middle + 1, high))
                with db:
                    db.execute("UPDATE partitions SET expected=?,status='split' WHERE id=?", (count, key))
                self.store.resolve("discovery", key)
                return
            self.store.check_space(self.min_free_bytes)
            with db:
                db.executemany("INSERT OR IGNORE INTO articles(pmcid,cohort_year,updated_at) VALUES (?,?,?)",
                               ((pmcid, start.year, now()) for pmcid in identifiers))
                db.execute("UPDATE partitions SET expected=?,received=?,status='complete',error=NULL WHERE id=?",
                           (count, unique_count, key))
            self.store.resolve("discovery", key)
            print(f"Discovered {start}–{end}: {count:,} PMC IDs", flush=True)
            self.store.write_status()
        except Exception as exc:
            with db:
                db.execute("UPDATE partitions SET status='failed',error=? WHERE id=?", (str(exc), key))
            self.store.fail("discovery", key, None, exc)
            self.store.write_status()
            raise

    def _ensure_models(self):
        if self.llm is None:
            from .qwen import Qwen3JsonClient
            self.llm = Qwen3JsonClient(self.config.model_id, self.config.max_tokens, 0.0)
        if self.pretagger is None:
            from .tagging import BioBertEntityTagger, EntityPreTaggingService, TaggerConfig, ViccNormalizer, SQLiteNormalizerCache
            self.pretagger = EntityPreTaggingService(
                tagger=BioBertEntityTagger(TaggerConfig(long_text_overlap=self.config.ner_overlap)),
                normalizer=ViccNormalizer(cache=SQLiteNormalizerCache(self.output / "normalizer.sqlite")),
            )
            # Resolve/load NER revisions before fingerprinting a fresh installation.
            for kind in ["drug", "gene"]:
                self.pretagger.tagger._get_pipeline(kind)
        if not hasattr(self, "execution_hash"):
            from .tagging.taggers import BioBertEntityTagger
            revisions = {model: _model_revision(model) for model in [self.config.model_id,
                         BioBertEntityTagger.MODEL_BY_TYPE["drug"], BioBertEntityTagger.MODEL_BY_TYPE["gene"]]}
            versions = {}
            for package in ["mlx", "mlx-lm", "transformers", "wags-llm"]:
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = "missing"
            fingerprint = {"config": asdict(self.config), "model_revisions": revisions,
                           "packages": versions, "prompt_hash": digest(self.prompt.build_system_prompt())}
            previous = self.store.meta("execution_fingerprint")
            if previous and previous != fingerprint:
                raise ValueError("Model/prompt/dependency versions changed; use a new output directory")
            self.store.set_meta("execution_fingerprint", fingerprint)
            self.execution_hash = digest(fingerprint)

    def _fetch_batch(self, identifiers):
        self.store.check_space(self.min_free_bytes)
        articles = self.client.fetch_articles(identifiers)
        for pmcid in identifiers:
            article = articles.get(pmcid)
            error = self.client.last_errors.get(pmcid)
            if error:
                self.store.fail("retrieval", pmcid, pmcid, error)
            else:
                self.store.resolve("retrieval", pmcid)
            if article is None:
                with self.store.db:
                    self.store.db.execute("UPDATE articles SET status='failed',coverage='retrieval_failed',error=?,updated_at=? WHERE pmcid=?",
                                          (error or "No record returned", now(), pmcid))
                continue
            self.store.cache_put(article, self.cache_bytes)
            text_hash = digest([s.model_dump() for s in article.sections])
            with self.store.db:
                self.store.db.execute('''UPDATE articles SET status=?,coverage=?,pmid=?,title=?,publication_year=?,
                    license=?,text_hash=?,section_count=?,retrieval_error=?,error=NULL,updated_at=? WHERE pmcid=?''',
                    ("ready" if article.sections else "failed" if error else "no_text", article.coverage,
                     article.pmid, article.title, article.publication_year, article.license, text_hash,
                     len(article.sections), error, now(), pmcid))

    def _prepare(self, row, article):
        db = self.store.db
        for index, section in enumerate(article.sections):
            section_hash = digest(section.text)
            section_id = digest([article.pmcid, row["text_hash"], index, section_hash])
            with db:
                db.execute("INSERT OR REPLACE INTO sections VALUES (?,?,?,?,?,?)",
                           (section_id, article.pmcid, row["text_hash"], index, section_hash,
                            json.dumps(section.model_dump(exclude={"text"}))))
            block = None
            for attempt in range(3):
                block = self.pretagger.tag_block(section.text, article.pmid, section_id)
                if not any(m.concept and m.concept.error_message for m in block.entities):
                    break
                if attempt < 2:
                    time.sleep(2 ** attempt)
            with db:
                for mention in block.entities:
                    entity_id = digest([section_id, mention.entity_type, mention.start, mention.end])
                    data = mention.model_dump()
                    if data.get("concept"):
                        data["concept"].pop("raw_response", None)
                    db.execute("INSERT OR REPLACE INTO entities VALUES (?,?,?,?)",
                               (entity_id, section_id, article.pmcid, json.dumps(data)))
            if any(m.concept and m.concept.error_message for m in block.entities):
                raise RuntimeError("Transient entity normalization failures; entities retained for retry")
            candidates = {}
            for kind in ["drug", "gene"]:
                candidates[kind] = {m.concept_id: m.concept_label for m in block.entities
                                    if m.entity_type == kind and m.is_grounded and m.concept_label}
            for (drug_id, drug), (gene_id, gene) in itertools.product(
                    sorted(candidates["drug"].items()), sorted(candidates["gene"].items())):
                for start, end in self.llm.context_windows(self.prompt, section.text, drug, gene, self.schema,
                                                          self.config.context_overlap):
                    context_hash = digest(section.text[start:end])
                    task_id = digest([section_id, context_hash, start, end, drug_id, gene_id, self.execution_hash])
                    with db:
                        db.execute('''INSERT OR IGNORE INTO tasks
                            (id,pmcid,article_hash,section_id,section_index,context_start,context_end,context_hash,
                             drug_id,gene_id,drug,gene,config_hash,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                            (task_id, article.pmcid, row["text_hash"], section_id, index, start, end, context_hash,
                             drug_id, gene_id, drug, gene, self.execution_hash, now()))
        count = db.execute("SELECT count(*) FROM tasks WHERE pmcid=? AND article_hash=?",
                           (article.pmcid, row["text_hash"])).fetchone()[0]
        with db:
            db.execute("UPDATE articles SET status=?,updated_at=? WHERE pmcid=?",
                       ("prepared" if count else "no_candidates", now(), article.pmcid))
        self.store.resolve("preparation", article.pmcid)

    def _classify(self, task, article):
        context = article.sections[task["section_index"]].text[task["context_start"]:task["context_end"]]
        if digest(context) != task["context_hash"]:
            raise ValueError("Cached text no longer matches the task; source must be prepared again")
        with self.store.db:
            self.store.db.execute("UPDATE tasks SET status='running',attempts=attempts+1,updated_at=? WHERE id=?", (now(), task["id"]))
        started = time.monotonic()
        try:
            payload = self.prompt.build_payload(context, task["drug"], task["gene"])
            response = self.llm.invoke_json(self.prompt.build_system_prompt(), self.prompt.build_user_prompt(payload), self.schema)
            result = InteractionClassificationResult.model_validate(response.parsed_json)
            evidence = " ".join((result.evidence or "").split()).strip('"“”')
            verified = bool(evidence) and evidence.casefold() in " ".join(context.split()).casefold()
            if result.interaction and not verified:
                raise ValueError("Positive classification lacks a verifiable direct evidence quote")
            record = result.model_dump()
            record.update(evidence_in_context=verified, model_id=self.config.model_id,
                          prompt_version=self.config.prompt_version, temperature=0, enable_thinking=False,
                          generation_stats=getattr(self.llm, "last_generation_stats", {}))
            with self.store.db:
                self.store.db.execute("UPDATE tasks SET status='done',result=?,error=NULL,elapsed_seconds=?,updated_at=? WHERE id=?",
                                      (json.dumps(record), time.monotonic()-started, now(), task["id"]))
            self.store.resolve("classification", task["id"])
        except Exception as exc:
            with self.store.db:
                self.store.db.execute("UPDATE tasks SET status='failed',error=?,elapsed_seconds=?,updated_at=? WHERE id=?",
                                      (str(exc), time.monotonic()-started, now(), task["id"]))
            self.store.fail("classification", task["id"], article.pmcid, exc)

    def _select_pilot(self):
        if self.store.meta("pilot_selected"):
            return
        db = self.store.db
        db.create_function("stable_hash", 1, digest, deterministic=True)
        years = [r[0] for r in db.execute("SELECT DISTINCT cohort_year FROM articles ORDER BY cohort_year")]
        chosen = []
        # Deterministic per-year streams, then round robin until 100 or exhaustion.
        streams = [iter(db.execute("SELECT pmcid FROM articles WHERE cohort_year=? ORDER BY stable_hash(pmcid) LIMIT 100", (year,))) for year in years]
        while streams and len(chosen) < 100:
            remaining = []
            for stream in streams:
                row = next(stream, None)
                if row and len(chosen) < 100:
                    chosen.append(row[0])
                    remaining.append(stream)
            streams = remaining
        with db:
            db.executemany("UPDATE articles SET pilot=1 WHERE pmcid=?", ((p,) for p in chosen))
        self.store.set_meta("pilot_selected", chosen)

    def _phase(self, pilot=False, max_calls=None):
        db, calls = self.store.db, 0
        selector = " AND pilot=1" if pilot else ""
        # Fetch and prepare all pilot articles even when its inference budget is spent.
        while True:
            self.store.check_space(self.min_free_bytes)
            pending = [r[0] for r in db.execute("SELECT pmcid FROM articles WHERE status='pending'" + selector + " ORDER BY cohort_year,pmcid LIMIT 25")]
            if pending:
                self._fetch_batch(pending)
            ready = db.execute("SELECT * FROM articles WHERE status IN ('ready','prepared')" + selector + " ORDER BY cohort_year,pmcid LIMIT 25").fetchall()
            if not ready and not pending:
                break
            progressed = False
            for row in ready:
                self.store.check_space(self.min_free_bytes)
                cached = self.store.cache_get(row["pmcid"])
                if cached is None:
                    self._fetch_batch([row["pmcid"]])
                    row = db.execute("SELECT * FROM articles WHERE pmcid=?", (row["pmcid"],)).fetchone()
                    cached = self.store.cache_get(row["pmcid"])
                    if cached is None or row["status"] not in {"ready", "prepared"}:
                        continue
                article = PMCArticle.model_validate(cached)
                try:
                    if row["status"] == "ready":
                        self._prepare(row, article)
                        progressed = True
                    while max_calls is None or calls < max_calls:
                        task = db.execute("SELECT * FROM tasks WHERE pmcid=? AND article_hash=? AND status='pending' ORDER BY id LIMIT 1",
                                          (row["pmcid"], row["text_hash"])).fetchone()
                        if task is None:
                            break
                        self.store.check_space(self.min_free_bytes)
                        self._classify(task, article)
                        calls += 1
                        progressed = True
                        if calls % 10 == 0:
                            self.store.write_status()
                            print(f"Classified {calls:,} contexts this phase; {row['pmcid']}", flush=True)
                    counts = {r[0]: r[1] for r in db.execute("SELECT status,count(*) FROM tasks WHERE pmcid=? AND article_hash=? GROUP BY status",
                                                           (row["pmcid"], row["text_hash"]))}
                    if not counts.get("pending") and not counts.get("running") and counts:
                        status = "failed" if counts.get("failed") else "complete_partial" if row["retrieval_error"] else "complete"
                        with db:
                            db.execute("UPDATE articles SET status=?,updated_at=? WHERE pmcid=?", (status, now(), row["pmcid"]))
                except ResourceLimit:
                    raise
                except Exception as exc:
                    self.store.fail("preparation", row["pmcid"], row["pmcid"], exc)
                    with db:
                        db.execute("UPDATE articles SET status='failed',error=?,updated_at=? WHERE pmcid=?", (str(exc), now(), row["pmcid"]))
                    progressed = True
                self.store.write_status()
            if max_calls is not None and calls >= max_calls:
                # Prepare remaining pilot articles without revisiting already prepared rows.
                if pilot:
                    self._prepare_remaining_pilot()
                break
            if not progressed and not pending:
                break
        return calls

    def _prepare_remaining_pilot(self):
        db = self.store.db
        while True:
            self.store.check_space(self.min_free_bytes)
            pending = [r[0] for r in db.execute("SELECT pmcid FROM articles WHERE pilot=1 AND status='pending' LIMIT 25")]
            if pending:
                self._fetch_batch(pending)
            rows = db.execute("SELECT * FROM articles WHERE pilot=1 AND status='ready' LIMIT 25").fetchall()
            for row in rows:
                try:
                    self._prepare(row, PMCArticle.model_validate(self.store.cache_get(row["pmcid"])))
                except ResourceLimit:
                    raise
                except Exception as exc:
                    self.store.fail("preparation", row["pmcid"], row["pmcid"], exc)
                    with db:
                        db.execute("UPDATE articles SET status='failed',error=? WHERE pmcid=?", (str(exc), row["pmcid"]))
            self.store.write_status()
            if not pending and not rows:
                break

    def _pilot_report(self, elapsed):
        db = self.store.db
        count = db.execute("SELECT count(*) FROM articles WHERE pilot=1").fetchone()[0]
        tasks = db.execute("SELECT count(*),sum(elapsed_seconds),count(elapsed_seconds) FROM tasks t JOIN articles a ON a.pmcid=t.pmcid AND a.text_hash=t.article_hash WHERE a.pilot=1").fetchone()
        total = db.execute("SELECT count(*) FROM articles").fetchone()[0]
        mean = (tasks[1] / tasks[2]) if tasks[2] else None
        db_bytes = db.execute("PRAGMA page_count").fetchone()[0] * db.execute("PRAGMA page_size").fetchone()[0]
        baseline = self.store.meta("pilot_baseline_state_bytes", db_bytes)
        sample_growth = max(0, db_bytes - baseline)
        estimated_calls = tasks[0] / count * total if count else None
        report = {
            "sample_articles": count, "corpus_articles": total,
            "sample_context_tasks": tasks[0], "measured_calls": tasks[2],
            "mean_call_seconds": mean, "pilot_elapsed_seconds": elapsed,
            "estimated_corpus_context_tasks": estimated_calls,
            "estimated_inference_days": estimated_calls * mean / 86400 if mean and estimated_calls else None,
            "inventory_baseline_bytes": baseline,
            "sample_state_growth_bytes": sample_growth,
            "estimated_state_bytes": baseline + sample_growth / count * total if count else None,
            "coverage": {r[0]: r[1] for r in db.execute("SELECT coalesce(coverage,'not_fetched'),count(*) FROM articles WHERE pilot=1 GROUP BY coverage")},
            "caveat": "Small stratified sample; estimates exclude retrieval/NER time and may be highly variable.",
        }
        self.store.set_meta("pilot_report", report)
        print(json.dumps(report, indent=2), flush=True)

    def run(self, pilot_only=False, max_calls=None):
        """Discover, pilot, then continue the full cohort; no default corpus cap."""
        with self.store.writer_lock():
            self.store.set_meta("last_run", {"status": "running", "started_at": now(), "pid": os.getpid()})
            try:
                self._discover()
                self._ensure_models()
                self._select_pilot()
                if not self.store.meta("pilot_report"):
                    if self.store.meta("pilot_baseline_state_bytes") is None:
                        db = self.store.db
                        baseline = db.execute("PRAGMA page_count").fetchone()[0] * db.execute("PRAGMA page_size").fetchone()[0]
                        self.store.set_meta("pilot_baseline_state_bytes", baseline)
                    started = time.monotonic()
                    self._phase(pilot=True, max_calls=100)
                    self._pilot_report(time.monotonic() - started)
                if not pilot_only:
                    self._phase(max_calls=max_calls)
                self.store.set_meta("last_run", {"status": "stopped_after_pilot" if pilot_only else "finished_pass", "finished_at": now()})
            except BaseException as exc:
                self.store.set_meta("last_run", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "blocked",
                                                 "error": str(exc), "finished_at": now()})
                raise
            finally:
                self.store.write_status(force=True)
        return self.store.status()

    def retry_failed(self):
        """Requeue failures explicitly, preserving successful classification work."""
        with self.store.writer_lock(), self.store.db:
            self.store.db.execute("UPDATE tasks SET status='pending',error=NULL WHERE status='failed'")
            self.store.db.execute("UPDATE articles SET status=CASE WHEN retrieval_error IS NOT NULL OR coverage='retrieval_failed' THEN 'pending' ELSE 'ready' END,error=NULL WHERE status='failed' OR retrieval_error IS NOT NULL")
        return self.store.write_status(force=True)

    def status(self):
        return self.store.status()

    def export(self):
        from .pmc_export import export_run
        # WAL permits a consistent read snapshot while the extraction writer runs.
        self.store.db.execute("BEGIN")
        try:
            self.store.check_space(self.min_free_bytes)
            return export_run(self.store, self.min_free_bytes)
        finally:
            self.store.db.rollback()
