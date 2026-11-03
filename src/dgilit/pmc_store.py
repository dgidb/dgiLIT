"""Transactional state and bounded source-text cache for corpus runs."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ResourceLimit(RuntimeError):
    pass


class Store:
    def __init__(self, output, config):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache = self.output / "text_cache"
        self.cache.mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.output / "state.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS partitions (
          id TEXT PRIMARY KEY, start TEXT, end TEXT, query TEXT, expected INTEGER,
          received INTEGER DEFAULT 0, status TEXT, discovered_at TEXT, error TEXT);
        CREATE TABLE IF NOT EXISTS articles (
          pmcid TEXT PRIMARY KEY, cohort_year INTEGER, pilot INTEGER DEFAULT 0,
          status TEXT DEFAULT 'pending', coverage TEXT, pmid TEXT, title TEXT,
          publication_year TEXT, license TEXT, text_hash TEXT, section_count INTEGER,
          retrieval_error TEXT, error TEXT, updated_at TEXT);
        CREATE INDEX IF NOT EXISTS article_status ON articles(status, pilot, cohort_year, pmcid);
        CREATE INDEX IF NOT EXISTS article_year ON articles(cohort_year,pmcid);
        CREATE TABLE IF NOT EXISTS cached_text (pmcid TEXT PRIMARY KEY, bytes INTEGER, touched TEXT);
        CREATE TABLE IF NOT EXISTS sections (
          id TEXT PRIMARY KEY, pmcid TEXT, article_hash TEXT, section_index INTEGER,
          text_hash TEXT, metadata TEXT);
        CREATE TABLE IF NOT EXISTS entities (
          id TEXT PRIMARY KEY, section_id TEXT, pmcid TEXT, data TEXT);
        CREATE TABLE IF NOT EXISTS tasks (
          id TEXT PRIMARY KEY, pmcid TEXT, article_hash TEXT, section_id TEXT,
          section_index INTEGER, context_start INTEGER, context_end INTEGER,
          context_hash TEXT, drug_id TEXT, gene_id TEXT, drug TEXT, gene TEXT,
          config_hash TEXT, status TEXT DEFAULT 'pending', result TEXT, error TEXT,
          attempts INTEGER DEFAULT 0, elapsed_seconds REAL, updated_at TEXT);
        CREATE INDEX IF NOT EXISTS task_queue ON tasks(pmcid, article_hash, status, id);
        CREATE TABLE IF NOT EXISTS failures (
          id INTEGER PRIMARY KEY, stage TEXT, item_id TEXT, pmcid TEXT,
          error TEXT, recorded_at TEXT, resolved INTEGER DEFAULT 0);
        CREATE INDEX IF NOT EXISTS failure_item ON failures(stage, item_id, resolved);
        ''')
        previous = self.meta("config")
        if previous and previous != config:
            raise ValueError("Run configuration differs from this checkpoint. Use a different output directory.")
        self.config_hash = digest(config)
        self.set_meta("config", config)

    def meta(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value)))

    @contextmanager
    def writer_lock(self):
        with (self.output / "writer.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another discover/run/export process owns this output directory") from exc
            try:
                # Only the exclusive writer may recover abandoned work.
                existing = {p.name.removesuffix('.json.gz'): p.stat().st_size for p in self.cache.glob('*.gz')}
                with self.db:
                    self.db.execute("DELETE FROM cached_text")
                    self.db.executemany("INSERT INTO cached_text VALUES (?,?,?)", ((p, size, now()) for p, size in existing.items()))
                    self.db.execute("UPDATE tasks SET status='pending' WHERE status='running'")
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def check_space(self, minimum=10 * 1024 ** 3):
        if shutil.disk_usage(self.output).free < minimum:
            raise ResourceLimit(f"Less than {minimum / 1024**3:g} GiB free; checkpoint saved")

    def fail(self, stage, item_id, pmcid, error):
        with self.db:
            self.db.execute("INSERT INTO failures(stage,item_id,pmcid,error,recorded_at) VALUES (?,?,?,?,?)",
                            (stage, item_id, pmcid, str(error), now()))

    def resolve(self, stage, item_id):
        with self.db:
            self.db.execute("UPDATE failures SET resolved=1 WHERE stage=? AND item_id=?", (stage, item_id))

    def cache_get(self, pmcid):
        path = self.cache / f"{pmcid}.json.gz"
        if not path.exists():
            return None
        with gzip.open(path, "rt") as handle:
            return json.load(handle)

    def cache_put(self, article, maximum=10 * 1024 ** 3):
        payload = gzip.compress(article.model_dump_json().encode())
        self.evict_cache(maximum - len(payload), protect=article.pmcid)
        cached_bytes = self.db.execute("SELECT coalesce(sum(bytes),0) FROM cached_text WHERE pmcid!=?", (article.pmcid,)).fetchone()[0]
        if len(payload) + cached_bytes > maximum:
            raise ResourceLimit("Text cache is full of unfinished articles; finish pending work before fetching more")
        path = self.cache / f"{article.pmcid}.json.gz"
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO cached_text VALUES (?,?,?)", (article.pmcid, len(payload), now()))

    def evict_cache(self, maximum, protect=None):
        total = self.db.execute("SELECT coalesce(sum(bytes),0) FROM cached_text").fetchone()[0]
        files = self.db.execute('''SELECT c.pmcid,c.bytes FROM cached_text c JOIN articles a ON a.pmcid=c.pmcid
            WHERE a.status IN ('complete','complete_partial','no_candidates','no_text') ORDER BY c.touched''')
        for pmcid, size in files:
            if total <= maximum:
                break
            if pmcid != protect:
                path = self.cache / f"{pmcid}.json.gz"
                path.unlink(missing_ok=True)
                with self.db:
                    self.db.execute("DELETE FROM cached_text WHERE pmcid=?", (pmcid,))
                total -= size

    def status(self):
        counts = lambda sql: {row[0]: row[1] for row in self.db.execute(sql)}
        articles = counts("SELECT status,count(*) FROM articles GROUP BY status")
        tasks = counts("SELECT t.status,count(*) FROM tasks t JOIN articles a ON t.pmcid=a.pmcid AND t.article_hash=a.text_hash GROUP BY t.status")
        failures = self.db.execute("SELECT count(*) FROM failures WHERE resolved=0").fetchone()[0]
        discovery_complete = self.meta("discovery_complete", False)
        terminal = {"complete", "complete_partial", "no_candidates", "no_text", "failed"}
        cohort_terminal = discovery_complete and not (set(articles) - terminal)
        return {
            "updated_at": now(), "discovery_complete": discovery_complete,
            "corpus_articles": sum(articles.values()), "articles": articles, "tasks": tasks,
            "coverage": counts("SELECT coalesce(coverage,'not_fetched'),count(*) FROM articles GROUP BY coverage"),
            "unresolved_failure_events": failures,
            "all_articles_terminal": cohort_terminal,
            "successful_completion": cohort_terminal and not failures and not articles.get("failed", 0),
            "cache_bytes": self.db.execute("SELECT coalesce(sum(bytes),0) FROM cached_text").fetchone()[0],
            "free_disk_bytes": shutil.disk_usage(self.output).free,
            "pilot": self.meta("pilot_report"),
            "last_run": self.meta("last_run"),
        }

    def write_status(self, force=False):
        if not force and hasattr(self, "_status_at") and time.monotonic() - self._status_at < 30:
            return self._last_status
        value = self.status()
        path = self.output / "status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
        self._status_at, self._last_status = time.monotonic(), value
        return value
