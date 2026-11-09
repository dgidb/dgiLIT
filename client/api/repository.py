"""SQL lives here, independently of HTTP and the extraction implementation."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import sqlite3


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class DatabaseUnavailable(RuntimeError):
    pass


class ContextNotFound(LookupError):
    pass


ACTIVE = " FROM tasks t JOIN articles a ON a.pmcid=t.pmcid AND a.text_hash=t.article_hash "
POSITIVE = " t.status='done' AND json_extract(t.result,'$.interaction')=1 "
COHORT = "CASE WHEN a.pilot=1 THEN 'pilot' ELSE 'beyond_pilot' END"


class Repository:
    def __init__(self, settings):
        self.settings = settings

    @contextmanager
    def connection(self):
        path = self.settings.data_dir / "state.sqlite"
        if not path.is_file():
            raise DatabaseUnavailable("No checkpoint database found. Configure DGILIT_DATA_DIR to an existing CLI run.")
        try:
            db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
        except sqlite3.Error as exc:
            raise DatabaseUnavailable("The checkpoint database could not be opened for reading.") from exc
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            yield db
        except sqlite3.Error as exc:
            raise DatabaseUnavailable("Checkpoint queries are temporarily unavailable or the schema is unsupported. Try refreshing.") from exc
        finally:
            db.rollback()
            db.close()

    @staticmethod
    def metadata(db):
        keys = ("config", "execution_fingerprint", "pilot_report", "discovery_complete", "last_run", "query")
        return {r["key"]: json.loads(r["value"]) for r in db.execute(
            "SELECT key,value FROM meta WHERE key IN (?,?,?,?,?,?)", keys)}

    def health(self):
        with self.connection() as db:
            db.execute("SELECT id FROM tasks LIMIT 1").fetchone()
        return {"status": "ok", "access": "read_only", "dataset": self.settings.data_dir.name}

    def summary(self):
        with self.connection() as db:
            cohorts = [dict(r) for r in db.execute("SELECT " + COHORT + " cohort,count(*) positive_contexts,"
                "count(DISTINCT t.pmcid) supporting_articles" + ACTIVE + "WHERE " + POSITIVE + " GROUP BY a.pilot")]
            for name in ("pilot", "beyond_pilot"):
                if not any(r["cohort"] == name for r in cohorts):
                    cohorts.append({"cohort": name, "positive_contexts": 0, "supporting_articles": 0})
            # The CLI has already computed this large inventory; do not repeatedly
            # scan millions of article rows on every browser page request.
            metadata = self.metadata(db)
            return {"read_at": utcnow(), "dataset": self.settings.data_dir.name, "cohorts": cohorts,
                "positive_contexts": sum(r["positive_contexts"] for r in cohorts),
                "supporting_articles": sum(r["supporting_articles"] for r in cohorts), "metadata": metadata}

    def contexts(self, *, drug="", gene="", article="", cohort="all", coverage="all",
                 rating_filter="all", review_db=None, page=1, page_size=50):
        where = " WHERE " + POSITIVE
        params = []
        for value, fields in [(drug, ["t.drug", "t.drug_id"]), (gene, ["t.gene", "t.gene_id"]),
                (article, ["a.title", "a.pmcid", "a.pmid"])]:
            if value:
                where += " AND (" + " OR ".join(f"instr(lower(coalesce({f},'')),lower(?))>0" for f in fields) + ")"
                params.extend([value] * len(fields))
        if cohort != "all":
            where += " AND a.pilot=?"
            params.append(1 if cohort == "pilot" else 0)
        if coverage != "all":
            where += " AND a.coverage=?"
            params.append(coverage)
        with self.connection() as db:
            if rating_filter != "all":
                has_ratings = False
                if review_db is not None and review_db.is_file():
                    # Filter in SQL before counting/paging. Both databases are
                    # opened read-only; absent review stores are never created.
                    db.execute("ATTACH DATABASE ? AS claim_reviews", (review_db.as_uri() + "?mode=ro",))
                    has_ratings = db.execute("SELECT 1 FROM claim_reviews.sqlite_master "
                        "WHERE type='table' AND name='claim_ratings'").fetchone() is not None
                if has_ratings:
                    if rating_filter == "up":
                        where += " AND t.id IN (SELECT task_id FROM claim_reviews.claim_ratings WHERE rating=?)"
                        params.append("up")
                    else:
                        where += " AND NOT EXISTS (SELECT 1 FROM claim_reviews.claim_ratings r WHERE r.task_id=t.id AND r.rating=?)"
                        params.append("down")
                elif rating_filter == "up":
                    where += " AND 0"
            total = db.execute("SELECT count(*)" + ACTIVE + where, params).fetchone()[0]
            rows = [dict(r) for r in db.execute("""SELECT t.id task_id,t.pmcid,a.pmid,a.title,a.publication_year,
                a.coverage,a.status article_status,t.drug,t.drug_id,t.gene,t.gene_id,t.updated_at,
                json_extract(t.result,'$.evidence') evidence,
                json_extract(t.result,'$.evidence_in_context') evidence_verified,
                json_extract(t.result,'$.interaction_type') interaction_type,
                json_extract(t.result,'$.directionality') directionality,
                json_extract(s.metadata,'$.label') section_label,
                json_extract(s.metadata,'$.section_type') section_type,""" + COHORT + " cohort" + ACTIVE +
                " LEFT JOIN sections s ON s.id=t.section_id " + where +
                " ORDER BY t.updated_at DESC,t.id DESC LIMIT ? OFFSET ?", [*params,page_size,(page-1)*page_size])]
        return {"items": rows, "total": total, "page": page, "page_size": page_size, "read_at": utcnow()}

    def detail(self, task_id):
        with self.connection() as db:
            row = db.execute("""SELECT t.*,a.pmid,a.title,a.publication_year,a.license,a.coverage,
                a.status article_status,a.text_hash current_article_hash,""" + COHORT + " cohort," +
                "s.text_hash section_hash,s.metadata section_metadata FROM tasks t JOIN articles a ON a.pmcid=t.pmcid "
                "LEFT JOIN sections s ON s.id=t.section_id WHERE t.id=? AND " + POSITIVE, (task_id,)).fetchone()
            if row is None:
                raise ContextNotFound("This positive context was not found.")
            task = dict(row)
            task["result"] = json.loads(task["result"])
            task["section_metadata"] = json.loads(task["section_metadata"]) if task["section_metadata"] else None
            task["is_current"] = task["article_hash"] == task["current_article_hash"]
            entities = []
            for entity in db.execute("SELECT id,data FROM entities WHERE section_id=? ORDER BY id", (task["section_id"],)):
                data = json.loads(entity["data"])
                if (data.get("concept") or {}).get("concept_id") in (task["drug_id"], task["gene_id"]):
                    entities.append({"id": entity["id"], **data})
            related = [dict(r) for r in db.execute("""SELECT status,count(*) contexts,
                sum(CASE WHEN status='done' AND json_extract(result,'$.interaction')=1 THEN 1 ELSE 0 END) positive,
                sum(CASE WHEN status='done' AND json_extract(result,'$.interaction')=0 THEN 1 ELSE 0 END) negative
                FROM tasks WHERE pmcid=? AND article_hash=? AND drug_id=? AND gene_id=? GROUP BY status""",
                (task["pmcid"],task["article_hash"],task["drug_id"],task["gene_id"]))]
            metadata = self.metadata(db)
        return {"task": task, "entities": entities, "related_contexts": related, "metadata": metadata,
                "read_at": utcnow(), "dataset": self.settings.data_dir.name}
