"""Shared human assessments stored independently of extraction checkpoints."""
from contextlib import closing
import json
import sqlite3
from typing import get_args

from .contracts import ErrorReason, RatingUpdate
from .repository import utcnow


class RatingsUnavailable(RuntimeError):
    pass


class RatingRepository:
    def __init__(self, settings):
        self.path = settings.review_db

    def get_many(self, task_ids):
        if not task_ids or not self.path.is_file():
            return {}
        try:
            with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=5)) as db:
                db.row_factory = sqlite3.Row
                db.execute("PRAGMA query_only=ON")
                if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='claim_ratings'").fetchone() is None:
                    return {}
                placeholders = ",".join("?" for _ in task_ids)
                columns = {row["name"] for row in db.execute("PRAGMA table_info(claim_ratings)")}
                reasons = "error_reasons" if "error_reasons" in columns else "'[]' AS error_reasons"
                return {row["task_id"]: {"rating": row["rating"], "rated_at": row["rated_at"],
                                        "error_reasons": RatingUpdate(rating=row["rating"], error_reasons=json.loads(row["error_reasons"])).error_reasons}
                        for row in db.execute(
                            f"SELECT task_id,rating,rated_at,{reasons} FROM claim_ratings WHERE task_id IN ({placeholders})",
                            task_ids)}
        except (sqlite3.Error, ValueError, TypeError) as exc:
            raise RatingsUnavailable("Shared claim ratings could not be read. Try refreshing.") from exc

    def get(self, task_id):
        return self.get_many([task_id]).get(task_id, {"rating": None, "rated_at": None, "error_reasons": []})

    def set(self, task_id, rating, error_reasons=None):
        # None means omitted by a legacy client, while [] explicitly clears reasons.
        requested = RatingUpdate(rating=rating, error_reasons=error_reasons or []).error_reasons
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
                # Serialize schema inspection/migration and the assessment update together.
                db.execute("BEGIN IMMEDIATE")
                db.execute("""CREATE TABLE IF NOT EXISTS claim_ratings (
                    task_id TEXT PRIMARY KEY,
                    rating TEXT NOT NULL CHECK(rating IN ('up','down')),
                    rated_at TEXT NOT NULL,
                    error_reasons TEXT NOT NULL DEFAULT '[]'
                )""")
                if "error_reasons" not in {row[1] for row in db.execute("PRAGMA table_info(claim_ratings)")}:
                    db.execute("ALTER TABLE claim_ratings ADD COLUMN error_reasons TEXT NOT NULL DEFAULT '[]'")
                if rating == "down" and error_reasons is None:
                    previous = db.execute("SELECT rating,error_reasons FROM claim_ratings WHERE task_id=?", (task_id,)).fetchone()
                    if previous and previous[0] == "down":
                        requested = RatingUpdate(rating="down", error_reasons=json.loads(previous[1])).error_reasons
                selected = set(requested) if rating == "down" else set()
                reasons = [code for code in get_args(ErrorReason) if code in selected]
                rated_at = utcnow() if rating is not None else None
                if rating is None:
                    db.execute("DELETE FROM claim_ratings WHERE task_id=?", (task_id,))
                else:
                    db.execute("""INSERT INTO claim_ratings(task_id,rating,rated_at,error_reasons) VALUES (?,?,?,?)
                        ON CONFLICT(task_id) DO UPDATE SET rating=excluded.rating,rated_at=excluded.rated_at,
                        error_reasons=excluded.error_reasons""", (task_id, rating, rated_at, json.dumps(reasons)))
        except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
            raise RatingsUnavailable("The claim rating could not be saved. Your previous assessment is unchanged.") from exc
        return {"rating": rating, "rated_at": rated_at, "error_reasons": reasons}
