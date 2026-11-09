"""Shared human assessments stored independently of extraction checkpoints."""
from contextlib import closing
import sqlite3

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
                return {row["task_id"]: {"rating": row["rating"], "rated_at": row["rated_at"]}
                        for row in db.execute(
                            f"SELECT task_id,rating,rated_at FROM claim_ratings WHERE task_id IN ({placeholders})",
                            task_ids)}
        except sqlite3.Error as exc:
            raise RatingsUnavailable("Shared claim ratings could not be read. Try refreshing.") from exc

    def get(self, task_id):
        return self.get_many([task_id]).get(task_id, {"rating": None, "rated_at": None})

    def set(self, task_id, rating):
        if rating not in ("up", "down", None):
            raise ValueError("Rating must be up, down, or null")
        rated_at = utcnow() if rating is not None else None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
                db.execute("""CREATE TABLE IF NOT EXISTS claim_ratings (
                    task_id TEXT PRIMARY KEY,
                    rating TEXT NOT NULL CHECK(rating IN ('up','down')),
                    rated_at TEXT NOT NULL
                )""")
                if rating is None:
                    db.execute("DELETE FROM claim_ratings WHERE task_id=?", (task_id,))
                else:
                    db.execute("""INSERT INTO claim_ratings(task_id,rating,rated_at) VALUES (?,?,?)
                        ON CONFLICT(task_id) DO UPDATE SET rating=excluded.rating,rated_at=excluded.rated_at""",
                        (task_id, rating, rated_at))
        except (sqlite3.Error, OSError) as exc:
            raise RatingsUnavailable("The claim rating could not be saved. Your previous assessment is unchanged.") from exc
        return {"rating": rating, "rated_at": rated_at}
