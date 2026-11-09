from dataclasses import dataclass
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    prompt_dir: Path = REPO_ROOT / "src/dgilit/dgilit_prompts"
    summary_ttl: float = 30
    source_ttl: float = 600
    review_db: Path = CLIENT_ROOT / ".local/reviews.sqlite"

    def __post_init__(self):
        data_dir = self.data_dir.expanduser().resolve()
        review_db = self.review_db.expanduser().resolve()
        if review_db.is_relative_to(data_dir):
            raise ValueError("DGILIT_REVIEW_DB must be outside the extraction run directory")
        object.__setattr__(self, "data_dir", data_dir)
        object.__setattr__(self, "review_db", review_db)

    @classmethod
    def from_environment(cls):
        default = REPO_ROOT / "data/pmc_last10years_2016-11-03_2026-11-03"
        return cls(Path(os.getenv("DGILIT_DATA_DIR", str(default))),
                   review_db=Path(os.getenv("DGILIT_REVIEW_DB", str(CLIENT_ROOT / ".local/reviews.sqlite"))))
