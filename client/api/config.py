from dataclasses import dataclass
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    prompt_dir: Path = REPO_ROOT / "src/dgilit/dgilit_prompts"
    summary_ttl: float = 30
    source_ttl: float = 600

    @classmethod
    def from_environment(cls):
        default = REPO_ROOT / "data/pmc_last10years_2016-11-03_2026-11-03"
        return cls(Path(os.getenv("DGILIT_DATA_DIR", str(default))).expanduser().resolve())
