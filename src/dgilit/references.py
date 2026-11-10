"""Deterministic, local DGIdb reference assembly; no runtime spreadsheet reader."""
from __future__ import annotations

import json
from pathlib import Path

REF_DIR = Path(__file__).parent / "ref"
REFERENCE_FILES = ("data_model.md", "druggable_genome.md", "interaction_types.md")


def load_vocabulary(ref_dir: Path = REF_DIR) -> dict[str, dict]:
    path = ref_dir / "interaction_vocabulary.json"
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))["types"]
        if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
            raise ValueError("types must be a list of objects")
        for entry in entries:
            if not {"name", "description", "directionality"} <= entry.keys():
                raise ValueError("type lacks name, description or directionality")
            if not all(isinstance(d, str) and d.strip() for d in entry.get("alternate_descriptions", [])):
                raise ValueError("invalid alternate descriptions")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid DGIdb vocabulary {path}: {exc}") from exc
    vocabulary = {}
    for entry in entries:
        name = entry["name"]
        if (not isinstance(name, str) or not name.strip() or name in vocabulary
                or not isinstance(entry["description"], str) or not entry["description"].strip()
                or entry["directionality"] not in ("activating", "inhibiting", "unclear")):
            raise ValueError(f"Invalid or duplicate interaction type in {path}: {name!r}")
        vocabulary[name] = entry
    if "other/unknown" not in vocabulary:
        raise ValueError(f"Missing other/unknown interaction type in {path}")
    return vocabulary


def assemble_system_prompt(version: str, prompt_dir: Path, ref_dir: Path = REF_DIR) -> str:
    """Also used by the review client to verify the actual system prompt hash."""
    template = (prompt_dir / f"{version}.text").read_text(encoding="utf-8")
    if version != "v3":
        return template
    sections = [template]
    for filename in REFERENCE_FILES:
        content = (ref_dir / filename).read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Empty DGIdb reference: {filename}")
        sections.append(f"DGIdb reference: {filename}\n{content}")
    vocabulary = load_vocabulary(ref_dir)
    sections.append("DGIdb canonical interaction vocabulary:\n" + json.dumps(
        [{"name": name, "description": entry["description"],
          "alternate_descriptions": entry.get("alternate_descriptions", []),
          "directionality": entry["directionality"]}
         for name, entry in sorted(vocabulary.items())], ensure_ascii=False, sort_keys=True))
    return "\n\n".join(sections)
