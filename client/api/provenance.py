import hashlib
import json
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def assemble_provenance(detail, settings):
    task = detail["task"]
    fingerprint = detail["metadata"].get("execution_fingerprint") or {}
    expected = fingerprint.get("prompt_hash")
    version = task["result"].get("prompt_version")
    prompt = {"version": version, "saved_hash": expected, "status": "unavailable", "template": None}
    if isinstance(version, str) and re.fullmatch(r"[A-Za-z0-9_-]+", version):
        try:
            if version == "v3":
                from dgilit.references import assemble_system_prompt
                template = assemble_system_prompt(version, settings.prompt_dir, settings.prompt_dir.parent / "ref")
            else:
                template = (settings.prompt_dir / f"{version}.text").read_text()
            if expected and digest(template) == expected:
                prompt.update(status="verified", template=template)
            else:
                prompt["status"] = "hash_mismatch"
        except (OSError, ValueError, KeyError, TypeError, ImportError):
            pass
    # Original tokenizer-rendered messages were not persisted by the producer.
    prompt["note"] = "This is the verified template when available; original rendered model messages were not saved."
    detail["prompt"] = prompt
    detail["links"] = {"pmc": f"https://pmc.ncbi.nlm.nih.gov/articles/{task['pmcid']}/"}
    if task.get("pmid"):
        detail["links"]["pubmed"] = f"https://pubmed.ncbi.nlm.nih.gov/{task['pmid']}/"
    return detail
