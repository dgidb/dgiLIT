"""Paired prompt evaluation on explicitly reviewed, context-level labels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .dgilit_models import InteractionClassificationPrompt, classification_result_model, validate_classification
from .pmc_store import digest


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def reviewed_cases(rows):
    cases = [r for r in rows if r.get("review_status") == "reviewed"]
    ids = [r["id"] for r in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("Benchmark needs reviewed cases with unique IDs")
    for r in cases:
        if type(r.get("expected_interaction")) is not bool:
            raise ValueError(f"Missing boolean human label: {r['id']}")
        if not all(isinstance(r.get(k), str) and r[k].strip() for k in ("drug", "gene", "context", "reviewer")):
            raise ValueError(f"Missing context, candidates or reviewer: {r['id']}")
    return cases


def run_comparison(rows, client, versions=("v2", "v3")):
    cases = reviewed_cases(rows)
    records = []
    for version in versions:
        prompt = InteractionClassificationPrompt(version)
        system = prompt.build_system_prompt()
        schema = classification_result_model(version).model_json_schema()
        for index, case in enumerate(cases, 1):
            record = {"id": case["id"], "version": version, "case_hash": digest(case),
                      "prompt_hash": digest(system), "valid": False}
            payload = prompt.build_payload(case["context"], case["drug"], case["gene"])
            try:
                # Identical complete contexts; never silently truncate or compare different windows.
                response = client.invoke_json(system, prompt.build_user_prompt(payload), schema)
                record["raw_result"] = response.parsed_json
                result = validate_classification(response.parsed_json, version=version,
                    context=case["context"], candidate_drug=case["drug"], candidate_gene=case["gene"])
                record.update(valid=True, result=result.model_dump())
            except Exception as exc:
                record["error"] = str(exc)
            # Qwen clears these per invocation, including failures before generation.
            record["generation_stats"] = dict(getattr(client, "last_generation_stats", {}) or {})
            records.append(record)
            print(f"{version}: {index}/{len(cases)} ({'valid' if record['valid'] else 'invalid'})", flush=True)
    return records


def load_v2_baseline(directory, rows, metadata):
    """Reuse measured v2 outputs only when the entire comparison setup matches."""
    directory = Path(directory)
    saved = json.loads((directory / "metadata.json").read_text())
    for key in ("model_id", "model_revision", "max_tokens", "temperature",
                "enable_thinking", "benchmark_sha256", "packages"):
        if saved.get(key) != metadata[key]:
            raise ValueError(f"Cannot reuse v2 baseline: {key} changed")
    records = [r for r in read_jsonl(directory / "results.jsonl") if r["version"] == "v2"]
    cases = {c["id"]: c for c in reviewed_cases(rows)}
    if len(records) != len(cases) or {r["id"] for r in records} != set(cases):
        raise ValueError("Cannot reuse v2 baseline: case IDs differ")
    expected = digest(InteractionClassificationPrompt("v2").build_system_prompt())
    for record in records:
        if record["case_hash"] != digest(cases[record["id"]]) or record["prompt_hash"] != expected:
            raise ValueError("Cannot reuse v2 baseline: context, labels or prompt changed")
    return records


def summarize(rows, records):
    cases = {c["id"]: c for c in reviewed_cases(rows)}
    results, report = {}, {}
    for version in ("v2", "v3"):
        selected = [r for r in records if r["version"] == version]
        if len(selected) != len(cases) or {r["id"] for r in selected} != set(cases):
            raise ValueError("Comparison must contain exactly one result per case and version")
        results[version] = {r["id"]: r for r in selected}
        counts, fp_categories = Counter(), Counter()
        tokens = Counter()
        for r in selected:
            c = cases[r["id"]]
            if r["case_hash"] != digest(c):
                raise ValueError("Benchmark changed since inference")
            for k in ("prompt_tokens", "generation_tokens"):
                tokens[k] += r.get("generation_stats", {}).get(k, 0)
            if c["expected_interaction"]:
                for field in ("interaction_type", "directionality"):
                    if c.get("expected_" + field) is not None:
                        counts[field + "_total"] += 1
            if not r["valid"]:
                counts["invalid"] += 1
                continue
            actual, expected = r["result"]["interaction"], c["expected_interaction"]
            counts[("tp" if expected else "fp") if actual else ("fn" if expected else "tn")] += 1
            if actual and not expected:
                fp_categories[c.get("category") or "uncategorized"] += 1
            if expected:
                for field in ("interaction_type", "directionality"):
                    if c.get("expected_" + field) is not None:
                        counts[field + "_correct"] += int(actual and r["result"].get(field) == c["expected_" + field])
        def ratio(n, d):
            return n / d if d else None
        positives = sum(c["expected_interaction"] for c in cases.values())
        report[version] = {"counts": dict(counts), "precision": ratio(counts['tp'], counts['tp'] + counts['fp']),
            # Invalid positive cases count as unrecovered, never as true negatives.
            "recall": ratio(counts['tp'], positives), "invalid_output_rate": counts['invalid'] / len(cases),
            "false_positive_categories": dict(fp_categories), "token_usage": dict(tokens),
            **{field + "_accuracy": ratio(counts[field + '_correct'], counts[field + '_total'])
               for field in ("interaction_type", "directionality")}}
    missed = [key for key, c in cases.items() if c["expected_interaction"]
              and results['v2'][key]['valid'] and results['v2'][key]['result']['interaction']
              and (not results['v3'][key]['valid'] or not results['v3'][key]['result']['interaction'])]
    better = (report['v3']['precision'] is not None and report['v2']['precision'] is not None
              and report['v3']['precision'] > report['v2']['precision']
              and report['v3']['counts'].get('fp', 0) < report['v2']['counts'].get('fp', 0)
              and report['v3']['invalid_output_rate'] <= report['v2']['invalid_output_rate'])
    report.update(newly_missed_positive_ids=missed,
                  newly_missed_known_ids=[k for k in missed if cases[k].get('known_interaction')],
                  quantitative_gate_passed=better,
                  promotion_status="human review required; v3 remains opt-in")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory for results and report")
    parser.add_argument("--model-id", default="mlx-community/Qwen3-14B-4bit")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--reuse-v2-from", type=Path, help="Reuse a verified matching v2 run; infer only v3")
    args = parser.parse_args(argv)
    rows = read_jsonl(args.benchmark)
    reviewed_cases(rows)  # Validate before loading weights.
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"model_id": args.model_id, "max_tokens": args.max_tokens, "temperature": 0,
                "enable_thinking": False, "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest()}
    from .pmc_pipeline import _model_revision
    import importlib.metadata
    metadata.update(model_revision=_model_revision(args.model_id),
                    packages={p: importlib.metadata.version(p) for p in ('mlx', 'mlx-lm', 'pydantic', 'wags-llm')})
    baseline = load_v2_baseline(args.reuse_v2_from, rows, metadata) if args.reuse_v2_from else []
    if args.reuse_v2_from:
        metadata.update(reused_v2_from=str(args.reuse_v2_from),
                        baseline_results_sha256=hashlib.sha256((args.reuse_v2_from / "results.jsonl").read_bytes()).hexdigest())
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + '\n')
    (args.output / "system_prompts.json").write_text(json.dumps(
        {v: InteractionClassificationPrompt(v).build_system_prompt() for v in ('v2', 'v3')}, indent=2) + '\n')
    from .qwen import Qwen3JsonClient
    client = Qwen3JsonClient(args.model_id, args.max_tokens, 0.0)
    records = baseline + run_comparison(rows, client, versions=("v3",) if baseline else ("v2", "v3"))
    (args.output / "results.jsonl").write_text(''.join(json.dumps(r) + '\n' for r in records))
    (args.output / "report.json").write_text(json.dumps(summarize(rows, records), indent=2) + '\n')


if __name__ == "__main__":
    main()
