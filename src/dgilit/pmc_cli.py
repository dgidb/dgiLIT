"""Command-line entry point: python -m dgilit.pmc_cli --help."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .pmc_pipeline import DEFAULT_OUTPUT, PipelineConfig, PMCPipeline
from .pmc_store import ResourceLimit


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resumable ten-year PMC drug–gene extraction")
    parser.add_argument("command", choices=["discover", "run", "status", "retry-failed", "export"])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start", help="Inclusive publication date YYYY-MM-DD (new runs)")
    parser.add_argument("--end", help="Inclusive publication date YYYY-MM-DD (new runs)")
    parser.add_argument("--prompt-version", choices=["v1", "v2", "v3"], help="Prompt for new runs; v3 is opt-in")
    parser.add_argument("--pilot-only", action="store_true", help="Stop after the initial 100-article / 100-inference pilot")
    parser.add_argument("--max-calls", type=int, help="Optional explicit inference cap for the full phase, after pilot")
    parser.add_argument("--detach", action="store_true", help="Run in a detached process with durable run.log (run only)")
    args = parser.parse_args(argv)
    if args.max_calls is not None and args.max_calls < 0:
        parser.error("--max-calls must be non-negative")
    if bool(args.start) != bool(args.end):
        parser.error("Supply both --start and --end")
    config = PipelineConfig(start=args.start, end=args.end) if args.start else None
    if args.prompt_version:
        from dataclasses import replace
        saved = config or PipelineConfig()
        if config is None and (Path(args.output) / "state.sqlite").exists():
            import sqlite3
            with sqlite3.connect(Path(args.output) / "state.sqlite") as db:
                saved = PipelineConfig(**json.loads(db.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0]))
        config = replace(saved, prompt_version=args.prompt_version)
    pipeline = PMCPipeline(args.output, config=config)
    if args.detach:
        if args.command != "run":
            parser.error("--detach applies only to run")
        command = [sys.executable, "-u", "-m", "dgilit.pmc_cli", "run", "--output", str(pipeline.output.resolve())]
        if args.pilot_only:
            command.append("--pilot-only")
        if args.max_calls is not None:
            command.extend(["--max-calls", str(args.max_calls)])
        with (pipeline.output / "run.log").open("a") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True, cwd=str(pipeline.output.resolve()))
        print(json.dumps({"pid": process.pid, "log": str(pipeline.output / "run.log")}), flush=True)
        return 0
    try:
        if args.command == "discover":
            result = pipeline.discover()
        elif args.command == "run":
            result = pipeline.run(pilot_only=args.pilot_only, max_calls=args.max_calls)
        elif args.command == "retry-failed":
            result = pipeline.retry_failed()
        elif args.command == "export":
            result = pipeline.export()
        else:
            result = pipeline.status()
        print(json.dumps(result, indent=2), flush=True)
    except (ResourceLimit, KeyboardInterrupt) as exc:
        print(json.dumps({"checkpoint_saved": True, "reason": str(exc)}), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
