# PMC interaction extraction: workflow and CLI

This workflow finds candidate drug–gene interactions in PMC papers published
**November 3, 2016 through November 3, 2026**, inclusive. It uses PMC text
retrieval, BioBERT entity recognition, VICC concept normalization, and local
Qwen3-14B classification. The **CLI owns all data generation**. The
[analysis notebook](01_pmc_interactions.ipynb) reads its existing checkpoints
through read-only SQLite connections; it cannot start discovery, retrieval,
NER, inference, retries, or exports.

The date filter is **publication date**, not the date a paper was added to PMC.
Discovery includes every matching PMC record, including papers without a PubMed
ID. There is no gene panel, disease filter, or citation threshold.

## Using the analysis notebook while the CLI runs

Open `01_pmc_interactions.ipynb` and run all cells to analyze the current saved
state. Rerun all cells to refresh. The notebook requires pandas, matplotlib, and
a Jupyter/IPython environment; it does not load models or need network access.
Its default output path resolves from this checkout, or you can set `OUTPUT_DIR`
to the absolute path of another existing CLI run.

The notebook provides:

- Live configuration, discovery state, and the pilot report when available.
- Pilot, beyond-pilot, and combined progress/coverage summaries.
- Positive, negative, and unresolved context/article-pair counts.
- Interaction, gene, and drug rankings by distinct supporting articles.
- A paginated evidence browser covering all saved positives by default, with
  cohort, gene, and drug filters and a total match count.
- Saved source-context inspection when the text is still cached, with its hash
  checked before display; no text is fetched if the cache was evicted.
- Failure summaries and a preview of unresolved events.

Set `BROWSER_COHORT`, `GENE_FILTER`, `DRUG_FILTER`, and `DISPLAY_LIMIT` in the
configuration cell to filter the evidence browser; global summaries retain all
saved work. The default `BROWSER_COHORT = "all"` includes pilot and non-pilot
articles, including saved positive contexts from papers still being processed.
For 50-row pages, set `EVIDENCE_OFFSET` to 0, 50, 100, and so on. The browser shows
the total matching positives and the displayed row range. New results may shift
page positions when you refresh; task IDs are stable.

Set `SELECTED_TASK_ID` to any active saved task ID and run the detail cell to
inspect it directly. This lookup is independent of browser filters, offset, and
page size, and can inspect negative, failed, or unfinished entries too. Leave it
unset to select the first positive row on the current page. Details include the
full evidence quote, complete structured model result and generation statistics,
article metadata/cohort, normalized entities, section provenance, and an aggregate
of every saved context for the same article pair. Full cached source text is
displayed only after its context hash is verified. Detail lookup reads the latest
checkpoint and may be newer than the summary snapshot.

The notebook handles incomplete discovery, an unfinished pilot, and
an absence of positive results without launching work to fill those gaps.

Connections use `mode=ro` and `query_only=ON`, and close after reading a consistent
snapshot, before plots/tables are displayed. Queries aggregate in SQLite and
limit evidence/failure previews instead of loading the corpus into memory.
Aggregations on a large checkpoint can still take time. The notebook reads the
live database directly; no Parquet export is required.

The **All** group includes the pilot; **Beyond pilot** excludes it. Early cohort
differences can reflect different processing stages. Positive fractions measure
model output composition, not precision, recall, or biological recovery.

## How the workflow runs

1. **Discover articles.** Search PMC by publication date and save PMCIDs in
   SQLite. Large searches are split into smaller date ranges. If one date still
   exceeds the search limit, split it into UID ranges. Verify returned counts
   before marking a partition complete. Each leaf uses the count and IDs from
   the same response, so live index changes between separate requests cannot
   invalidate an otherwise complete result. Incomplete responses are retried up
   to three times before the run records a failure.
2. **Select the pilot.** After discovery finishes, choose up to 100 articles
   deterministically across the discovered publication-year strata. Retrieve and
   prepare their text and measure at most 100 Qwen context tasks initially.
3. **Retrieve text.** Fetch PMC XML in batches of up to 25 articles. Preserve all
   parsed abstract/body sections, their labels, hierarchy, and source. When PMC
   text is unavailable, try an available PubMed abstract for a linked PMID.
   Full-text, abstract-only, unavailable, and failed retrieval are distinguished.
4. **Identify entities.** Run the BioBERT chemical and gene NER models on each
   section. Long sections use overlapping token windows, with offsets restored
   to the original section. Normalize mentions with VICC and cache successful
   responses, including successful no-match responses. Temporary normalization
   errors remain retryable.
5. **Generate candidates.** Within each section, form every distinct normalized
   drug × gene pair. Both candidates must have a usable concept ID and label.
   Raw and unnormalized mentions are retained for auditing but do not enter
   classification. Genes come from NER, not curated labels.
6. **Classify interactions.** Load Qwen once and process pairs serially on Metal.
   Use prompt `v2`, temperature `0`, thinking disabled, and a 1,000-token output
   budget. Use the complete section when it fits; otherwise split it with
   256-token overlap and evaluate every section pair against every context.
   Positive evidence must be a quote verifiable in its source context.
7. **Continue the corpus.** Save the pilot report, then process the remaining
   cohort. Reuse the pilot's successful work. There is no default article cap.
8. **Export a snapshot.** Run `export` whenever you want Parquet datasets. The
   extraction saves results continuously to SQLite; Parquet export is a separate
   command, not an automatic consequence of running inference.

A classification task is one **drug–gene pair in one text context**, not one
article. A section containing two normalized drugs and three genes creates six
pairs. If it needs three Qwen contexts, that section creates 18 tasks. This is
why article counts and task counts differ substantially.

The pipeline does not infer interactions by combining evidence across separate
sections. PDFs and supplementary attachments are not processed. PMC availability
also limits which full texts can be retrieved automatically.

## Setup

Run these commands from the `dgiLIT/` repository root, in the Python environment
you will also use for the notebook:

```bash
python --version  # Python 3.13 or newer
python -m pip install -e '.[pmc,qwen]'
```

Local Qwen inference requires **native arm64 Python on Apple Silicon macOS with
Metal available**. Initial model use may download weights to the Hugging Face
cache. NCBI retrieval and VICC normalization require network access.

Set contact information in the shell before starting the process:

```bash
export MC_EMAIL="your.email@example.org"
# NCBI_EMAIL is also accepted; MC_EMAIL takes precedence.
# Optional: export NCBI_API_KEY="your-key"
```

The CLI reads environment variables directly; it does **not** automatically load
a `.env` file. No PubTator annotation file is required.

Use `python -m dgilit.pmc_cli` in the examples below. After installation,
`dgilit-pmc` is equivalent. Use the Python interpreter containing the installed
MLX, NER, and dgiLIT dependencies.

## Commands and a typical run

If an extraction is already running, start by checking `status`; do not launch
a second writer against its output directory.

```bash
python -m dgilit.pmc_cli status
python -m dgilit.pmc_cli --help
```

For a new run, you can inspect the pilot before continuing:

```bash
python -m dgilit.pmc_cli discover
python -m dgilit.pmc_cli run --pilot-only
python -m dgilit.pmc_cli status
python -m dgilit.pmc_cli run --detach
```

Alternatively, `run` alone performs discovery, the initial pilot, and the full
corpus in sequence. You do not need to run `discover` separately.

| Command | What it does |
| --- | --- |
| `discover` | Discovers missing partitions and checkpoints article IDs. Does not load Qwen or run NER. |
| `run` | Completes discovery, runs the pilot if no pilot report exists, then processes the full cohort. Runs in the foreground by default. |
| `run --pilot-only` | Stops after the initial pilot. Still completes discovery first. If the pilot report already exists, it does not repeat the pilot. |
| `run --detach` | Starts an independent background process, prints its PID and log path, and returns to the shell. Appends output to `run.log`. |
| `status` | Reads current checkpoint state and prints JSON. Does not start inference or rewrite `status.json`. |
| `retry-failed` | Requeues failed work while retaining successful task results. Run `run` afterward to execute the retries. |
| `export` | Creates a consistent Parquet snapshot. Can run while extraction continues. |

`run --max-calls 100` limits the **full phase** to 100 additional context tasks.
It does not limit discovery or NER preparation, and it does not replace the
initial pilot's separate 100-task allowance. These controls can be combined:

```bash
python -m dgilit.pmc_cli run --pilot-only --detach
```

### Output directories and custom dates

The default directory in this checkout is:

```text
data/pmc_last10years_2016-11-03_2026-11-03/
```

Every command accepts `--output PATH`. Supply the same path when monitoring,
resuming, retrying, or exporting that run. Relative paths are interpreted from
your current directory. If using an installed package outside this checkout,
choose an explicit persistent output directory.

To create a separate run for a different date range:

```bash
python -m dgilit.pmc_cli discover \
  --start 2025-01-01 --end 2025-12-31 \
  --output data/pmc_2025

python -m dgilit.pmc_cli run --output data/pmc_2025 --detach
python -m dgilit.pmc_cli status --output data/pmc_2025
```

Both date arguments are required together. Later commands reuse the saved
configuration when dates are omitted. Configuration changes require a new output
directory. Model revisions, prompt contents, and inference package versions are
also checked before inference resumes.

## Reading `status.json`

During processing, `status.json` is an atomic progress snapshot refreshed at
checkpoints, throttled to approximately once every 30 seconds, and at run
boundaries. Long individual operations can leave it unchanged for longer. The
`status` command queries SQLite directly for a fresher view. Times are in UTC.

| Field | Meaning |
| --- | --- |
| `discovery_complete` | All discovery partitions finished successfully. |
| `corpus_articles` | Unique article IDs saved so far; this is a partial cohort count until discovery completes. |
| `articles` | Article counts by processing state; see the next table. |
| `tasks` | Context-classification counts by state for the active article text hashes. |
| `coverage` | Counts of `not_fetched`, `full_text`, `abstract_only`, `no_text`, or `retrieval_failed`. |
| `unresolved_failure_events` | Unresolved failure log entries, not necessarily unique papers. |
| `pilot` | Pilot throughput, coverage, candidate/task volume, and storage estimates; `null` until its report is saved. |
| `cache_bytes` | Compressed source-text cache size in bytes. |
| `free_disk_bytes` | Free space on the output filesystem in bytes. |
| `last_run` | Last run's state, start/finish time, and PID when recorded as running. |
| `all_articles_terminal` | Discovery completed and every article has a terminal outcome, including recorded failures. |
| `successful_completion` | All articles are terminal, with no failed articles or unresolved failure events. This can still include `no_text` and `no_candidates` outcomes. |

| Article state | Meaning |
| --- | --- |
| `pending` | Discovered; retrieval has not completed. |
| `ready` | Retrieved text awaits NER/candidate/task preparation. |
| `prepared` | Candidate tasks have been created; classification remains. |
| `complete` | All required tasks completed. |
| `complete_partial` | Classification completed on available text, but a retrieval error remains recorded. |
| `no_candidates` | Preparation produced no normalized drug–gene pairs. |
| `no_text` | Retrieval finished with no usable text. |
| `failed` | Retrieval, preparation, or classification encountered a failure requiring inspection or retry. |

Task states are `pending`, `running`, `done`, and `failed`.

**While discovery runs, `articles.pending` can increase while `tasks` stays empty
and `pilot` stays `null`. This is expected.** The workflow discovers the cohort
before preparing the pilot or starting classification.

`last_run.status` can be `running`, `stopped_after_pilot`, `finished_pass`,
`interrupted`, or `blocked`. A finished pass can still leave failed work or
pending tasks after an explicit call cap. A stored PID/status is not a process
heartbeat; after a crash, verify the process and check the log before assuming
it is still active.

## Monitoring, stopping, and resuming

For a detached run using the default directory:

```bash
tail -f data/pmc_last10years_2016-11-03_2026-11-03/run.log
```

Pressing Ctrl-C on `tail` only stops viewing the log. It does not stop the
extractor. A detached process survives terminal closure, but still depends on
the Mac remaining available; detaching does not prevent sleep or reboot.

For a foreground `run`, press Ctrl-C to interrupt and checkpoint. For a detached
run, find `last_run.pid` with `status`, verify that it belongs to the extractor,
and send that process SIGINT. Replace `PID` below with the verified numeric PID:

```bash
ps -p PID -o pid=,etime=,command=
kill -INT PID
```

Wait for the writer to exit, then resume with the same output directory:

```bash
python -m dgilit.pmc_cli run --detach
```

Completed discovery partitions and successful classifications are reused.
Abandoned `running` tasks return to the queue when the next writer acquires the
lock. At most one discovery/extraction/retry writer may own a run directory.
There is no CLI stop subcommand.

### Retrying failures

After the current writer stops:

```bash
python -m dgilit.pmc_cli retry-failed
python -m dgilit.pmc_cli run --detach
```

NCBI requests already retry transient failures up to five times, honor
`Retry-After`, and share a limiter below three requests per second within the
pipeline process. Persistent failures remain recorded. `retry-failed` requeues
failed classification and article work; incomplete discovery partitions are
retried when `discover` or `run` resumes. Successful results are retained.

The CLI exits with code `2` for a handled interruption or resource limit. Other
errors may exit with a traceback; inspect `run.log` and the recorded status.
For detached runs, the launch command's return only confirms process launch;
later errors appear in the child process's log/status.

## Files and exports

| File/directory | Purpose |
| --- | --- |
| `state.sqlite` | Authoritative article inventory, partitions, section provenance, entities, tasks, results, failures, and run metadata. |
| `state.sqlite-wal`, `state.sqlite-shm` | SQLite's active transaction files, when present. Keep them with an active database; do not delete them to reclaim space. |
| `normalizer.sqlite` | Persistent VICC normalization cache. |
| `text_cache/` | Compressed parsed source articles, bounded to 10 GiB. |
| `status.json` | Most recently written progress snapshot. |
| `run.log` | Combined output/error log for detached runs. |
| `writer.lock` | Writer coordination file; its existence alone does not mean a writer is active. |
| `exports/latest.json` | Pointer to the newest successfully completed export. |
| `exports/<generation>/manifest.json` | Snapshot configuration, provenance, status, and dataset row/part counts. |

Create a snapshot explicitly:

```bash
python -m dgilit.pmc_cli export
```

Exports use Zstandard-compressed Parquet shards of up to 10,000 rows each.
`latest.json` changes only after all datasets and the manifest finish. Each
export creates a new snapshot; old snapshots consume space separately.

| Dataset | Contents |
| --- | --- |
| `articles` | Article metadata, coverage, state, and source-text hash. |
| `partitions` | Discovery query ranges, expected/received counts, and status. |
| `sections` | Active section labels, hierarchy, source, indices, and hashes. |
| `entities` | Raw mentions, offsets, normalized identifiers/labels, and structured entity records. |
| `classifications` | All active context tasks, including positives, explicit negatives, pending/failed states, evidence, timings, and model/prompt metadata. |
| `article_pairs` | Aggregated drug–gene pairs with `positive`, `negative`, or `unresolved` status. |
| `positive_interactions` | Article pairs with at least one supported positive context. |
| `failures` | Failure history, including whether each event has been resolved. |

For an article pair, any supported context makes the pair positive. A negative
requires all required contexts to complete negatively and article processing to
finish. Missing/failed work remains unresolved. A positive pair can still have
unresolved additional contexts: inspect `unresolved_contexts` and
`article_status`. Use `supporting_task_ids` to locate its evidence in
`classifications`. Structured payloads are preserved in JSON string columns.

Read a small result sample without loading the full dataset:

```python
import json
from pathlib import Path
import polars as pl

output = Path("data/pmc_last10years_2016-11-03_2026-11-03")
pointer = json.loads((output / "exports/latest.json").read_text())
snapshot = Path(pointer["manifest"]).parent

positives = pl.scan_parquet(str(snapshot / "positive_interactions/*.parquet"))
print(positives.head(20).collect())
```

The source text itself is kept in the bounded cache, not copied into Parquet
section records. Evidence quotes, hashes, and offsets persist after eviction;
reproducing an entire evicted context requires retrieving the source again.

## Runtime, storage, and interpretation

The 100-article pilot estimates inference time and checkpoint storage from a
small stratified sample. Its inference estimate excludes retrieval/NER time and
can vary substantially with section length and candidate counts. It is not a
completion-time guarantee. Millions of papers can create many more millions of
classification tasks on one Mac.

The 10-GiB text-cache limit does **not** cap total output size. SQLite, model
weights, normalization caches, logs, and Parquet snapshots use additional space.
Completed article text can be evicted first; unfinished article text is retained.
Processing checkpoints if free space falls below 10 GiB or the cache cannot fit
unfinished work. Free space or choose a sufficiently sized storage location
before resuming; do not delete checkpoint tables or active SQLite files.

These outputs are **model-predicted interactions for expert review**. They do not
establish precision or recall without a labeled truth set. Abstract-only coverage,
NER/normalization exclusions, unavailable text, and unresolved failures all affect
what can be recovered. Interpret completion flags together with coverage and
failure counts.
