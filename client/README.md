# dgiLIT evidence explorer

A localhost browser for AI-discovered positive drug–gene contexts. The extraction
CLI generates the data; this application reads its existing checkpoints and
never starts discovery, NER, inference, retries, or exports.

The default view includes **both pilot and non-pilot articles**, including saved
positive contexts from papers whose other tasks remain unfinished. Each row is
one drug/gene pair assessed in one text context, rather than a unique interaction
or article. Model predictions are intended for expert review.

## Start locally

Requirements: Node.js 22.13 or newer, Python 3.13 or newer, Git, and an existing CLI
checkpoint directory. The browser does not need MLX, GPUs, model weights, or NCBI
credentials for browsing.

From this directory (`dgiLIT/client/`):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r api/requirements.txt
npm ci
npm run dev
```

Open **http://127.0.0.1:5173**. The API listens on **127.0.0.1:8000**. The launcher
prints both URLs and shuts down its own web/API processes on Ctrl-C. Closing
this application does not stop the independently running extraction CLI.

The isolated `.venv` keeps the extractor's installed dependencies unchanged.
The API installs this repository as an editable package to reuse its PMC parsing
and retrieval implementation. Requirements pin the compatible WAGS source
revision used by that implementation; no model is loaded during browsing.
On Windows, use `.venv\Scripts\python.exe` for the Python commands.

## Configure a run

By default the backend reads:

```text
../data/pmc_last10years_2016-11-03_2026-11-03/
```

The default resolves from the source location, independently of the current
shell directory. To inspect another existing run, set an absolute path:

```bash
export DGILIT_DATA_DIR="/absolute/path/to/a/cli-run"
npm run dev
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `DGILIT_DATA_DIR` | Existing ten-year run in this checkout | Directory containing `state.sqlite` and optional `text_cache/`. |
| `DGILIT_API_PORT` | `8000` | Local API port used by the combined launcher. |
| `PORT` | `5173` | Local frontend port. |
| `DGILIT_API_ORIGIN` | `http://127.0.0.1:8000` | Server-side upstream for the same-origin frontend gateway; the launcher adjusts it to `DGILIT_API_PORT`. |
| `PYTHON_BIN` | `.venv/bin/python` (Windows: `.venv/Scripts/python.exe`) | Interpreter used by the combined launcher. |
| `MC_EMAIL` / `NCBI_EMAIL` | Unset | Optional contact information for explicit PMC source retrieval. |
| `NCBI_API_KEY` | Unset | Optional API key for explicit NCBI requests. |

Export environment variables in the shell before starting. The launcher does not
load a `.env` file. No data-directory path can be supplied through a browser API
request. Missing/incompatible databases produce a readable error; the application
does not create a run or apply migrations.

## Explore results

- Search by drug or gene label/concept ID and by article title, PMCID, or PMID.
- Choose all articles, pilot only, or beyond-pilot results, and full-text or
  abstract-only coverage.
- Use 50-row server-side pages. Counts and displayed ranges describe every
  matching active positive context, not just a preview or the pilot.
- Search state and page selection live in the URL. Open an entry and use **Back
  to results** to restore your filters.
- Use **Refresh data** to load new saved results without automatically changing
  the page while you read. Newly saved data can shift pagination positions.
- Summary cards describe the entire processed dataset, even when the result
  table is filtered. Summaries are cached for 30 seconds; manual refresh bypasses
  that cache. Each result page shows when it was read.

Only completed positive tasks associated with the article's current text appear
in the list. Entry URLs use the full stable task ID:

```text
/contexts/<64-character-task-id>
```

A historical positive task remains accessible directly and is labeled when its
article text has been superseded. Article inventory metadata reflects the current
checkpoint; original task results, section records, and hashes remain attached to
that task's original version.

## Detail pages and provenance

A detail page includes the saved quote, evidence-verification flag, interaction
type, directionality, article identifiers, publication year, license, coverage,
cohort, processing state, and PMC/PubMed links. It also shows:

- Requested normalized concepts and available NER mentions, scores, spans,
  sources, and normalization records.
- Section label, parent label, original/internal section IDs, source, index,
  and context character offsets.
- Task ID, attempts, timing, article/section/context hashes, configuration
  fingerprint, and read timestamp.
- Complete saved classifier JSON and generation statistics.
- Saved model revisions, package versions, run configuration, and discovery query.
- Prompt version and saved hash. A template is displayed only if the current
  local file matches that hash. Original tokenizer-rendered messages were not
  saved by the extractor and are not presented as reconstructed originals.
- Context counts and positive/negative results for the same article pair.

**Copy task ID** copies the full identifier. **Provenance JSON** downloads the
currently displayed record and available source context directly in the browser;
it creates no server-side export or checkpoint file.

A positive context may be visible while the paper still has pending or failed
work. A negative context means the model found no support for that pair in that
particular text; it does not establish that no biological interaction exists.

## Cached and fetched source text

Opening a detail page checks the CLI's existing compressed text cache. Both the
section hash and the extracted context hash must match before the text is shown
as the original model input. The parser uses the same JSON/SHA-256 hashing rules
as the extractor.

If the source is missing or mismatched, click **Fetch source context**. This is
an explicit network operation through the existing supported PMC retrieval code,
with PubMed abstract fallback when the original section came from PubMed.

Retrieval uses per-request 10-second timeouts, one transient retry, and an NCBI
request limiter. Several requests may be required for a fallback. One upstream
retrieval runs at a time. Verified fetched contexts are cached only in memory
for up to ten minutes, bounded to 100 entries and 32 MiB. Restarting the API clears
that cache. No fetched data is written into the CLI database or text cache.

The UI distinguishes:

| State | Meaning |
| --- | --- |
| Verified | Both saved hashes match; the exact context is available. |
| Hash mismatch | Text exists, but it differs from the saved input; the changed text is not displayed as the original. |
| Unavailable | No accessible matching source text is available. |
| Retrieval error | A cache/network operation failed; the user can retry. |

The evidence quote and saved provenance remain accessible in all these cases.
Retrieval accepts only a task already known to the database; users cannot submit
arbitrary source URLs or local paths.

## Architecture and API

The React/Vinext frontend calls a same-origin `/api/` gateway. It forwards only
the supported routes to the configured FastAPI service. SQLite, source files,
and upstream credentials are accessed only by the Python backend.

- `api/repository.py`: parameterized, bounded database queries and short-lived
  read transactions using `mode=ro` plus `query_only=ON`.
- `api/provenance.py`: saved-record assembly and verified prompt templates.
- `api/sources.py` and `api/upstream.py`: cached context verification and explicit
  network retrieval, after database connections close.
- `api/app.py` and `api/contracts.py`: validated HTTP request/response contracts,
  response/error handling, and summary caching. OpenAPI documentation is available
  from the API server at `http://127.0.0.1:8000/docs`.
- `app/`, `components/`, and `lib/api.ts`: routes, user interface, and typed API contracts.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Check that an existing checkpoint can be read. |
| `GET /api/summary?refresh=true` | Global positive/context article counts, cohort breakdown, and run metadata. |
| `GET /api/contexts` | Current positive contexts; accepts `drug`, `gene`, `article`, `cohort`, `coverage`, `page`, and `page_size` (1–100). |
| `GET /api/contexts/{task_id}` | Full positive-entry provenance, including historical entries. |
| `GET /api/contexts/{task_id}/source` | Read/verify existing cached text; never fetches from the network. |
| `POST /api/contexts/{task_id}/source` | Explicitly retrieve and verify source text if no verified cached source exists. |

## Checks and browser tests

```bash
.venv/bin/python -m pytest tests/test_api.py -q
npm run typecheck
npm run build
```

Browser tests use an isolated fixture, not the live corpus. Create it once:

```bash
.venv/bin/python -c "from pathlib import Path; from tests.fixtures import populate; populate(Path('outputs/browser-fixture'))"
```

With the regular web app stopped, launch a fixture-backed instance:

```bash
export DGILIT_DATA_DIR="$PWD/outputs/browser-fixture"
npm run dev
```

In another terminal in `client/`:

```bash
npx playwright install chromium
npm run test:browser
```

Retrieval responses are mocked in browser tests. API tests cover the source
provider and retrieval/hash/failure states. Fixtures, test results, `.venv`,
`node_modules`, builds, and local runtime state are ignored by Git. To return to
live data, stop the fixture app, unset `DGILIT_DATA_DIR`, and restart the app.

## Future production deployment

The frontend and backend are independently deployable through the HTTP contract.
`npm run build` produces the frontend artifact; `npm start` previews the built
Worker locally. The API can run separately with:

```bash
.venv/bin/python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

This version is configured for loopback use. A production deployment would add
its access policy, TLS/reverse proxy configuration, and a suitable persistent
read-only dataset/replica. Point the frontend gateway at the deployed API using
`DGILIT_API_ORIGIN`. Database access is isolated behind `Repository`, so moving
to another database/query service does not require changing the UI contract.
No cloud registration, database migration, or publishing is needed for localhost.
