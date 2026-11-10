# Test5: DGIdb Qwen3-14B with prompt v3

This directory mirrors test4 using the current **inclusive v3** prompt and packaged
DGIdb reference knowledge. Earlier test inputs, notebooks and results are unchanged.

| Notebook | Purpose | Execution status |
|---|---|---|
| `01_wags_pretagging.ipynb` | Full curated-set tagging, Qwen v3 classification, CSV/Parquet export and prompt provenance | Prepared; full inference not run |
| `02_analysis.ipynb` | Precision/recall/F1, confusion matrices, full truth recovery and review workbooks | Prepared; requires outputs from 01 |
| `03_concordance.ipynb` | Full test4 v2 versus test5 v3 metrics, candidate agreement and recovery differences | Prepared; requires outputs from 01 |
| `04_reviewed_seed_analysis.ipynb` | Recompute the completed 42-case human-reviewed comparison | Executed; outputs saved |

Use a fresh kernel and run from this directory. From the repository root, install
`pip install -e '.[qwen,pmc]'` plus pandas, python-dotenv, openpyxl, fastexcel,
ipywidgets, matplotlib, nbformat, nbclient and xlsxwriter as needed. Set the same
`PUBTATOR_PATH` used by test2/test4. Run 01, then 02 and 03 for the complete curated
100-interaction experiment. Qwen inference requires Apple Silicon and Metal.

The full workflow preserves test4's candidate policy: normalized drug candidates,
BioBERT plus PubTator chemical tagging, and curated PMID-to-gene labels. It reruns
pre-tagging; use the concordance audit to inspect any candidate differences. It uses
Qwen3-14B 4-bit, temperature zero, thinking disabled, max_tokens=1000 and prompt v3.
The v3 result schema and shared source/identity validation are both applied. Failed
classifications retain their candidate identities and have null decisions, separate
from valid negatives. Results carry the assembled prompt hash, with the complete prompt
and run metadata saved here on execution.

`02` reports valid-candidate metrics and failed/skip counts separately. End-to-end
recovery includes all curated truth pairs, so missing candidates and failed
classifications remain unrecovered. Non-membership in the curated DGIdb set is a
proxy negative for comparison with test2/test4, not proof of an incorrect literature
claim. Human-corrected scoring requires a separately annotated test5 review workbook;
no earlier human annotations are silently reused.

Full-run filenames use `interaction_classifications-rohit-qwen3-14b-v3` for CSV/Parquet
and `qwen3_14b_v3_*` for review workbooks. Analysis also exports `v3_candidate_metrics.csv`
and `v3_recovery_metrics.csv`. These full-run artifacts do not exist until 01/02 run.

## Existing reviewed-seed numbers

The separately executed `04` notebook reads the archived experiments in
`data/classification_benchmark`. It makes no model or network calls. Its numbers cover
**42 reviewed BCL2 cases (38 positives, 4 negatives)**, not the curated 100-interaction
truth workbook:

| Run | Precision | Recall | F1 | Invalid outputs |
|---|---:|---:|---:|---:|
| v2 baseline | 89.74% | 92.11% | 90.91% | 1 |
| Original strict v3 | 91.30% | 55.26% | 68.85% | 3 |
| Revised inclusive v3 | 89.19% | 86.84% | 88.00% | 3 |

Precision excludes invalid outputs; recall counts invalid gold positives as unrecovered.
These are historical human labels on a small error-enriched seed, not held-out corpus
performance. The revised v3 fails the promotion gate and v2 remains the package default.
The executed notebook includes the raw-decision agreement, quote failures and pending
review cases. Its standalone outputs are `reviewed_seed_metrics.csv` and
`reviewed_seed_precision_recall.png`.

All notebooks were checked for notebook validity and Python syntax. Classification
failure/export handling and the complete 02/03 analysis paths were exercised on an
isolated synthetic fixture. Fixture-derived scores and workbooks were not saved here
or presented as v3 model results.
