# dgiLIT

Pre-print: https://www.biorxiv.org/content/10.64898/2026.01.16.699733v1

dgiLIT (Drug-Gene Interaction Literature Integration Tool) is a Python package for prioritizing biomedical literature and performing AI-assisted curation of drug–gene interactions for integration into the Drug-Gene Interaction Database (DGIdb).

The package combines lightweight natural language processing, named-entity recognition, biomedical concept normalization, and large language models (LLMs) to transform unstructured biomedical literature into structured, reviewable interaction data.

## Overview
The Drug-Gene Interaction Database (DGIdb) has a long history of curating drug–gene interactions from the biomedical literature. As the volume of published research continues to grow, maintaining comprehensive and up-to-date interaction data has become increasingly challenging.

dgiLIT addresses this problem by providing a modular pipeline that:

- identifies candidate drug and gene entities,
- normalizes entities to biomedical ontologies,
- extracts interactions using large language models, and
- produces structured outputs suitable for expert human review and downstream integration into DGIdb.

The workflow is designed to reduce the amount of literature requiring manual inspection while maintaining human oversight of the final curated interactions.

See the [package-to-client workflow diagrams](docs/workflow.md) for component connections, inputs and outputs, checkpoint storage, and how Claim Explorer loads and displays results.

## Installation

Install the latest release from PyPI:

```bash
pip install dgilit
```

Or install the development version from source:

```bash
git clone https://github.com/dgidb/dgiLIT.git
cd dgiLIT
pip install -e .
```

For entity pre-tagging with PubTator3, download the required PubTator3 annotation files and configure the path in your environment or notebook.

---

## Example

### PubMed article retrieval

Use the convenience function when you want notebook-friendly text keyed by PMID:

```python
from dgilit import fetch_pubmed_abstracts

abstract_map = fetch_pubmed_abstracts(
    pmid_df["pmid"],
    email=MC_EMAIL,
)

pmid_df["context"] = pmid_df["pmid"].astype(str).map(abstract_map)
```

Use the structured client when you want validated Pydantic article models with
sections preserved for downstream tagging. PubMed records provide citation
metadata and abstracts; when a linked PMC record is available, dgiLIT also
attempts to enrich the article with full-text sections such as Results,
Discussion, and Conclusions.

```python
from dgilit import PubMedClient

client = PubMedClient(
    email=MC_EMAIL,
    batch_size=100,
)

articles = client.fetch_articles(["123", "456"], include_full_text=True)

article = articles["123"]

print(article.pmid)
print(article.pmcid)
print(article.title)
print(article.full_text_available)

for section in article.sections:
    print(section.section_type, section.label, section.text)
    # Run downstream tagging on section.text independently.

results_and_conclusions = [
    section
    for section in article.sections
    if section.section_type in {"results", "conclusions"}
]
```

### Entity pre-tagging

The example below demonstrates the entity pre-tagging workflow, which identifies and normalizes candidate drug and gene mentions from biomedical text.

```python
from dgilit import (
    BioBertEntityTagger,
    CompositeEntityTagger,
    EntityPreTaggingService,
    PubTator3ChemicalTagger,
    SQLiteNormalizerCache,
    TaggerConfig,
    ViccNormalizer,
)

PUBTATOR_PATH = "/path/to/chemical2pubtator3"

# Configure entity taggers
tagger = CompositeEntityTagger(
    BioBertEntityTagger(
        TaggerConfig(
            batch_size=16,
            include_drugs=True,
            include_genes=True,
            include_diseases=False,
            device=-1,
        )
    ),
    PubTator3ChemicalTagger.from_pubtator_file(PUBTATOR_PATH),
)

# Configure concept normalization
pretagger = EntityPreTaggingService(
    tagger=tagger,
    normalizer=ViccNormalizer(
        cache=SQLiteNormalizerCache(".dgilit_normalizer_cache.sqlite")
    ),
)

text = """
Venetoclax inhibits BCL2 and induces apoptosis in leukemia cells.
"""

# Identify and normalize entities
result = pretagger.tag_block(text)

# Inspect grounded entities
for entity in result.entities:
    print(entity)

# LLM Curation Task of Choice
```

For complete examples including literature retrieval, AI-assisted interaction classification, and evaluation workflows, see the notebooks in the `notebooks/` directory.

### Qwen3-14B interaction assessment

Install the optional local inference dependencies:

```bash
pip install -e '.[qwen]'
```

`Qwen3JsonClient` loads `mlx-community/Qwen3-14B-4bit` once per instance with
MLX-LM on Apple Silicon. Use native arm64 Python on macOS with Metal available;
the client fails explicitly instead of falling back to CPU. Install the extra
in the environment used by your notebook kernel. Restart that kernel before
switching from the old client to release previously loaded PyTorch weights.

The extra uses MLX-LM 0.32.x and MLX 0.32.x. Transformers may still be used for
tokenization, but model inference runs through MLX/Metal. Initial loading downloads
the quantized weights to the Hugging Face cache. Leave unified-memory headroom
for context/KV cache, tagging models, and other applications; memory pressure can
still reduce throughput.

```python
from dgilit import (
    InteractionClassificationPrompt,
    InteractionClassificationResult,
    Qwen3JsonClient,
)
from wags_llm.prompts import build_empty_registry
from wags_llm.services import StructuredTaskRunner

prompt = InteractionClassificationPrompt(version="v2")
registry = build_empty_registry()
registry.register(prompt)
client = Qwen3JsonClient(
    model_id="mlx-community/Qwen3-14B-4bit",
    max_tokens=1000,
    temperature=0.0,
)
runner = StructuredTaskRunner(client=client, prompt_registry=registry)
result = runner.execute(
    prompt_name=prompt.name,
    prompt_version=prompt.version,
    payload=prompt.build_payload(
        context="Venetoclax inhibits BCL2.",
        candidate_drug="Venetoclax",
        candidate_gene="BCL2",
    ),
    response_model=InteractionClassificationResult,
)
```

The client disables thinking, uses greedy decoding at temperature zero, and
passes the response schema as prompt instructions. JSON parsing and WAGS's
Pydantic validation detect invalid responses; decoding is not schema-constrained.
`max_tokens` limits newly generated tokens. Inputs exceeding the model's context
window plus this budget fail explicitly rather than being truncated.

The notebook at
`notebooks/test_sets/test4_curated_dgidb_qwen3_14b/01_wags_pretagging.ipynb`
reproduces test2's assessment using its existing text workbook, with Qwen-specific
CSV and Parquet outputs. Its setup cell lists the additional notebook dependencies
and its one-pair smoke check precedes the full experiment. The existing test2
Claude/Bedrock workflow remains available.

Model loading time is exposed as `client.load_seconds`. After each inference,
`client.last_generation_stats` reports prompt/generated tokens, elapsed seconds,
generation tokens per second, and end-to-end tokens per second. Test4 prints
these for a synthetic smoke check and a warmed real candidate pair before the
full run. These checks validate the response schema and measure actual speed;
no fixed latency is guaranteed. CUDA `device_map` and `dtype` arguments from the
previous Qwen implementation are no longer accepted.

### Resumable ten-year PMC extraction

See the [workflow and CLI guide](notebooks/pmc_last10years/README.md) for setup,
command examples, status-field explanations, stopping/resuming, and exported data.

The corpus CLI reuses the section retrieval, BioBERT/VICC normalization, and Qwen
classification used by the test notebooks. The read-only analysis notebook at
`notebooks/pmc_last10years/01_pmc_interactions.ipynb` inspects the CLI's current
pilot/full-cohort checkpoints without generating data. The CLI discovers **all PMC records published from
2016-11-03 through 2026-11-03**, including records without a PMID. It uses every
parsed abstract/body section, with abstract fallback when full text cannot be
retrieved. PDFs and supplementary attachments are not processed. PMC content
availability varies; see [PMC's supported retrieval services](https://pmc.ncbi.nlm.nih.gov/tools/developers/).

Use native arm64 Python on Apple Silicon with Metal available:

```bash
pip install -e '.[pmc,qwen]'
# Set MC_EMAIL or NCBI_EMAIL for NCBI contact information; NCBI_API_KEY is optional.
python -m dgilit.pmc_cli discover
python -m dgilit.pmc_cli run --pilot-only
python -m dgilit.pmc_cli status
python -m dgilit.pmc_cli run
python -m dgilit.pmc_cli retry-failed
python -m dgilit.pmc_cli run
python -m dgilit.pmc_cli export
```

`dgilit-pmc` is an equivalent installed entry point. Every command accepts
`--output PATH`. A new run can set both `--start YYYY-MM-DD` and `--end YYYY-MM-DD`;
subsequent commands reuse its stored configuration. The default output is
`data/pmc_last10years_2016-11-03_2026-11-03/` in this checkout. Generated corpus
data are ignored by Git. Set a persistent output path explicitly for an installed
package outside this checkout.

`run` discovers any missing partitions, executes an initial 100-article
publication-year-stratified pilot with at most 100 Qwen tasks, and continues the
entire cohort. It has no default corpus cap. `--pilot-only` stops after the pilot;
`--max-calls N` explicitly bounds calls in the subsequent full phase. The pilot
report estimates candidate volume, inference time, and state storage; these are
extrapolations from a small sample, not a promised completion date. The saved
test4 benchmark was approximately 12 seconds per pair, so a millions-of-articles
run may be impractical on one Mac. Corpus-level processing remains resumable
across interruptions. For unattended use, redirect CLI output to a log and run
under your usual terminal/session manager; keep the Mac awake as appropriate.

SQLite stores transactional checkpoints and all successful classifications.
An exclusive writer lock prevents concurrent `run`/`discover` writers.
`status` reads current state; `status.json` is a durable atomic snapshot refreshed
about every 30 seconds during active batches and at run boundaries. `all_articles_terminal` includes articles with recorded
failures, while `successful_completion` additionally requires no unresolved
failure events. Inspect coverage and failure counts before calling an extraction
complete. `retry-failed` requeues failed work while preserving successful tasks.
Changed configurations, model revisions, prompt contents, or inference package
versions require a separate output directory. `run --detach` launches an
independent process and writes progress to `run.log` in the output directory.
`export` reads a consistent SQLite snapshot and can run while extraction proceeds.

Text retrieval uses supported NCBI APIs, a shared limiter below three requests
per second, five transient retries, and `Retry-After` handling. Downloaded parsed
text uses a 10-GiB cache; completed articles can be evicted. If free space falls
below 10 GiB, processing checkpoints and exits without reducing the cohort.
Results continue to grow separately from that cache; larger storage may still be
needed. Evidence, hashes, entity offsets, and provenance persist after text
eviction, but reproducing a complete evicted context requires refetching it.

NER uses each model's token limit with 64-token overlap and section-relative
offsets. Only normalized drug/gene concepts enter the section-level Cartesian
product; raw mentions and unsuccessful normalization outcomes remain auditable.
No gene panel or preassigned gene labels are used. Qwen uses prompt `v2`,
temperature zero, thinking disabled, and 1,000 generated tokens. Context windows
reserve prompt/schema space and use 256-token overlap; every section candidate
pair is assessed against every context of an oversized section. No text is
silently truncated. The pipeline does not infer across separate sections.

Any positive context supports an article-level positive pair. A negative pair
requires every required context to finish negatively; failures remain unresolved.
Positive evidence must be a direct quote verifiable in its context. These are
**model-predicted interactions**, intended for expert review. There is no labeled
corpus truth set, so these outputs do not establish precision or recall.

`export` writes Zstandard-compressed Parquet shards in a new snapshot directory
and updates `exports/latest.json` only after every dataset and its manifest are
complete. Datasets include articles, partitions, sections, entities,
classifications (including negatives), article pairs, positive interactions, and
failures. Positive pairs can have unresolved additional contexts; inspect
`unresolved_contexts` and `article_status`. Evidence and classification details
are linked through task IDs; JSON payload columns retain the complete records.
Exports are snapshots: run `export` again for newer results, and retain/remove
old snapshots according to available storage.
