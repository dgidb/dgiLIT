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
