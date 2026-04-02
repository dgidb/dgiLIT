# dgiLIT

The Drug-Gene Interaction Literature Integration Tool (dgiLIT) is a pipeline for prioritizing biomedical literature and performing AI-assisted curation of drug–gene interactions for integration into the Drug-Gene Interaction Database (DGIdb).

It combines lightweight NLP heuristics, entity normalization, and large language models (LLMs) to turn unstructured abstracts into structured, reviewable interaction data.

## Overview
The Drug-Gene Interaction Database (DGIdb) has a long history of driving hypothesis generation for biomedical research through the careful curation of drug-gene interaction data from primary and secondary sources with supporting literature. Recent advances in large-language model (LLM) and artificial intelligence (AI) technologies are enabling new paradigms for knowledge extraction and biocuration. The accelerating growth of biomedical literature presents a significant challenge for maintaining up-to-date interaction data. With more than 38 million citations indexed in PubMed alone, new strategies must evolve to identify and incorporate new interaction data into DGIdb. 

To address this need, we present the drug-gene interaction literature integration tool (dgiLIT) as an LLM-assistive toolkit to curate new drug-gene interactions into DGIdb. dgiLIT utilizes a combined approach of basic natural language processing techniques, existing harmonization technologies, and AI-assisted curation to retrieve, prioritize, and curate drug-gene interactions from published literature.

## Examples
An example for search methods, lemma indicator score generation, and full curation search with agentic curation are available in demo notebooks: `example-1.ipynb`, `example-2.ipynb`, and `example-3.ipynb` within the `research/notebooks` directory.

## Supplementary Tables
Supplementary tables generated as part of our BCL2 stress test are available for access within the `research/paper` directory. Available data are:

- **Supplemental Table 1**. Ontology entries for lemmatization groupings in dgiLIT
- **Supplemental Table 2**. Pre-processing Dataframe for AI Curation Task			
- **Supplemental Table 3**. AI Curated Interactions from BCL2 associated literature in PubMed
- **Supplemental Table 4**. Human Evaluation of AI-Curated Drug-Gene Interactions