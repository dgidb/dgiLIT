"""Methods for matching lemmas to defined lemma groups. Current groups include clinical and interaction"""
import ast
import csv
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm

logger = logging.getLogger(__name__)


nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def _normalize(text: str) -> list[str]:
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

def parse_drug_terms(drug_entry: str) -> list[str]:
    """Return all names from a ``(brand, generic)`` tuple string."""
    try:
        value = ast.literal_eval(drug_entry)
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value if isinstance(v, str)]
    except (ValueError, SyntaxError) as e:
        logger.debug("Failed to parse drug_entry %s: %s", drug_entry, e)
    return [drug_entry]


# Words Sets & Intended Weights TODO: Lemma-tize
CLINICAL_STUDY = {'randomized', 'randomised', 'placebo', 'double-blind', 'single-blind', 'controlled', 'trial', 'pragmatic trial', 'phase i', 'phase ii', 'phase iii', 'phase iv', 'clinical study', 'clinical trial', 'multicenter', 'interventional', 'crossover', 'parallel group', 'allocation', 'intention-to-treat', 'efficacy', 'endpoint', 'primary outcome', 'progression free survival', 'overall survival'} # Weight = 1

CASE_REPORT = {'case report', 'case reports', 'case series', 'case study', 'case studies', 'single patient', 'n-of-1'} # Weight = 1

ANIMAL_EVIDENCE = {'in vivo', 'xenograft', 'patient-derived xenograft', 'PDX', 'orthotopic', 'murine', 'mouse', 'mice', 'rat', 'rats', 'zebrafish', 'Drosophila', 'C. elegans', 'animal model', 'live animal'} # Weight = 0.5

CELL_LINES = {'in vitro', 'cell line', 'cultured cells', 'cell culture', 'monolayer', '3D culture', 'organoid', 'primary cells', 'immortalized cell line', 'transfected cell line'} # Weight = 0.5

IMAGING_EVIDENCE = {'MRI', 'magnetic resonance imaging', 'CT', 'computed tomography', 'PET', 'positron emission tomography', 'ultrasound', 'radiography', 'fluorescence microscopy', 'confocal microscopy', 'histopathology', 'histological imaging', 'digital image analysis', 'quantitative imaging', 'imaging biomarkers'} # Weight = 0.5

RETROSPECTIVE_STUDY = {'retrospective study', 'retrospective analysis', 'retrospective review', 'chart review', 'registry data', 'historical cohort', 'medical record review', 'retrospective cohort'} # Weight = 0.75

CHEMOTHERAPY_AGENTS = {
    # Alkylating agents
    "Cyclophosphamide", "Carmustine", "Cisplatin", "Carboplatin", "Oxaliplatin",

    # Anthracyclines / related
    "Doxorubicin", "Daunorubicin", "Daunomycin", "Idarubicin",

    # Antimetabolites (nucleoside analogues)
    "Fluorouracil", "Methotrexate", "Azacitidine",

    # Taxanes & microtubule disruptors
    "Paclitaxel", "Docetaxel", "Vincristine", "Vinblastine", "Vinorelbine",

    # Topoisomerase inhibitors
    "Etoposide", "Topotecan", "Irinotecan",
} # Weight = 0, Should be hidable


# ---------- LEMMA-LEVEL LEXICONS (single-token lemmas) ----------
DIRECT_INTERACTION = {
    "inhibit", "suppress", "block", "antagonize", "antagonist", "activate", "agonist", "stimulate", "abrogate", "attenuate", "neutralize", "potentiate", "enhance", "impair", "diminish", "abolish"
}

BINDING_INTERACTION = {
    "bind", "target", "interact", "affinity", "associate", "association", "complex", "recruit", "sequester", "dock", "occupy", "ligand"
}

REGULATION_CHANGES = {
    "upregulate", "downregulate", "overexpress", "silence", "knockdown", "knockout", "crispr", "knockin", "dysregulate", "downmodulate", "repress", "repression","transactivate", "transactivation", "deplete", "depletion","induce", "induction"
}

SENSITIVITY_RESISTANCE = {
    "sensitivity", "sensitive", "sensitize", "resistant", "resistance", "resensitize", "synergy", "synergistic", "tolerant", "tolerance", "refractory", "respond", "responder", "nonresponder", "response", "hypersensitive"
}

PHARMACOGENOMIC_SIGNALS = {
    "metabolize", "substrate", "induce", "inducer", "inhibit", "polymorphism", "variant", "mutation", "mutant", "allele", "genotype", "haplotype", "isoform", "splice", "wildtype", "germline", "somatic", "clearance", "exposure", "auc", "cmax", "tmax"
}


def analyze_relation_interaction(gene: str, abstract: str) -> tuple[str, float]: # noqa: ARG001 (TODO: fix logic in notebooks before breaking change?)
    """Identify lemma matches using interaction lemma groups"""
    text = abstract.lower()
    lemmas = _normalize(text)
    lemma_set = set(lemmas)

    indicators = {}
    indicators['direct_interaction'] = sum(lemma in lemma_set for lemma in DIRECT_INTERACTION)
    indicators['binding_interaction'] = sum(lemma in lemma_set for lemma in BINDING_INTERACTION)
    indicators['regulation_changes'] = sum(lemma in lemma_set for lemma in REGULATION_CHANGES)
    indicators['sensitivity_resistance'] = sum(lemma in lemma_set for lemma in SENSITIVITY_RESISTANCE)
    indicators['pharmacogenomic_signals'] = sum(lemma in lemma_set for lemma in PHARMACOGENOMIC_SIGNALS)

    indicators['unweighted_total'] = sum(indicators.values())

    label = 'interaction_evidence' if indicators['unweighted_total'] > 0 else 'no_interaction_evidence'

    return label, indicators

def analyze_relation(drug: str, gene: str, abstract: str) -> tuple[str, float]:
    """Identify lemma matches using clinical lemma groups"""
    text = abstract.lower()
    drug_present = re.search(re.escape(drug.lower()), text) is not None
    gene_present = re.search(re.escape(gene.lower()), text) is not None

    if not (drug_present and gene_present):
        return ('not_evaluated', 0.0)

    indicators = {}

    indicators['clinical_study'] = sum(word in text for word in CLINICAL_STUDY)
    indicators['case_report'] = sum(word in text for word in CASE_REPORT)
    indicators['animal_evidence'] = sum(word in text for word in ANIMAL_EVIDENCE)
    indicators['cell_line'] = sum(word in text for word in CELL_LINES)
    indicators['imaging_evidence'] = sum(word in text for word in IMAGING_EVIDENCE)
    indicators['retrospective'] = sum(word in text for word in RETROSPECTIVE_STUDY)
    indicators['unweighted_total'] = sum(indicators.values())

    if indicators['unweighted_total'] > 0:
        label = 'indicator_evidence'
    else:
        label = 'no_indicator_evidence'

    return label, indicators


def generate_interaction_evidence(abstracts: pd.DataFrame , reference_df: pd.DataFrame, start: int = 0, stop: int = -1) -> None:
    """Generate drug-gene interaction indicators from abstracts and save results to disk."""
    gene = reference_df['Gene'][0] # TODO: can remove??? just using for filename

    timestamp = datetime.now(tz='utc').strftime("%Y-%m-%d")
    timestamp_folder = f'{timestamp}_{gene}'

    results = []
    for _, row in tqdm(abstracts.iloc[start:stop].iterrows(), desc="Abstracts", leave=False):
        pmid = row['pmid'] if 'pmid' in row else row.iloc[0]
        abstract = row['abstract'] if 'abstract' in row else row.iloc[1]
        tagged_drugs = row['DRUG_LABELS']
        concept = row['DRUG_IDS']

        label,scores = analyze_relation_interaction(gene, abstract)

        results.append({"pmid": pmid, 'abstract': abstract, "label": label, "scores": scores, 'tagged_drugs': tagged_drugs, 'concepts': concept  })

    Path(timestamp_folder).mkdir(parents=True, exist_ok=True)
    out_filename = f'{gene}_interaction_search.csv'.replace("/", "-")
    out_path = Path(timestamp_folder) / out_filename
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pmid",'abstract', "label", "scores", 'tagged_drugs', 'concepts'])
        writer.writeheader()
        writer.writerows(results)

    shutil.rmtree(timestamp_folder)
    print(f'Results saved to {timestamp_folder}.zip!') # noqa: T201



def generate_indicators(abstracts: pd.DataFrame, reference_df: pd.DataFrame, start: int =0, stop: int =-1, mode: str ='clinical') -> None:
    """Generate drug-gene interaction indicators from abstracts and save results to disk."""
    gene = reference_df['Gene'][0]

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    timestamp_folder = f'{timestamp}_{gene}'

    for _idx, row in tqdm(reference_df.iterrows()):
        gene = str(row.get('Gene'))
        drug = str(row.get('Drug'))
        drug = parse_drug_terms(drug)

        results = []
        for _, row in tqdm(abstracts.iloc[start:stop].iterrows(), desc="Abstracts", leave=False): # noqa: PLW2901
            pmid = row['pmid'] if 'pmid' in row else row.iloc[0]
            abstract = row['abstract'] if 'abstract' in row else row.iloc[1]

            # Handle Extra Columns for NLP Processing
            try:
                tagged_drugs = row['DRUG_LABELS']
                concept = row['DRUG_IDS']
            except KeyError:
                tagged_drugs = None
                concept = None
            if mode == 'interaction':
                label,scores = analyze_relation_interaction(drug[0], gene, abstract)
            else:
                label, scores = analyze_relation(drug[0], gene, abstract)
            if label:
                results.append({"pmid": pmid, "label": label, "scores": scores, 'tagged_drugs': tagged_drugs, 'concepts': concept  })
            else:
                results.append({"pmid": pmid, "label": label, "scores": scores })

            if results:
                Path(timestamp_folder).mkdir(parents=True, exist_ok=True)
                out_filename = f'{gene}_{drug[0]}.csv'.replace("/", "-")
                out_path = Path(timestamp_folder) / out_filename
                with out_path.open("w", newline="") as fh:
                    if label:
                        writer = csv.DictWriter(fh, fieldnames=["pmid", "label", "scores", 'tagged_drugs', 'concepts'])
                    else:
                        writer = csv.DictWriter(fh, fieldnames=["pmid", "label", "scores"])
                    writer.writeheader()
                    writer.writerows(results)

    shutil.rmtree(timestamp_folder)
    print(f'Results saved to {timestamp_folder}.zip!') # noqa: T201

