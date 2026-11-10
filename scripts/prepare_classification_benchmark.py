"""Build a reproducible reviewed seed and an UNLABELED review queue from repo artifacts.

Only human_truth cells are imported as labels; DGIdb truth/predictions are metadata.
No Excel libraries are needed. Run from the repository root.
"""
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def spreadsheet(path):
    with zipfile.ZipFile(path) as z:
        shared = [''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si', NS)] if 'xl/sharedStrings.xml' in z.namelist() else []
        rows = []
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:row', NS):
            cells = {}
            for cell in row.findall('s:c', NS):
                col = cell.attrib['r'].rstrip('0123456789')
                v = cell.find('s:v', NS)
                cells[col] = shared[int(v.text)] if cell.attrib.get('t') == 's' else (v.text if v is not None else ''.join(cell.itertext()))
            rows.append(cells)
        return [{name: row.get(col, '') for col, name in rows[0].items() if name} for row in rows[1:]]


def case(drug, gene, context, pmid, source):
    key = '\0'.join((drug.casefold(), gene.casefold(), context))
    return dict(id=hashlib.sha256(key.encode()).hexdigest()[:20], drug=drug, gene=gene,
                context=context, pmid=str(pmid), source=source, review_status='pending',
                reviewer=None, expected_interaction=None, expected_interaction_type=None,
                expected_directionality=None, category=None)


def main():
    destination = ROOT / 'data/classification_benchmark'
    destination.mkdir(exist_ok=True)
    reviewed = {}
    for path in sorted((ROOT / 'notebooks/test_sets').glob('*/*manual_review*.xlsx')):
        for row in spreadsheet(path):
            if row.get('human_truth') not in {'0', '1'} or not row.get('review_context'):
                continue
            c = case(row['drug'], row['gene'], row['review_context'], row['pmid'], str(path.relative_to(ROOT)))
            c.update(review_status='reviewed', reviewer='mc (existing workbook)',
                     expected_interaction=row['human_truth'] == '1', known_interaction=row.get('truth') == '1',
                     category=row.get('human_truth_reason'), review_notes=row.get('note'),
                     original_interaction_type=row.get('interaction_type'), original_directionality=row.get('directionality'))
            reviewed[c['id']] = c
    queue = {}
    source = ROOT / 'data/2026-06-29-test2-RohitList-wText.xlsx'
    contexts = spreadsheet(source)
    # Known recovery pairs still need context-level review: membership is not a label.
    for row in contexts[:30]:
        c = case(row['drug_name'], row['gene_name'], row['context'], row['pmid'], str(source.relative_to(ROOT)))
        c['known_interaction'] = True
        queue[c['id']] = c
    source = ROOT / 'notebooks/test_sets/test4_curated_dgidb_qwen3_14b/interaction_classifications-rohit-qwen3-14b.csv'
    with source.open() as f:
        predictions = [r for r in csv.DictReader(f) if r['interaction'] == 'true']
    # Reproducible sample, independent of labels. These are suspected review candidates, NOT negatives.
    predictions.sort(key=lambda r: hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest())
    for row in predictions:
        try:
            context_row = contexts[int(row['block_id'])]
        except (IndexError, ValueError):
            continue
        if str(context_row['pmid']) != row['pmid']:
            continue
        c = case(row['drug'], row['gene'], context_row['context'], row['pmid'], str(source.relative_to(ROOT)))
        c.update(previous_prediction=True, previous_evidence=row['evidence'])
        if c['id'] not in reviewed:
            queue[c['id']] = c
        if len(queue) >= 70:
            break
    for name, records in [('reviewed.jsonl', reviewed), ('review_queue.jsonl', queue)]:
        (destination / name).write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in sorted(records.values(), key=lambda r: r['id'])))
        print(f'{name}: {len(records)} cases')


if __name__ == '__main__':
    main()
