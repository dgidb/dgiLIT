"""Regenerate the canonical JSON from the supplied spreadsheet, without Excel libraries."""
import json
from pathlib import Path

from prepare_classification_benchmark import spreadsheet

ROOT = Path(__file__).resolve().parents[1]


def main():
    ref = ROOT / 'src/dgilit/ref'
    types = {}
    for number, row in enumerate(spreadsheet(ref / 'interaction_types.xlsx'), 2):
        name = row['interaction type']
        description = row['description']
        direction = {'ACTIVATING': 'activating', 'INHIBITORY': 'inhibiting', '': 'unclear'}[row['directionality']]
        entry = types.setdefault(name, dict(name=name, description=description,
            directionality=direction, references=[], source_rows=[], alternate_descriptions=[]))
        if entry['directionality'] != direction:
            raise ValueError(f'Conflicting duplicate directionality: {name}')
        if description != entry['description'] and description not in entry['alternate_descriptions']:
            entry['alternate_descriptions'].append(description)
        if row['reference'] and row['reference'] not in entry['references']:
            entry['references'].append(row['reference'])
        entry['source_rows'].append(number)
    output = dict(source='interaction_types.xlsx, sheet1',
        normalization='ACTIVATING → activating; INHIBITORY → inhibiting; blank → unclear. Duplicate names merged, preserving descriptions, references and source rows.',
        types=sorted(types.values(), key=lambda entry: entry['name']))
    (ref / 'interaction_vocabulary.json').write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
