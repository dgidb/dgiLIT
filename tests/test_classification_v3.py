from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dgilit import InteractionClassificationPrompt, classification_result_model, validate_classification
from dgilit.evaluation import load_v2_baseline, reviewed_cases, run_comparison, summarize
from dgilit.pmc_pipeline import PMCPipeline, PipelineConfig
from dgilit.references import REF_DIR, assemble_system_prompt, load_vocabulary


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.positive = dict(drug='Drug', gene='GENE', interaction=True,
                             interaction_type='inhibitor', directionality='inhibiting',
                             evidence='Drug inhibits GENE.', error_message=None)
        self.payload = dict(version='v3', context='Results: Drug inhibits GENE.',
                            candidate_drug='Drug', candidate_gene='GENE')

    def test_contract_and_legacy(self):
        self.assertTrue(validate_classification(self.positive, **self.payload).interaction)
        for change in [dict(drug='Other'), dict(gene='OTHER'), dict(interaction='true'),
                       dict(interaction_type='inhibition'), dict(directionality='activating'),
                       dict(evidence='Drug binds GENE.'), dict(evidence=None),
                       dict(interaction=False), dict(error_message='generation failed'),
                       dict(interaction_type=None), dict(directionality=None)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_classification(self.positive | change, **self.payload)
        negative = self.positive | dict(interaction=False, evidence=None,
                                      interaction_type=None, directionality=None)
        self.assertFalse(validate_classification(negative, **self.payload).interaction)
        unknown = self.positive | dict(interaction_type='other/unknown', directionality='unclear')
        self.assertTrue(validate_classification(unknown, **self.payload).interaction)
        response = unknown | dict(evidence='GENE overexpression conferred resistance to Drug.')
        self.assertTrue(validate_classification(response, **(self.payload | dict(context=response['evidence']))).interaction)
        # Existing v2 free-text labels remain accepted.
        self.assertEqual(classification_result_model('v2').model_validate(
            self.positive | dict(interaction_type='inhibition')).interaction_type, 'inhibition')
        labels = classification_result_model('v3').model_json_schema()['properties']['interaction_type']['anyOf'][0]['enum']
        self.assertEqual(set(labels), set(load_vocabulary()))

    def test_references_and_deterministic_assembly(self):
        prompt = InteractionClassificationPrompt('v3')
        original = prompt.build_system_prompt()
        self.assertEqual(original, prompt.build_system_prompt())
        self.assertEqual(len(load_vocabulary()), 30)
        self.assertEqual(load_vocabulary()['antibody']['source_rows'], [10, 19])
        for name in ('data_model.md', 'druggable_genome.md', 'interaction_types.md'):
            self.assertIn((REF_DIR / name).read_text().strip(), original)
        with tempfile.TemporaryDirectory() as temp:
            refs = Path(temp) / 'ref'
            shutil.copytree(REF_DIR, refs)
            with (refs / 'data_model.md').open('a') as f:
                f.write('\nAdditional definition.')
            self.assertNotEqual(original, assemble_system_prompt('v3', prompt.PROMPT_DIR, refs))
            (refs / 'interaction_types.md').unlink()
            with self.assertRaises(FileNotFoundError):
                assemble_system_prompt('v3', prompt.PROMPT_DIR, refs)
            self.assertEqual(assemble_system_prompt('v2', prompt.PROMPT_DIR, refs),
                             (prompt.PROMPT_DIR / 'v2.text').read_text())
            vocabulary = json.loads((refs / 'interaction_vocabulary.json').read_text())
            vocabulary['types'].append(vocabulary['types'][0])
            (refs / 'interaction_vocabulary.json').write_text(json.dumps(vocabulary))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_vocabulary(refs)

    def test_reference_change_invalidates_execution_fingerprint(self):
        from test_pmc_pipeline import FakeClient, FakeTagger, FakeLLM
        with tempfile.TemporaryDirectory() as temp:
            p = PMCPipeline(Path(temp) / 'run', PipelineConfig(prompt_version='v3'),
                            client=FakeClient({}), pretagger=FakeTagger(), llm=FakeLLM())
            with patch('dgilit.pmc_pipeline._model_revision', return_value='test'):
                p._ensure_models()
                del p.execution_hash
                system = p.prompt.build_system_prompt()
                with patch.object(p.prompt, 'build_system_prompt', return_value=system + '\nchanged reference'):
                    with self.assertRaisesRegex(ValueError, 'new output directory'):
                        p._ensure_models()
            p.store.db.close()

    def test_pipeline_rejects_malformed_and_accepts_valid_v3(self):
        from test_pmc_pipeline import FakeClient, FakeTagger, article
        class LLM:
            last_generation_stats = {}
            context_windows = staticmethod(lambda *a, **kw: [(0, len(a[1]))])
            def invoke_json(inner, *args):
                return SimpleNamespace(parsed_json=self.positive | {'drug': 'wrong'})
        with tempfile.TemporaryDirectory() as temp:
            p = PMCPipeline(temp, PipelineConfig(start='2020-01-01', end='2020-12-31', prompt_version='v3'),
                            client=FakeClient({'PMC1': article()}), pretagger=FakeTagger(), llm=LLM(),
                            min_free_bytes=0, cache_bytes=100000)
            with patch('dgilit.pmc_pipeline._model_revision', return_value='test'):
                p.run(pilot_only=True)
            task = p.store.db.execute('SELECT * FROM tasks').fetchone()
            self.assertEqual(task['status'], 'failed')
            p.llm.invoke_json = lambda *a: SimpleNamespace(parsed_json=self.positive)
            p._classify(task, article())
            task = p.store.db.execute('SELECT * FROM tasks').fetchone()
            self.assertEqual(task['status'], 'done')
            self.assertTrue(json.loads(task['result'])['evidence_in_context'])
            p.store.db.close()

    def test_cli_selects_v3_and_keeps_saved_runs(self):
        from dgilit.pmc_cli import main
        with tempfile.TemporaryDirectory() as temp:
            main(['status', '--output', temp, '--prompt-version', 'v3'])
            p = PMCPipeline(temp)
            self.assertEqual(p.config.prompt_version, 'v3')
            p.store.db.close()
            with self.assertRaises(ValueError):
                main(['status', '--output', temp, '--prompt-version', 'v2'])

    def test_v3_reference_overhead_is_included_in_context_budget(self):
        from dgilit.qwen import Qwen3JsonClient
        from test_pmc_pipeline import CharacterTokenizer
        client = object.__new__(Qwen3JsonClient)
        client._context_window, client.max_tokens = 25000, 1000
        client._tokenizer = CharacterTokenizer()
        client.prompt_tokens = lambda system, user, schema: len(system) + len(user) + len(json.dumps(schema))
        prompt = InteractionClassificationPrompt('v3')
        schema = classification_result_model('v3').model_json_schema()
        text = 'Long article context. ' * 1500
        windows = client.context_windows(prompt, text, 'Drug', 'GENE', schema)
        self.assertGreater(len(windows), 1)
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][1], len(text))
        for start, end in windows:
            payload = prompt.build_payload(text[start:end], 'Drug', 'GENE')
            self.assertLessEqual(client.prompt_tokens(prompt.build_system_prompt(), prompt.build_user_prompt(payload), schema)
                                 + client.max_tokens, client._context_window)

    def test_paired_evaluation_does_not_label_pending_cases(self):
        rows = [dict(id='positive', drug='Drug', gene='GENE', context='Drug inhibits GENE.',
                     review_status='reviewed', reviewer='curator', expected_interaction=True,
                     expected_interaction_type='inhibitor', expected_directionality='inhibiting'),
                dict(id='negative', drug='Drug', gene='GENE', context='Drug and GENE were mentioned.',
                     review_status='reviewed', reviewer='curator', expected_interaction=False,
                     category='co-mention'), dict(id='pending', review_status='pending')]
        outputs = iter([self.positive, self.positive | {'evidence': 'Drug and GENE were mentioned.'},
                        self.positive, self.positive | dict(interaction=False, evidence=None,
                                                          interaction_type=None, directionality=None)])
        class Client:
            last_generation_stats = {'prompt_tokens': 100, 'generation_tokens': 20}
            def invoke_json(self, *args):
                return SimpleNamespace(parsed_json=next(outputs))
        records = run_comparison(rows, Client())
        self.assertEqual(len(records), 4)
        report = summarize(rows, records)
        self.assertEqual(report['v2']['precision'], .5)
        self.assertEqual(report['v3']['precision'], 1)
        self.assertEqual(report['v3']['interaction_type_accuracy'], 1)
        self.assertTrue(report['quantitative_gate_passed'])
        self.assertEqual(report['newly_missed_positive_ids'], [])
        records[-2]['valid'] = False
        report = summarize(rows, records)
        self.assertEqual(report['v3']['recall'], 0)
        self.assertEqual(report['v3']['interaction_type_accuracy'], 0)
        self.assertEqual(report['newly_missed_positive_ids'], ['positive'])
        self.assertFalse(report['quantitative_gate_passed'])
        with self.assertRaises(ValueError):
            reviewed_cases([dict(id='unlabeled', review_status='reviewed')])

    def test_baseline_reuse_requires_same_model_context_prompt_and_settings(self):
        from dgilit.pmc_store import digest
        case = dict(id='case', drug='Drug', gene='GENE', context='Drug inhibits GENE.',
                    review_status='reviewed', reviewer='curator', expected_interaction=True)
        metadata = dict(model_id='model', model_revision='revision', max_tokens=1000,
                        temperature=0, enable_thinking=False, benchmark_sha256='benchmark', packages={'mlx': 'version'})
        record = dict(id='case', version='v2', case_hash=digest(case),
                      prompt_hash=digest(InteractionClassificationPrompt('v2').build_system_prompt()),
                      valid=True, result=self.positive)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'metadata.json').write_text(json.dumps(metadata))
            (root / 'results.jsonl').write_text(json.dumps(record) + '\n')
            self.assertEqual(load_v2_baseline(root, [case], metadata), [record])
            for key, value in [('model_revision', 'changed'), ('max_tokens', 500),
                               ('packages', {}), ('benchmark_sha256', 'changed')]:
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    load_v2_baseline(root, [case], metadata | {key: value})
            with self.assertRaisesRegex(ValueError, 'context, labels or prompt'):
                load_v2_baseline(root, [case | dict(context='different context')], metadata)
            with self.assertRaisesRegex(ValueError, 'case IDs'):
                load_v2_baseline(root, [case | dict(id='other')], metadata)
            record['prompt_hash'] = 'changed'
            (root / 'results.jsonl').write_text(json.dumps(record) + '\n')
            with self.assertRaisesRegex(ValueError, 'context, labels or prompt'):
                load_v2_baseline(root, [case], metadata)


if __name__ == '__main__':
    unittest.main()
