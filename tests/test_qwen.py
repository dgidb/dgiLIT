from __future__ import annotations

import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dgilit import (
    InteractionClassificationPrompt, InteractionClassificationResult, Qwen3JsonClient,
)
from wags_llm.client.exceptions import (
    LLMEmptyResponseError, LLMInvocationError, LLMJsonDecodeError,
    LLMResponseFormatError,
)
from wags_llm.prompts import build_empty_registry
from wags_llm.services import StructuredTaskRunner


class QwenTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MagicMock()
        self.model = MagicMock()
        self.tokenizer.encode.return_value = [10, 11, 12]
        self.config = {'max_position_embeddings': 32768, 'quantization': {'bits': 4}}
        self.output = json.dumps({'drug': 'Venetoclax', 'gene': 'BCL2', 'interaction': True})
        self.mlx = MagicMock()
        self.lm = SimpleNamespace(
            load=MagicMock(return_value=(self.model, self.tokenizer, self.config)),
            stream_generate=MagicMock(side_effect=self.responses),
        )
        self.sampling = SimpleNamespace(make_sampler=MagicMock(return_value='sampler'))
        self.patches = [
            patch.dict(sys.modules, {'mlx': SimpleNamespace(core=self.mlx),
                                    'mlx.core': self.mlx, 'mlx_lm': self.lm,
                                    'mlx_lm.sample_utils': self.sampling}),
            patch('dgilit.qwen.platform.system', return_value='Darwin'),
            patch('dgilit.qwen.platform.machine', return_value='arm64'),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.client = Qwen3JsonClient()

    def responses(self, *args, **kwargs):
        return iter([SimpleNamespace(text=self.output, prompt_tokens=3,
                                     generation_tokens=12, generation_tps=20.0)])

    def test_generation_and_reuse(self):
        for _ in range(2):
            result = self.client.invoke_json('system', 'user', {'type': 'object'})
            self.assertTrue(result.parsed_json['interaction'])
        self.lm.load.assert_called_once_with('mlx-community/Qwen3-14B-4bit', return_config=True)
        self.mlx.set_default_device.assert_called_once_with(self.mlx.gpu)
        self.assertEqual(self.client.quantization_bits, 4)
        self.assertEqual(self.client.last_generation_stats['generation_tokens'], 12)
        args, kwargs = self.tokenizer.apply_chat_template.call_args
        self.assertFalse(kwargs['enable_thinking'])
        self.assertFalse(kwargs['tokenize'])
        self.assertTrue(kwargs['add_generation_prompt'])
        self.assertIn('"type": "object"', args[0][0]['content'])
        self.assertEqual(args[0][1], {'role': 'user', 'content': 'user'})
        self.lm.stream_generate.assert_called_with(
            self.model, self.tokenizer, prompt=[10, 11, 12],
            max_tokens=1000, sampler='sampler',
        )
        self.sampling.make_sampler.assert_called_with(temp=0.0)

    def test_sampling(self):
        self.client.temperature = 0.7
        self.client.invoke_json('system', 'user')
        self.sampling.make_sampler.assert_called_with(temp=0.7)

    def test_streamed_chunks(self):
        self.lm.stream_generate.side_effect = None
        self.lm.stream_generate.return_value = iter([
            SimpleNamespace(text=self.output[:10]),
            SimpleNamespace(text=self.output[10:], prompt_tokens=3,
                            generation_tokens=12, generation_tps=20.0),
        ])
        self.assertTrue(self.client.invoke_json('system', 'user').parsed_json['interaction'])

    def test_empty_stream(self):
        self.lm.stream_generate.side_effect = lambda *args, **kwargs: iter(())
        with self.assertRaises(LLMEmptyResponseError):
            self.client.invoke_json('system', 'user')
        self.assertEqual(self.client.last_generation_stats, {})

    def test_missing_dependencies(self):
        with patch.dict(sys.modules, {'mlx.core': None}):
            with self.assertRaisesRegex(ImportError, 'dgilit\\[qwen\\]'):
                Qwen3JsonClient()

    def test_response_errors(self):
        for text, error in [('', LLMEmptyResponseError), ('  ', LLMEmptyResponseError),
                            ('broken', LLMJsonDecodeError), ('[]', LLMResponseFormatError),
                            ('null', LLMResponseFormatError)]:
            with self.subTest(text=text):
                self.output = text
                with self.assertRaises(error):
                    self.client.invoke_json('system', 'user')

    def test_generation_failure(self):
        self.lm.stream_generate.side_effect = RuntimeError('out of memory')
        with self.assertRaisesRegex(LLMInvocationError, 'out of memory'):
            self.client.invoke_json('system', 'user')

    def test_context_overflow(self):
        self.client._context_window = 1000
        with self.assertRaisesRegex(LLMInvocationError, 'context window'):
            self.client.invoke_json('system', 'user')
        self.lm.stream_generate.assert_not_called()

    def test_runner_validation(self):
        prompt = InteractionClassificationPrompt(version='v2')
        registry = build_empty_registry()
        registry.register(prompt)
        runner = StructuredTaskRunner(client=self.client, prompt_registry=registry)
        params = dict(prompt_name=prompt.name, prompt_version='v2',
                      payload=prompt.build_payload('context', 'Venetoclax', 'BCL2'),
                      response_model=InteractionClassificationResult)
        self.assertIsInstance(runner.execute(**params), InteractionClassificationResult)
        self.output = '{}'
        with self.assertRaisesRegex(RuntimeError, 'Task failed'):
            runner.execute(**params)

    def test_requires_metal(self):
        self.mlx.metal.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'Metal'):
            Qwen3JsonClient()
        self.assertEqual(self.lm.load.call_count, 1)

    def test_requires_apple_silicon(self):
        with patch('dgilit.qwen.platform.machine', return_value='x86_64'):
            with self.assertRaisesRegex(RuntimeError, 'arm64'):
                Qwen3JsonClient()

    def test_requires_four_bit(self):
        self.config['quantization']['bits'] = 8
        with self.assertRaisesRegex(ValueError, '4-bit'):
            Qwen3JsonClient()

    def test_invalid_configuration(self):
        for kwargs in [{'max_tokens': 0}, {'max_tokens': True},
                       {'temperature': -1}, {'temperature': float('nan')}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Qwen3JsonClient(**kwargs)


if __name__ == '__main__':
    unittest.main()
