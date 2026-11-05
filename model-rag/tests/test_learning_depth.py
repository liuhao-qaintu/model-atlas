"""新增正文必须可完整提供、可独立检索，不能只在页面上存在。"""
import asyncio
import unittest
from unittest.mock import patch

from auth_fixtures import TestClient

from backend import main
from backend.generation.evidence_context import select_excerpt


class LearningDepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(main.app); cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_six_new_chapters_have_stable_links_and_fit_complete_context(self):
        for model in ('cnn-1d', 'resnet-18', 'transformer'):
            result = self.client.get(f'/api/models/{model}').json()
            slugs = ('mechanism', 'practice', 'io', 'masks') if model == 'transformer' else ('mechanism', 'practice')
            for slug in slugs:
                section = next(s for s in result['sections'] if s['id'] == f'{model}:{slug}')
                excerpt = select_excerpt(section['content'], '请详细解释')
                self.assertEqual(excerpt['strategy'], 'full')
                self.assertEqual(excerpt['text'], section['content'])
                self.assertTrue(all(sid in main.retriever.sources for sid in section['source_ids']))

    def test_specific_questions_retrieve_actual_new_body_without_remote_calls(self):
        queries = [('cnn-1d', '分组卷积groups权重形状怎样计算参数量', 'mechanism'),
                   ('resnet-18', 'requires_grad=False冻结参数BatchNorm running_mean buffer train eval', 'practice'),
                   ('transformer', 'teacher forcing标签错位 reshape ignore_index交叉熵padding损失', 'practice'),
                   ('transformer', 'tgt_mask与tgt_key_padding_mask布尔掩码叠加 query key batch索引', 'masks'),
                   ('transformer', 'custom_decoder自定义解码器return输出形状TwoSymbolDecoder', 'io')]
        for model, question, expected in queries:
            with patch.object(main.retriever, 'vectors', None):
                result = asyncio.run(main.retriever.search(question, focus=model, allow_remote=False, use_rerank=False, strict_focus=True))
            self.assertIn(f'{model}:{expected}', [e['section_id'] for e in result['evidence']])
            self.assertLessEqual(len(result['evidence']), 5)

    def test_new_source_check_dates_do_not_inherit_old_global_dates(self):
        sources = {s['id']: s for s in self.client.get('/api/sources').json()}
        for sid in ('S07', 'S44', 'S45'):
            self.assertEqual(sources[sid]['checked_at'], '2026-10-16')
        self.assertIn('/vision/0.21/', sources['S07']['url'])


if __name__ == '__main__':
    unittest.main()
