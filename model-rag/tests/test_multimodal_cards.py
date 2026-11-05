"""视觉/图文资料的范围、别名和关系证据验证，不调用外部模型。"""
import asyncio
import unittest
from unittest.mock import patch

from auth_fixtures import TestClient

from backend import main
from backend.rag.graph import build_graph
from backend.rag.pipeline import execute_plan
from backend.rag.planning import rule_plan


class MultimodalCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(main.app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def setUp(self):
        self.retriever = main.retriever

    def test_cards_separate_teaching_code_from_official_sources(self):
        vit = self.client.get('/api/models/vit').json()
        clip = self.client.get('/api/models/clip').json()
        self.assertTrue(vit['ready'] and clip['ready'])
        sections = {s['slug']: s for s in clip['sections']}
        self.assertEqual(sections['code']['source_ids'], ['E06'])
        self.assertIn('不是官方 CLIP 主干', sections['code']['content'])
        self.assertIn('EOT', sections['io']['content'])
        self.assertTrue(any('class TinyPatchViT' in s['content'] for s in vit['sections']))
        sources = {s['id']: s for s in self.client.get('/api/sources').json()}
        self.assertIn('a1d071733d7111c9c014f024669f959182114e33', sources['S26']['url'])
        self.assertEqual(sources['E06']['url'], '/models/clip?section=code')

    def test_complete_vision_name_suppresses_nested_transformer(self):
        for name in ('Vision Transformer', 'VisionTransformer', 'TinyPatchViT'):
            with self.subTest(name=name):
                plan = rule_plan(name + ' 的输入输出是什么？', self.retriever)
                self.assertEqual(plan['model_ids'], ['vit'])
                self.assertEqual(plan['mode'], 'single')
        self.assertEqual(self.retriever.identify_models('activity sensitivity eclipse'), [])

    def test_standalone_transformer_remains_in_explicit_comparison(self):
        plan = rule_plan('比较 Vision Transformer 与 Transformer 的输入输出', self.retriever)
        self.assertEqual(set(plan['model_ids']), {'vit', 'transformer'})
        self.assertEqual(plan['mode'], 'comparison')

    def test_nested_clip_module_keeps_owner_and_relation_mode(self):
        plan = rule_plan('CLIP 中的 CLIP 文本 Transformer 与残差相加有什么关系？', self.retriever)
        self.assertEqual(plan['model_ids'], ['clip'])
        self.assertEqual(plan['mode'], 'relation')
        dual = rule_plan('CLIP 的输入与 ViT 的输出', self.retriever)
        self.assertEqual(set(dual['model_ids']), {'clip', 'vit'})

    def test_comparison_covers_both_cards_with_separate_reader_links(self):
        plan = rule_plan('ViT 和 CLIP 输出表示有什么不同？', self.retriever)
        with patch.object(self.retriever, 'vectors', None), patch('backend.infra.config.RERANK_KEY', ''):
            result = asyncio.run(execute_plan(plan, self.retriever))
        self.assertEqual({e['model_id'] for e in result['evidence']}, {'vit', 'clip'})
        self.assertLessEqual(len(result['evidence']), 5)
        self.assertTrue(all(step['status'] == 'found' for step in result['steps']))

    def test_clip_has_sourced_two_hop_path_to_vit(self):
        graph = build_graph(self.retriever, 'clip', 2)
        path = next(p for p in graph['paths'] if p['target'] == 'vit')
        self.assertEqual(path['hops'], 2)
        self.assertEqual([s['edge_id'] for s in path['steps']], ['clip-vision', 'clip-vit-family'])
        self.assertFalse(any(s['reverse'] for s in path['steps']))
        edges = {e['id']: e for e in graph['edges']}
        self.assertTrue(all(edges[s['edge_id']]['evidence'] for s in path['steps']))

    def test_clip_relation_steps_depend_on_existing_body(self):
        plan = rule_plan('CLIP 的文本编码器和残差相加有什么关系？', self.retriever)
        with patch.object(self.retriever, 'vectors', None), patch('backend.infra.config.RERANK_KEY', ''):
            result = asyncio.run(execute_plan(plan, self.retriever))
        self.assertGreater(len(result['steps']), 1)
        self.assertLessEqual(len(result['steps']), 3)
        self.assertTrue(all(e['model_id'] == 'clip' for e in result['evidence']))
        self.assertTrue(all(step['anchor']['proofs'] for step in result['steps'][1:]))

    def test_shared_class_token_preserves_different_output_purposes(self):
        graph = build_graph(self.retriever, 'class-token', 1)
        shared = next(s for s in graph['shared'] if s['node_id'] == 'class-token')
        self.assertEqual({s['model_name'] for s in shared['implementations']}, {'ViT', 'CLIP'})
        self.assertTrue(any('共享' in s['text'] for s in shared['implementations']))
        self.assertTrue(any('分类头' in s['text'] for s in shared['implementations']))
