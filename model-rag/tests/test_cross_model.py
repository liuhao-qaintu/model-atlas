import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from auth_fixtures import TestClient
from backend import main
from backend.rag.planning import explicit_models, plan_question, rule_plan, validate_draft
from backend.rag.pipeline import execute_plan
from backend.rag.retrieval import Retriever
from knowledge_fixtures import without_model_material


class CrossModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(main.app)
        cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def setUp(self):
        self.retriever = Retriever()
        self.retriever.vectors = None
        self.ranking = patch('backend.infra.config.RERANK_KEY', '')
        self.ranking.start()
        self.addCleanup(self.ranking.stop)

    def test_relation_uses_known_direction_even_when_question_names_vit_first(self):
        for question in ['CLIP 和 ViT 有什么联系？', 'ViT 与 CLIP 的视觉主干有什么关系？']:
            with patch('backend.rag.planning.provider_post', AsyncMock()) as planner:
                plan = asyncio.run(plan_question(question, self.retriever))
            planner.assert_not_called()
            self.assertEqual(plan['mode'], 'relation')
            self.assertEqual(plan['model_id'], 'clip')
            self.assertEqual([edge['edge_id'] for edge in plan['relation_route']], ['clip-vision', 'clip-vit-family'])
            self.assertEqual([q['model_id'] for q in plan['queries']], ['clip'])

    def test_explicit_comparison_keeps_parallel_strategy(self):
        plan = rule_plan('比较 CLIP 和 ViT 的输出区别', self.retriever)
        self.assertEqual(plan['mode'], 'comparison')
        self.assertEqual(plan['relation_route'], [])
        self.assertEqual(len(plan['queries']), 2)

    def test_cross_model_hops_search_target_and_keep_previous_body_anchors(self):
        plan = rule_plan('CLIP 与 ViT 的视觉主干有什么关系？', self.retriever)
        spy = AsyncMock(wraps=self.retriever.search)
        with patch.object(self.retriever, 'search', spy):
            result = asyncio.run(execute_plan(plan, self.retriever))
        self.assertEqual([s['model_id'] for s in result['steps']], ['clip', 'clip', 'vit'])
        self.assertEqual([s['depends_on'] for s in result['steps']], [None, 's1', 's2'])
        self.assertEqual({e['model_id'] for e in result['evidence']}, {'clip', 'vit'})
        self.assertLessEqual(len(result['evidence']), 5)
        for step in result['steps'][1:]:
            self.assertIsNotNone(step['anchor']['proofs'][0]['reference'])

    def test_mentions_without_declared_quote_do_not_trigger_a_hop(self):
        plan = rule_plan('CLIP 与 ViT 有什么关系？', self.retriever)
        body = {'evidence': [{'section_id': 'clip:overview', 'model_id': 'clip', 'title': 'test', 'content': 'CLIP 的 ViT 视觉编码器相关讨论', 'reader_url': '/models/clip', 'channels': [], 'sources': []}], 'warnings': [], 'paths': [], 'channels': {}, 'reranked': False}
        fake = AsyncMock(return_value=body)
        with patch.object(self.retriever, 'search', fake):
            result = asyncio.run(execute_plan(plan, self.retriever))
        self.assertEqual(fake.await_count, 1)
        self.assertEqual(result['paths'], [])
        self.assertIn('未支持完整跨模型路径', ' '.join(result['warnings']))

    def test_missing_target_body_stops_without_borrowing_other_model(self):
        active = without_model_material(self.retriever, 'vit')
        plan = rule_plan('CLIP 与 ViT 有什么关系？', active)
        result = asyncio.run(execute_plan(plan, active))
        self.assertEqual(result['steps'][-1]['model_id'], 'vit')
        self.assertEqual(result['steps'][-1]['status'], 'missing')
        self.assertTrue(all(e['model_id'] == 'clip' for e in result['evidence']))

    def test_local_only_target_blocks_all_remote_stages(self):
        self.retriever.chunks = [{**c, 'cloud_allowed': False} if c['model_id'] == 'vit' else c for c in self.retriever.chunks]
        with patch('backend.rag.planning.provider_post', AsyncMock()) as planner:
            plan = asyncio.run(plan_question('CLIP 和 ViT 有什么关系？', self.retriever))
        planner.assert_not_called()
        self.assertFalse(plan['remote_allowed'])
        spy = AsyncMock(wraps=self.retriever.search)
        with patch.object(self.retriever, 'search', spy), patch('backend.rag.pipeline.provider_post', AsyncMock()) as provider:
            result = asyncio.run(execute_plan(plan, self.retriever))
        provider.assert_not_called()
        self.assertTrue(all(call.kwargs['allow_remote'] is False for call in spy.await_args_list))
        self.assertFalse(result['remote_allowed'])

    def test_route_query_validation_keeps_both_entities_but_only_root_query(self):
        plan = rule_plan('CLIP 与 ViT 有什么关系？', self.retriever)
        good = {'query': plan['normalized_query'], 'queries': [{'model_id': 'clip', 'query': 'CLIP 与 ViT 的视觉主干关系'}]}
        self.assertEqual(validate_draft(json.dumps(good), plan, self.retriever)[1][0]['model_id'], 'clip')
        good['queries'][0]['model_id'] = 'vit'
        with self.assertRaises(ValueError):
            validate_draft(json.dumps(good), plan, self.retriever)

    def test_owned_vit_does_not_hide_independently_mentioned_vit(self):
        self.assertEqual(set(explicit_models('CLIP 的 ViT 视觉主干与 ViT 架构有什么关系？', self.retriever)), {'clip', 'vit'})

    def test_named_followup_requires_real_history_and_accepts_transition_word(self):
        previous = [{'role': 'user', 'content': '比较 CLIP 与 ViT 输出'}, {'role': 'assistant', 'metadata': {'plan': {'model_ids': ['clip', 'vit']}}}]
        plan = rule_plan('那 Torchvision ViT 的对应模块怎么读取 CLS？', self.retriever, previous=previous)
        self.assertTrue(plan['followup'])
        self.assertEqual(plan['model_id'], 'vit')
        self.assertFalse(rule_plan('CLIP 的输出是什么，它直接是概率吗？', self.retriever)['followup'])

    def test_official_query_reserves_verified_source_candidates(self):
        if not self.retriever.document_kinds:
            self.skipTest('本机尚未导入固定公开来源')
        result = asyncio.run(self.retriever.search('官方 CLIP ViT 视觉编码器读取首 token 后如何投影', 'clip', top_k=8, use_rerank=False, strict_focus=True))
        self.assertTrue(result['channels'].get('原始来源'))
        self.assertTrue(any('self.ln_post' in e['content'] and 'x = x @ self.proj' in e['content'] for e in result['evidence']))

    def test_official_only_constraint_survives_query_rewrite_and_excludes_teaching_cards(self):
        if not self.retriever.document_kinds:
            self.skipTest('本机尚未导入固定公开来源')
        draft = json.dumps({'query': 'ViT 如何读取 CLS 并分类', 'queries': [{'model_id': 'vit', 'query': 'ViT forward CLS 分类输出'}]})
        previous = [{'role': 'user', 'content': '比较 CLIP 与 ViT'}, {'role': 'assistant', 'metadata': {'plan': {'model_ids': ['clip', 'vit']}}}]
        with patch('backend.rag.planning.provider_post', AsyncMock(return_value={'choices': [{'message': {'content': draft}}]})):
            plan = asyncio.run(plan_question('那 Torchvision ViT 如何分类？请只沿用官方实现。', self.retriever, previous=previous))
        self.assertEqual(plan['source_kinds'], ['official_code'])
        result = asyncio.run(execute_plan(plan, self.retriever))
        self.assertTrue(result['evidence'])
        self.assertTrue(all(self.retriever.document_kinds.get(e.get('document_id')) == 'official_code' for e in result['evidence']))

    def test_missing_official_body_does_not_fall_back_to_teaching_cards(self):
        plan = rule_plan('Autoencoder 输入输出是什么？只参考官方实现。', self.retriever)
        result = asyncio.run(execute_plan(plan, self.retriever))
        self.assertEqual(result['evidence'], [])
        self.assertEqual(result['steps'][0]['status'], 'missing')
