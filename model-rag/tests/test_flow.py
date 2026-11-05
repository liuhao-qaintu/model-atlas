"""验证行为边界；模型响应使用替身，测试不产生外部 API 费用。"""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from auth_fixtures import TestClient
from sqlalchemy import delete
from sqlalchemy.orm import Session

from backend import main
from backend.infra.database import ChatMessage, ChatSession, engine, history
from backend.infra.knowledge import parse_sections, split_blocks
from knowledge_fixtures import without_model_material


class KnowledgeFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(main.app)
        cls.client = cls.client_context.__enter__()
        cls.sessions = []

    @classmethod
    def tearDownClass(cls):
        # 只清理本次测试创建且已记录的会话，不触碰用户的会话。
        with Session(engine) as db:
            db.execute(delete(ChatMessage).where(ChatMessage.session_id.in_(cls.sessions)))
            db.execute(delete(ChatSession).where(ChatSession.id.in_(cls.sessions)))
            db.commit()
        cls.client_context.__exit__(None, None, None)

    def setUp(self):
        self.vector_patch = patch.object(main.retriever, 'vectors', None)
        self.rerank_patch = patch('backend.infra.config.RERANK_KEY', '')
        self.planning_patch = patch('backend.rag.planning.provider_post', AsyncMock(side_effect=RuntimeError('test-only planner fallback')))
        self.vector_patch.start()
        self.rerank_patch.start()
        self.planning_patch.start()
        self.addCleanup(self.vector_patch.stop)
        self.addCleanup(self.rerank_patch.stop)
        self.addCleanup(self.planning_patch.stop)

    def chat(self, **fields):
        session = fields.get('session_id') or str(uuid4())
        if session not in self.sessions:
            self.sessions.append(session)
        return self.client.post('/api/chat', json={'session_id': session, **fields})

    def test_catalog_and_reader_order(self):
        models = self.client.get('/api/models').json()
        self.assertEqual({m['id'] for m in models if m['ready']}, {'cnn-1d', 'cnn-2d', 'resnet-18', 'autoencoder', 'lstm', 'transformer', 'vit', 'clip', 'detr', 'llava', 'dinov2', 'mamba', 'mamba2', 'mamba3', 'siglip2', 'dinov3'})
        self.assertTrue(models[0]['ready'])
        detail = self.client.get('/api/models/resnet-18').json()
        self.assertEqual(detail['sections'][0]['slug'], 'overview')
        code = next(s for s in detail['sections'] if s['slug'] == 'code')
        self.assertIn('resnet18(weights=None)', code['content'])

    def test_code_and_formula_blocks_stay_whole(self):
        code = '```python\n' + '\n'.join('## inside = 1' for _ in range(20)) + '\n```'
        formula = '$$\na = b + c\n\n d = e\n$$'
        chunks = split_blocks('前文\n\n' + code + '\n\n' + formula + '\n\n后文', max_chars=50)
        self.assertTrue(any(code in chunk for chunk in chunks))
        self.assertTrue(any(formula in chunk for chunk in chunks))
        self.assertEqual(len(parse_sections('## A\n' + code + '\n## B\n正文')), 2)

    def test_invalid_context_and_blank_question(self):
        for fields in [
            {'question': '   '},
            {'question': '输入是什么', 'model_id': 'missing'},
            {'question': '输入是什么', 'model_id': 'cnn-1d', 'section_id': 'resnet-18:io'},
        ]:
            self.assertEqual(self.client.post('/api/search', json=fields).status_code, 422)

    def test_keyword_retrieval_parent_context(self):
        result = self.client.post('/api/search', json={'question': '1D CNN 输入通道长度', 'model_id': 'cnn-1d', 'section_id': 'cnn-1d:io'}).json()
        self.assertTrue(result['evidence'])
        self.assertLessEqual(len(result['evidence']), 5)
        self.assertTrue(all(e['model_id'] == 'cnn-1d' for e in result['evidence']))
        io = next(e for e in result['evidence'] if e['section_slug'] == 'io')
        expected = next(s for s in main.retriever.sections if s['id'] == io['section_id'])
        self.assertEqual(io['content'], expected['content'])

    def test_graph_paths_have_connectivity_and_evidence(self):
        paths = main.retriever.paths(['resnet-18'])
        routed = self.client.post('/api/search', json={'question': 'ResNet-18 与 BasicBlock、残差连接有什么联系？'}).json()
        self.assertTrue(routed['paths'])
        self.assertTrue(any(p['hops'] == 2 for p in paths))
        for path in paths:
            node = path['root']
            for edge in path['edges']:
                self.assertIn(node, (edge['source'], edge['target']))
                node = edge['target'] if node == edge['source'] else edge['source']
                self.assertTrue(edge['source_ids'])
                self.assertTrue(all(s in main.retriever.sources for s in edge['source_ids']))

    def test_pending_model_does_not_generate_answer(self):
        provider = AsyncMock()
        with patch('backend.main.retriever', without_model_material(main.retriever, 'autoencoder')), patch('backend.main.provider_post', provider):
            result = self.chat(question='Autoencoder 的输入输出是什么？').json()
        self.assertFalse(result['generated'])
        self.assertEqual(result['evidence'], [])
        self.assertIn('尚未完成', result['answer'])
        provider.assert_not_awaited()

    def test_model_inference_followup_and_history_order(self):
        provider = AsyncMock(return_value={'choices': [{'message': {'content': '这是 ResNet-18 的解释。[1]'}}]})
        with patch('backend.main.provider_post', provider):
            first = self.chat(question='ResNet-18 的输入输出是什么？').json()
            second = self.chat(question='它去掉分类头之后呢？', session_id=first['session_id']).json()
        self.assertEqual(first['model_id'], 'resnet-18')
        self.assertEqual(second['model_id'], 'resnet-18')
        self.assertIn('ResNet-18', second['normalized_query'])
        self.assertTrue(first['links'])
        saved = history(first['session_id'])
        self.assertEqual([m['role'] for m in saved], ['user', 'assistant', 'user', 'assistant'])
        self.assertEqual(saved[-1]['content'], second['answer'])
        sent_messages = provider.await_args.kwargs if provider.await_args.kwargs else provider.await_args.args[3]
        self.assertIn('messages', sent_messages)
        self.assertEqual(second['execution']['retrieval'], 'local')
        self.assertEqual(second['execution']['generation'], 'remote')
        self.assertEqual(second['execution']['llm_model'], main.config.LLM_MODEL)
        self.assertTrue(second['execution']['generated'])

    def test_generation_failure_is_explicit(self):
        with patch('backend.main.provider_post', AsyncMock(side_effect=RuntimeError('offline'))):
            result = self.chat(question='1D CNN 的输入输出是什么？').json()
        self.assertFalse(result['generated'])
        self.assertTrue(result['evidence'])
        self.assertIn('未完成模型生成', ' '.join(result['warnings']))
        self.assertEqual(result['execution']['generation'], 'none')

    def test_invalid_citation_is_removed(self):
        provider = AsyncMock(return_value={'choices': [{'message': {'content': '已知结论 [1]，错误编号 [99]。'}}]})
        with patch('backend.main.provider_post', provider):
            result = self.chat(question='1D CNN 是什么？').json()
        self.assertNotIn('[99]', result['answer'])
        self.assertIn('[1]', result['answer'])

    def test_generated_links_require_existing_targets(self):
        answer = '解释 [1]。[虚构章节](/made-up)；[已存在章节](/models/cnn-1d?section=code)'
        provider = AsyncMock(return_value={'choices': [{'message': {'content': answer}}]})
        with patch('backend.main.provider_post', provider):
            result = self.chat(question='1D CNN 是什么？').json()
        self.assertNotIn('/made-up', result['answer'])
        self.assertIn('虚构章节', result['answer'])
        self.assertIn('](/models/cnn-1d?section=code)', result['answer'])

    def test_reranker_uses_returned_order(self):
        fake = AsyncMock(return_value={'results': [{'index': 0, 'relevance_score': 0.9}]})
        with patch('backend.infra.config.RERANK_KEY', 'test-placeholder'), patch('backend.rag.retrieval.provider_post', fake):
            result = asyncio.run(main.retriever.search('ResNet-18 输入', focus='resnet-18'))
        self.assertTrue(result['reranked'])
        self.assertEqual(len(result['evidence']), 5)
        first=result['evidence'][0]
        self.assertEqual(first['title']+'\n'+first['retrieval_excerpt'],fake.await_args.args[3]['documents'][0])
        self.assertEqual(fake.await_args.args[2], 'rerank')


if __name__ == '__main__':
    unittest.main()
