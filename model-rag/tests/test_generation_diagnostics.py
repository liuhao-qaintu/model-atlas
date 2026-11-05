"""失败诊断区分超时和空正文，只保留元数据，不记录秘密或推理正文。"""
import unittest
import asyncio
from unittest.mock import AsyncMock, patch

import httpx
from auth_fixtures import TestClient

from backend import main
from backend.rag.planning import rule_plan


class GenerationDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(main.app); cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls): cls.context.__exit__(None, None, None)

    def request(self, provider):
        plan = rule_plan('LSTM输入输出', main.retriever, 'lstm')
        plan['duration_ms'] = 0
        evidence = {'id': 'lstm:io:0', 'section_id': 'lstm:io', 'title': 'LSTM输入输出', 'content': 'output为逐步隐藏状态。',
                    'model_id': 'lstm', 'cloud_allowed': True, 'sources': [], 'source_ids': [], 'reader_url': '/models/lstm?section=io'}
        retrieval = {'evidence': [evidence], 'warnings': [], 'paths': [], 'steps': [], 'stop_reason': '已完成',
                     'duration_ms': 0, 'channels': {}, 'reranked': False, 'remote_allowed': True}
        with patch.object(main, 'plan_question', AsyncMock(return_value=plan)), \
             patch.object(main, 'execute_plan', AsyncMock(return_value=retrieval)), \
             patch.object(main, 'provider_post', provider), patch.object(main, 'history', return_value=[]), \
             patch.object(main, 'save_turn'):
            response = self.client.post('/api/chat', json={'question': 'LSTM输入输出', 'model_id': 'lstm'})
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_empty_completion_has_metadata_without_reasoning_body(self):
        response = {'choices': [{'message': {'content': '', 'reasoning_content': 'SECRET_REASONING_NOT_STORED'}, 'finish_reason': 'length'}],
                    'usage': {'completion_tokens': 1600}}
        result = self.request(AsyncMock(return_value=response))
        self.assertFalse(result['generated'])
        self.assertEqual(result['generation_diagnostics']['finish_reason'], 'length')
        self.assertEqual(result['generation_diagnostics']['answer_chars'], 0)
        self.assertEqual(result['generation_diagnostics']['completion_tokens'], 1600)
        self.assertNotIn('SECRET_REASONING_NOT_STORED', str(result))

    def test_timeout_does_not_expose_exception_details(self):
        result = self.request(AsyncMock(side_effect=httpx.ReadTimeout('SECRET_REQUEST_DETAIL')))
        self.assertFalse(result['generated'])
        self.assertEqual(result['generation_diagnostics']['error_type'], 'ReadTimeout')
        self.assertNotIn('SECRET_REQUEST_DETAIL', str(result))
        self.assertTrue(any('超时' in warning for warning in result['warnings']))

    def test_strict_nonstream_deadline_cancels_provider_and_saves_excerpt(self):
        question = '只依据PyTorch 2.6.0官方nn.Transformer.forward，返回什么？'
        plan = rule_plan(question, main.retriever, 'transformer'); plan['duration_ms'] = 0
        evidence = {'title': '接口原文', 'content': 'SOURCE_BODY return output', 'section_id': 'fixture:forward',
                    'model_id': 'transformer', 'cloud_allowed': True, 'sources': []}
        retrieval = {'evidence': [evidence], 'warnings': [], 'paths': [], 'steps': [], 'stop_reason': '完成',
                     'duration_ms': 0, 'channels': {}, 'reranked': False, 'remote_allowed': True}
        closed = []
        async def waiting(*args):
            try: await asyncio.Event().wait()
            finally: closed.append(True)
        with patch.object(main, 'plan_question', AsyncMock(return_value=plan)), \
             patch.object(main, 'execute_plan', AsyncMock(return_value=retrieval)), \
             patch.object(main, 'provider_post', waiting), patch.object(main, 'history', return_value=[]), \
             patch.object(main, 'STRICT_GENERATION_SECONDS', 0.01), patch.object(main, 'save_turn') as save:
            result = self.client.post('/api/chat', json={'question': question, 'model_id': 'transformer'}).json()
        self.assertFalse(result['generated'])
        self.assertEqual(result['generation_diagnostics']['error_type'], 'TimeoutError')
        self.assertIn('SOURCE_BODY', result['answer'])
        self.assertEqual(closed, [True])
        self.assertEqual(save.call_args.args[2], result['answer'])


if __name__ == '__main__': unittest.main()
