"""核对修订必须真正进入最终正文；覆盖失败、取消、隐私和流式保存边界。"""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from auth_fixtures import TestClient

from backend.generation import answer_review

from backend import main

from backend.rag import retrieval
from backend.generation.answer_policy import answer_profile
from backend.rag.planning import rule_plan


def completion(status='revised', answer='按当前输入得到56个参数。[1]', issues=None, finish='stop'):
    value = {'status': status, 'issues': issues if issues is not None else ['原摘要混用了另一组参数。'],
             'edits': [{'block': 0, 'replacement': answer}] if answer else []}
    return {'choices': [{'message': {'content': json.dumps(value, ensure_ascii=False)}, 'finish_reason': finish}]}


class ReviewUnitTests(unittest.IsolatedAsyncioTestCase):
    async def run_review(self, provider, profile=None, scope=None):
        return await answer_review.review_answer('DRAFT [1]',
            [{'role': 'system', 'content': '旧长提示'}, {'role': 'user', 'content': 'CURRENT_QUESTION_AND_EVIDENCE'}],
            profile or {'level': 'detailed'}, scope or {}, 1,
            'https://provider.example/v1', 'test-key', 'unchanged-model', provider)

    async def test_simple_question_skips_extra_call_but_strict_brief_is_checked(self):
        provider = AsyncMock(return_value=completion('unchanged', '', []))
        text, diagnostics = await self.run_review(provider, {'level': 'concise'})
        provider.assert_not_awaited()
        self.assertEqual((text, diagnostics['status']), ('DRAFT [1]', 'skipped'))
        _, diagnostics = await self.run_review(provider, {'level': 'concise'}, {'strict': True})
        self.assertEqual(diagnostics['trigger'], 'strict_source')
        provider.assert_awaited_once()
        ordinary = '为什么CNN适合局部模式？'
        self.assertIsNone(answer_review.review_trigger(answer_profile(ordinary), {}, ordinary))
        self.assertEqual(answer_review.review_trigger(answer_profile(ordinary), {'strict': True}, ordinary), 'strict_source')

    async def test_corrected_answer_and_trace_use_only_current_context(self):
        provider = AsyncMock(return_value=completion())
        text, diagnostic = await self.run_review(provider)
        self.assertEqual(text, '按当前输入得到56个参数。[1]')
        self.assertEqual(diagnostic['status'], 'revised')
        self.assertNotEqual(diagnostic['draft_sha256'], diagnostic['final_sha256'])
        body = provider.await_args.args[3]
        self.assertEqual(body['model'], 'unchanged-model')
        self.assertEqual(body['response_format'], {'type': 'json_object'})
        self.assertIn('CURRENT_QUESTION_AND_EVIDENCE', str(body))
        self.assertNotIn('旧长提示', str(body))
        self.assertNotIn('reference_answer', str(body))

    async def test_rejects_invalid_citations_incomplete_or_inconsistent_json(self):
        for result in [completion(answer='新答案[99]'), completion(answer='新答案[1][证据99]'), completion(answer='没有引用'),
                       completion(answer='部分答案[1]', finish='length'), completion('unchanged'),
                       completion(issues=[]), {'choices': [{'message': {'content': 'not JSON'}, 'finish_reason': 'stop'}]}]:
            with self.subTest(result=result):
                text, diagnostics = await self.run_review(AsyncMock(return_value=result))
                self.assertEqual(text, 'DRAFT [1]')
                self.assertEqual(diagnostics['status'], 'unavailable')

    async def test_timeout_is_bounded_and_does_not_retry_or_erase_original(self):
        closed = []
        async def blocked(*args):
            try:
                await asyncio.Event().wait()
            finally:
                closed.append(True)
        provider = AsyncMock(side_effect=blocked)
        with patch.object(answer_review, 'TIMEOUT_SECONDS', .01):
            text, diagnostics = await self.run_review(provider)
        self.assertEqual(text, 'DRAFT [1]')
        self.assertEqual(diagnostics['error_type'], 'TimeoutError')
        self.assertEqual(closed, [True])
        provider.assert_awaited_once()

    async def test_missing_repeated_or_unchanged_blocks_are_rejected_atomically(self):
        for edits in [[{'block': 99, 'replacement': '新答案[1]'}],
                      [{'block': 0, 'replacement': '新答案[1]'}, {'block': 0, 'replacement': '另一个答案[1]'}],
                      [{'block': 0, 'replacement': 'DRAFT [1]'}]]:
            result = completion()
            value = json.loads(result['choices'][0]['message']['content'])
            value['edits'] = edits
            result['choices'][0]['message']['content'] = json.dumps(value)
            text, diagnostics = await self.run_review(AsyncMock(return_value=result))
            self.assertEqual((text, diagnostics['status']), ('DRAFT [1]', 'unavailable'))

    async def test_multiple_edits_apply_to_original_offsets_and_only_supported_provider_disables_thinking(self):
        result = completion()
        result['choices'][0]['message']['content'] = json.dumps({'status': 'revised', 'issues': ['改正两个结果'],
            'edits': [{'block': 1, 'replacement': 'B [1]'}, {'block': 0, 'replacement': 'AAA'}]})
        status, _, text, _ = answer_review.parse_review(result, 1, 'LEFT\n\nRIGHT [1]')
        self.assertEqual((status, text), ('revised', 'AAA\n\nB [1]'))
        provider = AsyncMock(return_value=completion('unchanged', '', []))
        _, diagnostic = await answer_review.review_answer('DRAFT [1]',
            [{'role': 'system', 'content': 'test'}, {'role': 'user', 'content': 'public evidence'}],
            {'level': 'detailed'}, {}, 1, 'https://api.siliconflow.cn/v1', 'test-key',
            'deepseek-ai/DeepSeek-V3.2', provider)
        self.assertFalse(provider.await_args.args[3]['enable_thinking'])
        self.assertTrue(diagnostic['thinking_disabled'])

    async def test_paragraph_numbers_keep_code_and_formula_whole(self):
        text = '解释\n\n```python\nx=1\n\ny=2\n```\n\n$$\na=b\n\nc=d\n$$\n\n结尾[1]'
        blocks = answer_review.draft_blocks(text)
        self.assertEqual(len(blocks), 4)
        self.assertIn('x=1\n\ny=2', blocks[1]['text'])
        self.assertIn('a=b\n\nc=d', blocks[2]['text'])
        self.assertTrue(all(text[b['start']:b['end']] == b['text'] for b in blocks))

    async def test_real_provider_boundary_assembles_streamed_review_json_and_closes_it(self):
        closed = []
        async def chunks(*args):
            try:
                yield {'text': '{"status":"unchanged",', 'finish_reason': None}
                yield {'text': '"issues":[],"edits":[]}', 'finish_reason': 'stop'}
            finally:
                closed.append(True)
        with patch('backend.infra.streaming.stream_completion', chunks):
            text, diagnostic = await self.run_review(retrieval.provider_post)
        self.assertEqual((text, diagnostic['status']), ('DRAFT [1]', 'unchanged'))
        self.assertEqual(diagnostic['transport'], 'stream')
        self.assertIsNotNone(diagnostic['stream']['first_text_ms'])
        self.assertEqual(closed, [True])

    async def test_real_stream_timeout_records_progress_and_closes_connection(self):
        closed = []
        async def chunks(*args):
            try:
                yield {'text': '{"status":', 'finish_reason': None}
                await asyncio.Event().wait()
            finally:
                closed.append(True)
        with patch('backend.infra.streaming.stream_completion', chunks), patch.object(answer_review, 'TIMEOUT_SECONDS', .01):
            text, diagnostic = await self.run_review(retrieval.provider_post)
        self.assertEqual((text, diagnostic['status']), ('DRAFT [1]', 'unavailable'))
        self.assertEqual(diagnostic['stream']['received_chars'], len('{"status":'))
        self.assertEqual(closed, [True])

    async def test_cancel_propagates_and_error_details_never_leak(self):
        with self.assertRaises(asyncio.CancelledError):
            await self.run_review(AsyncMock(side_effect=asyncio.CancelledError()))
        _, diagnostic = await self.run_review(AsyncMock(side_effect=httpx.ConnectError('SECRET_MUST_NOT_LEAK')))
        self.assertNotIn('SECRET_MUST_NOT_LEAK', str(diagnostic))
        self.assertEqual(diagnostic['error_type'], 'ConnectError')


class ReviewFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(main.app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def run_flow(self, question, provider, *, streaming=False, local=False, mode='research'):
        plan = rule_plan(question, main.retriever, 'cnn-1d')
        plan['duration_ms'] = 0
        section = next(s for s in main.retriever.sections if s['id'] == 'cnn-1d:mechanism')
        evidence = {**section, 'section_id': section['id'], 'section_slug': section['slug'],
                    'cloud_allowed': not local, 'sources': []}
        retrieval = {'evidence': [evidence], 'warnings': [], 'paths': [], 'steps': [], 'duration_ms': 0,
                     'stop_reason': '完成', 'channels': {}, 'reranked': False, 'remote_allowed': not local}
        async def stream(*args):
            yield {'text': 'DRAFT [1]', 'finish_reason': 'stop'}
        with patch.object(main, 'plan_question', AsyncMock(return_value=plan)), \
             patch.object(main, 'execute_plan', AsyncMock(return_value=retrieval)), \
             patch.object(main, 'provider_post', provider), patch.object(main, 'history', return_value=[]), \
             patch.object(main, 'save_turn') as save, patch.object(main, 'stream_completion', stream):
            response = self.client.post('/api/chat/stream' if streaming else '/api/chat',
                                        json={'question': question, 'model_id': 'cnn-1d', 'mode': mode})
        self.assertEqual(response.status_code, 200)
        if streaming:
            frames = [frame.split('\ndata: ', 1) for frame in response.text.split('\n\n') if frame.startswith('event:')]
            result = json.loads(next(raw for event, raw in frames if event == 'event: done'))
            self.assertIn('reviewing', response.text)
        else:
            result = response.json()
        self.assertEqual(save.call_args.args[2], result['answer'])
        return result

    def test_regular_generation_keeps_draft_and_saves_revision(self):
        provider = AsyncMock(side_effect=[{'choices': [{'message': {'content': 'DRAFT [1]'}, 'finish_reason': 'stop'}]}, completion()])
        result = self.run_flow('请推导1D CNN的分组参数量', provider)
        self.assertEqual(result['answer'], '按当前输入得到56个参数。[1]')
        self.assertEqual(result['generation_diagnostics']['review']['draft_answer'], 'DRAFT [1]')
        self.assertEqual(provider.await_count, 2)

    def test_stream_done_and_saved_answer_use_revision(self):
        provider = AsyncMock(return_value=completion())
        result = self.run_flow('请推导1D CNN的分组参数量', provider, streaming=True)
        self.assertTrue(result['generated'])
        self.assertEqual(result['answer'], '按当前输入得到56个参数。[1]')
        provider.assert_awaited_once()

    def test_failed_review_preserves_generation_and_adds_explicit_warning(self):
        provider = AsyncMock(side_effect=[{'choices': [{'message': {'content': 'DRAFT [1]'}, 'finish_reason': 'stop'}]}, RuntimeError('unavailable')])
        result = self.run_flow('请推导1D CNN的分组参数量', provider)
        self.assertTrue(result['generated'])
        self.assertEqual(result['answer'], 'DRAFT [1]')
        self.assertIn('补充核对未完成', ' '.join(result['warnings']))
        self.assertEqual(result['generation_diagnostics']['review']['status'], 'unavailable')

    def test_local_only_question_never_sends_draft_to_reviewer(self):
        provider = AsyncMock()
        result = self.run_flow('请推导1D CNN的分组参数量', provider, local=True)
        provider.assert_not_awaited()
        self.assertFalse(result['generated'])

    def test_learning_mode_keeps_one_generation_call_for_technical_question(self):
        provider = AsyncMock(return_value={'choices': [{'message': {'content': 'DRAFT [1]'}, 'finish_reason': 'stop'}]})
        result = self.run_flow('请推导1D CNN的分组参数量', provider, mode='learning')
        provider.assert_awaited_once()
        self.assertEqual(result['generation_diagnostics']['review']['skip_reason'], 'learning_mode')


if __name__ == '__main__':
    unittest.main()
