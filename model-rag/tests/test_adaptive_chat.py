"""真实失败的主体漂移，以及非预设答案的动态补查与授权边界。"""
import asyncio
import json
import unittest
from copy import deepcopy
from unittest.mock import AsyncMock, patch

from backend.rag.adaptive_chat import ChatEvidenceAgent, needs_agent, refine_events
from backend.rag.planning import rule_plan
from test_scope_expansion import Fixture
from backend import main


class FollowupSubjectTests(unittest.TestCase):
    def setUp(self):
        self.r = Fixture()
        self.r.paths = lambda *_: []
        self.r.models.update({
            'ae': {'id': 'ae', 'name': 'Autoencoder', 'kind': '方法家族', 'aliases': ['自编码器', 'autoencoder']},
            'task': {'id': 'task', 'name': '故障诊断', 'kind': '用户模型', 'category': '应用主题', 'aliases': ['故障诊断']},
            'tag': {'id': 'tag', 'name': '故障诊断', 'kind': '知识标签', 'aliases': ['故障诊断']}})
        self.previous = [{'role': 'assistant', 'metadata': {'model_id': 'ae', 'plan': {'model_ids': ['ae']}}}]

    def test_application_task_keeps_subject_in_exact_user_followup(self):
        plan = rule_plan('这个模块可以用于故障诊断吗？应用场景有哪些/', self.r, previous=self.previous)
        self.assertEqual(plan['model_ids'], ['ae'])
        self.assertIn('Autoencoder', plan['normalized_query'])
        self.assertIn('故障诊断', plan['normalized_query'])
        self.assertNotEqual(plan['mode'], 'comparison')

    def test_old_mistaken_task_turn_does_not_poison_next_followup(self):
        previous = self.previous + [{'role': 'assistant', 'metadata': {'plan': {'model_ids': ['tag', 'task']}}}]
        self.assertEqual(rule_plan('它的实际用途呢？', self.r, previous=previous)['model_ids'], ['ae'])

    def test_explicit_new_model_still_changes_subject(self):
        self.assertEqual(rule_plan('残差网络的输出呢？', self.r, previous=self.previous)['model_ids'], ['resnet-18'])

    def test_standalone_task_uses_global_search_and_comparison_is_not_invented(self):
        plan = rule_plan('故障诊断有哪些应用？', self.r)
        self.assertEqual(plan['model_ids'], [])
        self.assertEqual(plan['mode'], 'single')

    def test_multi_model_pronoun_requires_clarification(self):
        previous = [{'role': 'assistant', 'metadata': {'plan': {'model_ids': ['ae', 'resnet-18']}}}]
        self.assertIsNotNone(rule_plan('这个模块能用于故障诊断吗？', self.r, previous=previous)['clarification'])


class EvidenceLoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.r = Fixture()
        self.primary = dict(evidence=[{**self.r.sections[2], 'section_id': 'resnet:overview', 'sources': []}],
                            warnings=[], paths=[], steps=[], channels={}, duration_ms=0, remote_allowed=True, reranked=False)
        self.plan = rule_plan('残差网络可以用于故障诊断吗？应用场景有哪些？', self.r)
        rerank = patch('backend.rag.adaptive_chat.config.RERANK_KEY', '')
        rerank.start()
        self.addCleanup(rerank.stop)

    def check(self, status='missing', ids=None, actions=None):
        return {'checks': [{'question': '应用及条件', 'status': status, 'evidence_ids': ids or []}], 'actions': actions or []}

    async def test_gap_changes_query_then_reassesses_actual_new_sources(self):
        decision = AsyncMock(side_effect=[self.check(actions=[{'tool': 'search', 'query': 'residual fault diagnosis', 'model_id': None}]),
                                         self.check('inference', ['resnet:overview', 'b:1'])])
        result = await ChatEvidenceAgent(self.r, self.plan, self.primary, model_call=decision).run()
        self.assertEqual(decision.await_count, 2)
        self.assertEqual(result['agentic']['tool_calls'], 1)
        self.assertEqual(result['agentic']['stop_reason'], 'evidence_sufficient')
        self.assertIn('b:1', [e['section_id'] for e in result['evidence']])
        self.assertIn('resnet:overview', [e['section_id'] for e in result['evidence']])
        self.assertNotIn('PRIVATE SECRET', str(decision.call_args_list))
        self.assertNotIn('secret', self.r.search.await_args.kwargs['allowed_section_ids'])

    async def test_repeated_queries_stop_instead_of_looping(self):
        decision = AsyncMock(return_value=self.check(actions=[{'tool': 'search', 'query': 'unknown concept', 'model_id': None}]))
        result = await ChatEvidenceAgent(self.r, self.plan, self.primary, model_call=decision).run()
        self.assertEqual(result['agentic']['stop_reason'], 'repeated_actions')
        self.assertEqual(result['agentic']['tool_calls'], 1)

    async def test_one_invalid_action_does_not_discard_the_other_valid_search(self):
        actions = [{'tool': 'search', 'query': 'residual application', 'model_id': None},
                   {'tool': 'read_section', 'query': 'read unknown chapter', 'model_id': 'doc2', 'section_id': None}]
        decision = AsyncMock(side_effect=[self.check(actions=actions), self.check('supported', ['b:1'])])
        result = await ChatEvidenceAgent(self.r, self.plan, self.primary, model_call=decision).run()
        self.assertEqual(result['agentic']['tool_calls'], 1)
        self.assertEqual(result['agentic']['stop_reason'], 'evidence_sufficient')
        self.assertEqual(result['agentic']['diagnostics'][0]['rejected_actions'], ['未获准章节'])

    async def test_fabricated_evidence_or_private_read_cannot_be_approved(self):
        for raw in [self.check('supported', ['invented']), self.check(actions=[{'tool': 'read_section', 'query': 'read', 'section_id': 'secret'}])]:
            result = await ChatEvidenceAgent(self.r, self.plan, self.primary, model_call=AsyncMock(return_value=raw)).run()
            self.assertEqual(result['agentic']['stop_reason'], 'assessment_unavailable')
            self.assertEqual(result['agentic']['tool_calls'], 0)
        self.r.search.assert_not_awaited()

    async def test_unavailable_planner_retains_initial_evidence_and_records_gap(self):
        result = await ChatEvidenceAgent(self.r, self.plan, self.primary, model_call=AsyncMock(side_effect=TimeoutError())).run()
        self.assertEqual(result['evidence'][0]['section_id'], 'resnet:overview')
        self.assertEqual(result['agentic']['stop_reason'], 'assessment_unavailable')

    async def test_strict_document_private_history_and_simple_questions_do_not_start_agent(self):
        with patch('backend.rag.adaptive_chat.config.LLM_KEY', 'placeholder'):
            self.assertTrue(needs_agent(self.plan['original_query'], self.plan, self.primary))
            self.assertFalse(needs_agent('输入是什么？', self.plan, self.primary))
            self.assertFalse(needs_agent('请推导1D CNN的分组参数量', self.plan, self.primary))
            self.assertFalse(needs_agent(self.plan['original_query'], self.plan, self.primary, False))
            for change in [{'document_ids': ['a']}, {'source_scope': {'strict': True}}, {'remote_allowed': False}]:
                self.assertFalse(needs_agent(self.plan['original_query'], {**self.plan, **change}, self.primary))

    async def test_cancelled_stream_cancels_pending_model_call(self):
        started, cancelled = asyncio.Event(), asyncio.Event()
        async def pending(_):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        original = ChatEvidenceAgent
        def factory(*args, **kwargs):
            return original(*args, **kwargs, model_call=pending)
        with patch('backend.rag.adaptive_chat.ChatEvidenceAgent', factory):
            events = refine_events(self.r, self.plan, self.primary)
            await anext(events)
            await started.wait()
            await events.aclose()
        self.assertTrue(cancelled.is_set())

    async def test_chat_entry_really_executes_loop_and_saves_matching_references(self):
        decisions = [self.check(actions=[{'tool': 'search', 'query': 'residual application', 'model_id': None}]),
                     self.check('inference', ['resnet:overview', 'b:1'])]
        provider = AsyncMock(side_effect=[{'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(d)}}]} for d in decisions])
        plan = {**self.plan, 'duration_ms': 0}
        request = main.ChatRequest(question=self.plan['original_query'])
        generation = AsyncMock(return_value={'choices': [{'finish_reason': 'stop', 'message': {'content': '据此可设计，仍需验证。[1][2]'}}]})
        with patch.object(main, 'history', return_value=[]), patch.object(main, 'plan_question', AsyncMock(return_value=plan)), \
             patch.object(main, 'execute_plan', AsyncMock(return_value=deepcopy(self.primary))), \
             patch.object(main, 'provider_post', generation), patch.object(main, 'save_turn') as saved, \
             patch('backend.rag.adaptive_chat.provider_post', provider), patch('backend.rag.adaptive_chat.config.LLM_KEY', 'placeholder'):
            events = [event async for event in main.chat_events(request, self.r)]
        result = events[-1][1]
        self.assertEqual(events[-1][0], 'done')
        self.assertTrue(result['generated'])
        self.assertEqual(result['agentic']['tool_calls'], 1)
        self.assertTrue(any(name == 'stage' and value.get('stage') == 'evidence_check' for name, value in events))
        self.assertEqual(saved.call_args.args[3]['agentic'], result['agentic'])
        self.assertTrue(all(ref['number'] <= len(result['evidence']) for step in result['retrieval_steps'] for ref in step['references']))


if __name__ == '__main__':
    unittest.main()
