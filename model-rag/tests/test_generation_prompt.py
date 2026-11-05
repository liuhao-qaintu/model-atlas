"""验证实际请求中的事实边界、历史隐私、引用保存与耗时诊断。"""
import hashlib
import json
import unittest
from unittest.mock import AsyncMock, patch

from auth_fixtures import TestClient

from backend import main
from backend.generation.answer_policy import answer_profile
from backend.generation.generation_prompt import VERSION, generation_messages, generation_system
from backend.rag.planning import rule_plan


class PromptBuilderTests(unittest.TestCase):
    def test_strict_scope_is_separate_from_history_and_keeps_full_evidence(self):
        messages = generation_messages('当前问句', '旧查询改写', 'UNIQUE_EVIDENCE\nreturn x', '已证实关系', [],
                                       answer_profile('当前问句'), 'learning', {'strict': True},
                                       [{'role': 'user', 'content': '前序问题'}, {'role': 'assistant', 'content': '前序答案'}])
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'user'])
        payload = json.loads(messages[-1]['content'])
        self.assertEqual(payload['用户当前原问题'], '当前问句')
        self.assertEqual(payload['资料正文'], 'UNIQUE_EVIDENCE\nreturn x')
        self.assertNotIn('前序答案', messages[0]['content'])
        self.assertNotIn('reference_answer', str(messages))
        self.assertGreater(len(messages[0]['content']), 0)

    def test_calculation_and_brevity_do_not_force_repeated_numerical_summaries(self):
        math = generation_system('手算一个卷积参数量', 'learning', answer_profile('手算一个卷积参数量'), {})
        brief = generation_system('一句话回答为什么输出是特征', 'learning', answer_profile('一句话回答为什么输出是特征'), {})
        self.assertNotIn('先给核心结论', math)
        self.assertIn('不在推导前另写数值摘要', math)
        self.assertIn('用户要求简短', brief)
        self.assertLess(len(math), 1600)


class PromptFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(main.app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_real_generation_uses_new_builder_and_excludes_private_history(self):
        question = '只依据已入库PyTorch 2.8.0官方Conv1d接口，输入输出是什么？'
        plan = rule_plan(question, main.retriever, 'cnn-1d'); plan['duration_ms'] = 0
        section = next(s for s in main.retriever.sections if s['id'] == 'cnn-1d:mechanism')
        evidence = {**section, 'section_id': section['id'], 'section_slug': section['slug'], 'cloud_allowed': True, 'sources': []}
        retrieval = {'evidence': [evidence], 'warnings': [], 'paths': [], 'steps': [], 'duration_ms': 0,
                     'stop_reason': '完成', 'channels': {}, 'reranked': False, 'remote_allowed': True}
        provider = AsyncMock(return_value={'choices': [{'message': {'content': '返回特征。[证据1]'}, 'finish_reason': 'stop'}]})
        with patch.object(main, 'plan_question', AsyncMock(return_value=plan)), \
             patch.object(main, 'execute_plan', AsyncMock(return_value=retrieval)), \
             patch.object(main, 'history', return_value=[{'role': 'user', 'content': 'PRIVATE_HISTORY_MARKER'}]), \
             patch.object(main, 'history_is_cloud_safe', return_value=False), \
             patch.object(main, 'provider_post', provider), patch.object(main, 'save_turn') as save:
            response = self.client.post('/api/chat', json={'question': question, 'model_id': 'cnn-1d'})
        self.assertEqual(response.status_code, 200)
        result = response.json(); body = provider.await_args.args[3]
        self.assertEqual(len(body['messages']), 2)
        self.assertNotIn('PRIVATE_HISTORY_MARKER', str(body))
        self.assertTrue(json.loads(body['messages'][-1]['content'])['来源范围']['strict'])
        self.assertEqual(body['messages'][0]['content'], generation_system(question, 'learning', answer_profile(question), plan['source_scope']))
        self.assertEqual(result['generation_diagnostics']['generation_prompt'], VERSION)
        self.assertEqual(result['generation_diagnostics']['saved_answer_sha256'], hashlib.sha256(result['answer'].encode()).hexdigest())
        self.assertEqual(result['answer'], '返回特征。[1]')
        self.assertEqual(save.call_args.args[2], result['answer'])
        self.assertIn('generation_ms', result['generation_diagnostics'])
        provider.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
