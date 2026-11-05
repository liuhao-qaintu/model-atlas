"""验证解释深度尊重用户要求，并确实进入真实生成请求。"""
import unittest
from unittest.mock import AsyncMock, patch

from auth_fixtures import TestClient

from backend import main
from backend.generation.answer_policy import VERSION, answer_profile
from backend.rag.planning import rule_plan


class AnswerPolicyTests(unittest.TestCase):
    def test_user_brief_request_overrides_why_or_multiple_questions(self):
        self.assertEqual(answer_profile('一句话解释为什么不能把卷积特征当概率？')['level'], 'concise')
        self.assertEqual(answer_profile('请简短回答：输入是什么？输出是什么？')['max_tokens'], 800)

    def test_negated_brevity_is_not_a_request_for_a_short_answer(self):
        self.assertEqual(answer_profile('不要只说结论，详细推导这个感受野')['level'], 'detailed')
        self.assertEqual(answer_profile('不要再简短回答，请逐层解释')['level'], 'detailed')

    def test_simple_fact_is_not_forced_into_a_long_tutorial(self):
        self.assertEqual(answer_profile('ResNet-18默认类别数是什么？')['level'], 'standard')
        self.assertEqual(answer_profile('CLIP是什么？', 'research')['max_tokens'], 1800)
        self.assertEqual(answer_profile('为什么冻结参数后BatchNorm还在变化？')['max_tokens'], 3200)

    def test_api_sends_requested_budget_without_sending_reference_answers(self):
        with TestClient(main.app) as client:
            question = '请详细解释ResNet-18的残差连接为什么不能保证梯度永不消失？'
            plan = rule_plan(question, main.retriever, 'resnet-18'); plan['duration_ms'] = 0
            section = next(s for s in main.retriever.sections if s['id'] == 'resnet-18:mechanism')
            evidence = {**section, 'section_id': section['id'], 'section_slug': section['slug'], 'cloud_allowed': True, 'sources': []}
            retrieval = {'evidence': [evidence], 'warnings': [], 'paths': [], 'steps': [], 'duration_ms': 0,
                         'stop_reason': '完成', 'channels': {}, 'reranked': False, 'remote_allowed': True}
            provider = AsyncMock(return_value={'choices': [{'message': {'content': '实际块还有ReLU门，因此不能保证梯度总是非零。[1]'}, 'finish_reason': 'stop'}]})
            with patch.object(main, 'plan_question', AsyncMock(return_value=plan)), patch.object(main, 'execute_plan', AsyncMock(return_value=retrieval)), \
                 patch.object(main, 'provider_post', provider), patch.object(main, 'history', return_value=[]), patch.object(main, 'save_turn'):
                result = client.post('/api/chat', json={'question': question, 'model_id': 'resnet-18'}).json()
            body = provider.await_args_list[0].args[3]
            self.assertEqual(body['max_tokens'], 3200)
            self.assertNotIn('约300字', body['messages'][0]['content'])
            self.assertIn(question, body['messages'][-1]['content'])
            self.assertEqual(result['generation_diagnostics']['answer_policy'], VERSION)
            self.assertEqual(result['generation_diagnostics']['answer_profile']['level'], 'detailed')
            self.assertTrue(result['generated'])


if __name__ == '__main__':
    unittest.main()
