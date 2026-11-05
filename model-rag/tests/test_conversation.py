"""会话原话、连续追问和检索隔离的回归检查，不调用外部模型。"""
import unittest
import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from auth_fixtures import TestClient
from sqlalchemy import delete
from sqlalchemy.orm import Session

from backend import main
from backend.infra.conversation import conversation_reply, is_conversation_query
from backend.infra.database import ChatMessage, ChatSession, engine, history, save_turn, user_questions
from backend.rag.planning import explicit_models


class ConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(main.app)
        cls.client = cls.context.__enter__()
        cls.sessions = []

    @classmethod
    def tearDownClass(cls):
        with Session(engine) as db:
            db.execute(delete(ChatMessage).where(ChatMessage.session_id.in_(cls.sessions)))
            db.execute(delete(ChatSession).where(ChatSession.id.in_(cls.sessions)))
            db.commit()
        cls.context.__exit__(None, None, None)

    def session(self, questions=()):
        sid = str(uuid4())
        self.sessions.append(sid)
        for q in questions:
            save_turn(sid, q, '测试回答', {'model_id': 'resnet-18', 'plan': {'model_ids': ['resnet-18']}, 'evidence': []})
        return sid

    def ask(self, sid, q, streaming=False):
        return self.client.post('/api/chat/stream' if streaming else '/api/chat', json={'session_id': sid, 'question': q})

    def test_previous_question_is_latest_saved_user_not_seed_suggestion(self):
        sid = self.session(['ResNet-18 输出是什么？', '把分类头去掉之后呢？'])
        with patch('backend.main.plan_question', AsyncMock()) as planner, patch('backend.main.execute_plan', AsyncMock()) as retrieval, patch('backend.main.provider_post', AsyncMock()) as provider:
            result = self.ask(sid, '我问你的上一个问题是什么？').json()
        self.assertIn('把分类头去掉之后呢？', result['answer'])
        self.assertNotIn('ResNet-18 输出是什么？', result['answer'])
        self.assertEqual(result['response_kind'], 'conversation')
        self.assertEqual(result['evidence'], [])
        self.assertFalse(result['generated'])
        for dependency in (planner, retrieval, provider):
            dependency.assert_not_called()
        self.assertEqual(user_questions(sid)[-1], '我问你的上一个问题是什么？')

    def test_first_and_upper_previous_questions_use_exact_order(self):
        questions = [f'第{i}个测试问题' for i in range(1, 13)]
        sid = self.session(questions)
        self.assertEqual(user_questions(sid), questions)
        self.assertIn(questions[0], self.ask(sid, '我第一个问题是什么？').json()['answer'])
        # 回忆提问本身也是一次用户提问，因此上上条恰是原第 12 条。
        self.assertIn(questions[-1], self.ask(sid, '我上上个问题是什么？').json()['answer'])
        self.assertEqual([item['role'] for item in history(sid, 4)], ['user', 'assistant', 'user', 'assistant'])

    def test_new_session_does_not_read_another_session(self):
        self.session(['只属于另一个会话的原话'])
        result = self.ask(self.session(), '上一条问题是什么？').json()
        self.assertIn('没有上一条', result['answer'])
        self.assertNotIn('另一个会话的原话', result['answer'])

    def test_streaming_memory_uses_same_routing_and_persistence(self):
        sid = self.session(['这是本会话最近的原话'])
        response = self.ask(sid, '刚才的问题是什么？', True)
        self.assertIn('event: done', response.text)
        self.assertIn('这是本会话最近的原话', response.text)
        self.assertNotIn('event: delta', response.text)
        self.assertEqual(history(sid)[-1]['metadata']['response_kind'], 'conversation')

    def test_question_list_and_missing_ordinal(self):
        questions = ['CNN 输入是什么？', 'LSTM 状态是什么？']
        answer = conversation_reply('列出我之前问过的所有问题', questions)
        self.assertTrue(all(q in answer for q in questions))
        self.assertIn('没有可定位', conversation_reply('第9个问题是什么？', questions))
        for q in ['上一个问题中的维度为什么是512？', 'CNN 的输入是什么？', '继续解释刚才的模型结构', '把分类头去掉之后呢？']:
            self.assertFalse(is_conversation_query(q), q)

    def test_knowledge_followup_preserves_original_and_history_in_generation(self):
        sid = self.session(['ResNet-18 输出的 1000 和 512 分别是什么？'])
        captured = []
        async def fake(base, key, endpoint, body):
            captured.append(body['messages'])
            return {'choices': [{'message': {'content': '去掉分类头后为512维特征。[1]'}}]}
        with patch.object(main.retriever, 'vectors', None), patch('backend.infra.config.RERANK_KEY', ''), patch('backend.rag.planning.provider_post', AsyncMock(side_effect=RuntimeError('fallback'))), patch('backend.main.provider_post', fake):
            result = self.ask(sid, '它去掉分类头之后呢？').json()
        self.assertEqual(result['model_id'], 'resnet-18')
        self.assertTrue(result['plan']['followup'])
        self.assertTrue(result['generated'])
        self.assertTrue(any(m['content'] == 'ResNet-18 输出的 1000 和 512 分别是什么？' for m in captured[0]))
        self.assertEqual(json.loads(captured[0][-1]['content'])['用户当前原问题'], '它去掉分类头之后呢？')

    def test_owned_module_names_do_not_create_extra_comparison_model(self):
        cases = [
            ('比较 ResNet-18 中的 BasicBlock 与 CLIP 中的 CLIP文本Transformer 的残差相加', {'resnet-18', 'clip'}),
            ('比较 CLIP 中的 ViT视觉编码器 与 ViT 中的 TinyPatchViT 的分类头', {'clip', 'vit'}),
            ('比较 ViT 与 Transformer 的输入输出', {'vit', 'transformer'}),
        ]
        for q, expected in cases:
            self.assertEqual(set(explicit_models(q, main.retriever)), expected, q)

    def test_local_refresh_retains_documents_and_reuses_unchanged_vectors(self):
        previous = {d['id'] for d in main.retriever.documents}
        previous_models = set(main.retriever.models)
        result = self.client.post('/api/knowledge/refresh', json={})
        self.assertEqual(result.status_code, 200)
        self.assertEqual({d['id'] for d in main.retriever.documents}, previous)
        self.assertEqual(set(main.retriever.models), previous_models)
        self.assertEqual(result.json()['models'], len(previous_models))

    def test_teaching_alias_comparison_and_ambiguous_followup(self):
        from backend.rag.planning import rule_plan
        plan = rule_plan('比较 TinySignalCNN 和 TinyBiLSTM 的输入与分类输出', main.retriever)
        self.assertEqual(set(plan['model_ids']), {'cnn-1d', 'lstm'})
        previous = [{'role': 'assistant', 'metadata': {'plan': plan}}]
        self.assertTrue(rule_plan('它需要什么输入？', main.retriever, previous=previous)['clarification'])

    def test_memory_keeps_local_only_session_restriction(self):
        from backend.rag.planning import history_is_cloud_safe, plan_question
        import asyncio
        sid = self.session()
        save_turn(sid, '我的本地资料问题', '仅本地内容', {'model_id': 'resnet-18', 'evidence': [{'cloud_allowed': False}]})
        result = self.ask(sid, '我问你的上一个问题是什么？').json()
        self.assertFalse(result['cloud_allowed'])
        previous = history(sid)
        self.assertFalse(history_is_cloud_safe(previous))
        with patch('backend.rag.planning.provider_post', AsyncMock()) as provider:
            plan = asyncio.run(plan_question('它输出什么？', main.retriever, previous=previous))
        provider.assert_not_called()
        self.assertFalse(plan['remote_allowed'])


if __name__ == '__main__':
    unittest.main()
