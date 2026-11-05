import unittest
from unittest.mock import AsyncMock, patch

from auth_fixtures import TestClient

from backend import main
from backend.rag.planning import rule_plan
from backend.generation.source_guard import inspect_strict_answer


class SourceGuardTests(unittest.TestCase):
    def test_rejects_observed_attach_and_mapping_recommendations(self):
        evidence = [{'content': 'def forward(self, x):\n    output = self.decoder(x)\n    return output'}]
        drafts = [
            '要得到词表概率，还需要额外在外部接入一个映射到词表维度的线性层并做 softmax。',
            '词表概率通常需要一个将 d_model 维映射到词表大小的线性层，再经过 softmax 等操作。',
        ]
        for draft in drafts:
            with self.subTest(draft=draft):
                result = inspect_strict_answer(draft, evidence, {'strict': True})
                self.assertEqual(result['status'], 'blocked')
                self.assertIn('unprovided_task_head_proposal', result['rules'])

    def test_hidden_linear_is_not_evidence_of_a_task_head(self):
        evidence = [{'content': 'self.feed_forward = nn.Linear(128, 128)\ndef forward(self, x): return self.feed_forward(x)'}]
        result = inspect_strict_answer('需要额外接入线性层映射到词表。', evidence, {'strict': True})
        self.assertEqual(result['status'], 'blocked')

    def test_negative_boundary_descriptions_remain_usable(self):
        evidence = [{'content': 'def forward(self, x): return self.decoder(x)'}]
        for draft in ['本题不需要额外接入线性层。', '不能推断需要一个词表投影头。',
                      '本轮资料不代表需要额外添加分类头。']:
            with self.subTest(draft=draft):
                self.assertEqual(inspect_strict_answer(draft, evidence, {'strict': True})['status'], 'no_pattern_detected')

    def test_rejects_proposed_missing_head_but_preserves_provided_head_or_normal_scope(self):
        answer = '要获得词表概率，需要额外添加一个线性层，再应用softmax。'
        base = [{'content': 'def forward(self, x):\n    return self.encoder(x)'}]
        self.assertEqual(inspect_strict_answer(answer, base, {'strict': True})['status'], 'blocked')
        self.assertEqual(inspect_strict_answer(answer, base, {})['status'], 'not_required')
        defined = [{'content': 'self.fc = nn.Linear(512, num_classes)\ndef forward(self,x): return self.fc(x)'}]
        self.assertEqual(inspect_strict_answer(answer, defined, {'strict': True})['status'], 'no_pattern_detected')

    def test_affirmative_feature_equivalence_is_blocked_but_negation_is_kept(self):
        evidence = [{'content': 'Conv1d returns (N,C_out,L_out) after convolution.'}]
        self.assertEqual(inspect_strict_answer('输出是未归一化分数logits。', evidence, {'strict': True})['status'], 'blocked')
        for answer in ['输出是特征，不能称为logits。', '本轮正文未提供分类头。', '本轮未提供后续分类实现。',
                       '本轮没有分类头定义，不能额外添加全连接层来说明本题。']:
            self.assertEqual(inspect_strict_answer(answer, evidence, {'strict': True})['status'], 'no_pattern_detected')

    def test_api_rejects_whole_generated_explanation_and_saves_actual_excerpt(self):
        with TestClient(main.app) as client:
            question = '只依据官方Conv1d接口说明输出。'
            plan = rule_plan(question, main.retriever, 'cnn-1d'); plan['duration_ms'] = 0
            plan['source_scope'] = {'strict': True, 'kind': 'implementation', 'label': '官方接口'}
            evidence = {'section_id': 'fixture:official', 'section_slug': 'official', 'model_id': 'cnn-1d',
                        'title': '官方函数', 'content': 'ORIGINAL_SOURCE\ndef forward(self,x):\n    return self._conv_forward(x)',
                        'cloud_allowed': True, 'sources': []}
            retrieval = {'evidence': [evidence], 'warnings': [], 'paths': [], 'steps': [], 'duration_ms': 0,
                         'stop_reason': '完成', 'channels': {}, 'reranked': False, 'remote_allowed': True}
            draft = '输出是特征。要获得类别概率，必须在其后添加全连接分类头。[1]'
            provider = AsyncMock(return_value={'choices': [{'message': {'content': draft}, 'finish_reason': 'stop'}]})
            with patch.object(main, 'plan_question', AsyncMock(return_value=plan)), \
                 patch.object(main, 'execute_plan', AsyncMock(return_value=retrieval)), \
                 patch.object(main, 'history', return_value=[]), patch.object(main, 'provider_post', provider), \
                 patch.object(main, 'save_turn') as save:
                result = client.post('/api/chat', json={'question': question, 'model_id': 'cnn-1d'}).json()
            self.assertFalse(result['generated'])
            self.assertEqual(result['generation_diagnostics']['rejected_answer'], draft)
            self.assertIn('ORIGINAL_SOURCE', result['answer'])
            self.assertNotIn('必须在其后添加全连接分类头', result['answer'])
            self.assertEqual(save.call_args.args[2], result['answer'])
            self.assertIn('未完成可采用', ' '.join(result['warnings']))
            provider.assert_awaited_once()
