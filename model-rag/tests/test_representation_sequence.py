"""表示与序列知识的章节范围、输出用途、关系依据；不调用云端模型。"""
import asyncio
import unittest
from unittest.mock import patch

from auth_fixtures import TestClient

from backend import main
from backend.rag.graph import build_graph
from backend.rag.pipeline import execute_plan
from backend.rag.planning import rule_plan
from backend.infra.knowledge import TOPICS


class RepresentationSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(main.app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def setUp(self):
        self.retriever = main.retriever

    def search(self, question, model_id, section_id=None):
        with patch.object(self.retriever, 'vectors', None), patch('backend.infra.config.RERANK_KEY', ''):
            return asyncio.run(self.retriever.search(question, focus=model_id, section_id=section_id, strict_focus=True))

    def execute(self, question):
        plan = rule_plan(question, self.retriever)
        with patch.object(self.retriever, 'vectors', None), patch('backend.infra.config.RERANK_KEY', ''):
            result = asyncio.run(execute_plan(plan, self.retriever))
        return plan, result

    def test_foundation_topics_remain_ready_alongside_application_topics(self):
        models = self.client.get('/api/models').json()
        registered = {m['id']: m for m in models}
        self.assertTrue({t['id'] for t in TOPICS} <= set(registered))
        self.assertTrue(all(registered[t['id']]['ready'] for t in TOPICS))
        sources = {s['id']: s for s in self.client.get('/api/sources').json()}
        self.assertEqual(sources['E07']['url'], '/models/autoencoder?section=code')
        self.assertEqual(sources['E08']['url'], '/models/lstm?section=code')
        self.assertIn('/v2.6.0/', sources['S11']['url'])

    def test_autoencoder_preserves_latent_reconstruction_and_loss_sources(self):
        detail = self.client.get('/api/models/autoencoder').json()
        sections = {s['slug']: s for s in detail['sections']}
        self.assertEqual(sections['formula']['source_ids'], ['S10', 'S30'])
        self.assertIn('不是类别数', sections['io']['content'])
        self.assertIn('不是概率', sections['io']['content'])
        self.assertIn('没有分类头', sections['formula']['content'])
        result = self.search('Autoencoder 4 32 重构 潜在表示', 'autoencoder', 'autoencoder:io')
        self.assertTrue(any(e['section_id'] == 'autoencoder:io' for e in result['evidence']))

    def test_lstm_states_and_projection_explanation_are_retrievable(self):
        result = self.search('LSTM batch_first h_n c_n proj_size', 'lstm', 'lstm:io')
        io = next(e for e in result['evidence'] if e['section_id'] == 'lstm:io')
        self.assertIn('不影响 h_n/c_n', io['content'])
        self.assertIn('最后一维可以不同', io['content'])
        self.assertTrue(all(e['model_id'] == 'lstm' for e in result['evidence']))

    def test_variable_length_section_uses_actual_pack_source(self):
        result = self.search('LSTM lengths pack_padded_sequence padding', 'lstm', 'lstm:masks')
        proof = next(e for e in result['evidence'] if e['section_id'] == 'lstm:masks')
        self.assertEqual(proof['source_ids'], ['S28', 'E08'])
        self.assertIn('不同接口', proof['content'])
        self.assertEqual(proof['reader_url'], '/models/lstm?section=masks')

    def test_encoder_comparison_keeps_autoencoder_and_transformer_separate(self):
        plan, result = self.execute('Autoencoder 与 Transformer 编码器输入输出有什么不同？')
        self.assertEqual(plan['mode'], 'comparison')
        self.assertEqual({e['model_id'] for e in result['evidence']}, {'autoencoder', 'transformer'})
        self.assertLessEqual(len(result['evidence']), 5)
        self.assertTrue(all(step['status'] == 'found' for step in result['steps']))

    def test_shared_linear_operator_does_not_merge_reconstruction_and_classes(self):
        graph = build_graph(self.retriever, 'linear-map', 1)
        shared = next(s for s in graph['shared'] if s['node_id'] == 'linear-map')
        self.assertEqual({s['model_name'] for s in shared['implementations']}, {'Autoencoder', 'LSTM'})
        self.assertTrue(any('重构' in s['text'] for s in shared['implementations']))
        self.assertTrue(any('类别分数' in s['text'] for s in shared['implementations']))
        ae = build_graph(self.retriever, 'autoencoder', 2)
        self.assertNotIn('linear-head', {e['target'] for e in ae['edges']})
        self.assertNotIn('transformer-encoder-layer', {n['id'] for n in ae['nodes']})

    def test_autoencoder_relation_steps_keep_body_proofs(self):
        plan, result = self.execute('Autoencoder 中的 DenseDecoder 与线性映射有什么关系？')
        self.assertEqual(plan['mode'], 'relation')
        self.assertGreater(len(result['steps']), 1)
        self.assertLessEqual(len(result['steps']), 3)
        self.assertTrue(all(e['model_id'] == 'autoencoder' for e in result['evidence']))
        self.assertTrue(all(step['anchor']['proofs'] for step in result['steps'][1:]))

    def test_lstm_paths_explain_gate_state_and_classifier_with_their_sources(self):
        graph = build_graph(self.retriever, 'lstm', 2)
        targets = {n['id'] for n in graph['nodes']}
        self.assertTrue({'lstm-gates', 'recurrent-state', 'linear-head', 'packed-sequence'} <= targets)
        classifier = next(e for e in graph['edges'] if e['id'] == 'lstm-classifier')
        self.assertEqual(classifier['source_ids'], ['E08'])
        self.assertIn('整条序列', classifier['implementation'])
