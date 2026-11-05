"""新增资料的检索范围、章节来源和跨模型关系；不调用外部模型。"""
import asyncio
import unittest
from unittest.mock import patch

from auth_fixtures import TestClient

from backend import main
from backend.rag.graph import build_graph
from backend.rag.pipeline import execute_plan
from backend.rag.planning import rule_plan


class NewCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(main.app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_new_readers_include_named_code_and_specific_sources(self):
        cnn = self.client.get('/api/models/cnn-2d').json()
        transformer = self.client.get('/api/models/transformer').json()
        self.assertTrue(cnn['ready'] and transformer['ready'])
        cnn_sections = {s['slug']: s for s in cnn['sections']}
        seq_sections = {s['slug']: s for s in transformer['sections']}
        self.assertIn('class LeNetTutorial', cnn_sections['code']['content'])
        self.assertIn('class TinySeq2Seq', seq_sections['code']['content'])
        self.assertEqual(cnn_sections['formula']['source_ids'], ['S03'])
        self.assertEqual(seq_sections['formula']['source_ids'], ['S13'])
        self.assertEqual(seq_sections['masks']['source_ids'], ['S14', 'S23', 'E04'])
        self.assertNotEqual(seq_sections['io']['source_ids'], seq_sections['overview']['source_ids'])

    def test_comparison_has_distinct_evidence_for_new_and_existing_card(self):
        retriever = main.retriever
        plan = rule_plan('LeNet 与 ResNet-18 的输入和特征输出有什么不同？', retriever)
        with patch.object(retriever, 'vectors', None), patch('backend.infra.config.RERANK_KEY', ''):
            result = asyncio.run(execute_plan(plan, retriever))
        self.assertEqual(plan['mode'], 'comparison')
        self.assertEqual({e['model_id'] for e in result['evidence']}, {'cnn-2d', 'resnet-18'})
        self.assertTrue(all(step['status'] == 'found' for step in result['steps']))
        self.assertLessEqual(len(result['evidence']), 5)

    def test_mask_question_retrieves_actual_transformer_mask_section(self):
        retriever = main.retriever
        with patch.object(retriever, 'vectors', None), patch('backend.infra.config.RERANK_KEY', ''):
            result = asyncio.run(retriever.search('Transformer tgt_mask True padding mask', focus='transformer', section_id='transformer:masks', strict_focus=True))
        self.assertTrue(all(e['model_id'] == 'transformer' for e in result['evidence']))
        proof = next(e for e in result['evidence'] if e['section_id'] == 'transformer:masks')
        self.assertIn('True 表示禁止注意', proof['content'])
        self.assertEqual(proof['reader_url'], '/models/transformer?section=masks')

    def test_transformer_layer_relation_queries_keep_the_owner(self):
        retriever = main.retriever
        question = 'Transformer 中的 TransformerEncoderLayer 与多头注意力、残差相加有什么联系？'
        plan = rule_plan(question, retriever)
        with patch.object(retriever, 'vectors', None), patch('backend.infra.config.RERANK_KEY', ''):
            result = asyncio.run(execute_plan(plan, retriever))
        self.assertEqual(plan['model_ids'], ['transformer'])
        self.assertEqual(plan['mode'], 'relation')
        self.assertGreater(len(result['steps']), 1)
        self.assertTrue(all(e['model_id'] == 'transformer' for e in result['evidence']))
        self.assertLessEqual(len(result['steps']), 3)
        self.assertTrue(all(step['anchor']['proofs'] for step in result['steps'][1:]))

    def test_structural_comparison_has_sources_from_different_owners(self):
        graph = build_graph(main.retriever, 'residual')
        shared = next(item for item in graph['shared'] if item['node_id'] == 'residual')
        self.assertEqual({item['model_name'] for item in shared['implementations']}, {'ResNet-18', 'Transformer', 'ViT', 'CLIP'})
        cnn = build_graph(main.retriever, 'conv2d')
        conv = next(item for item in cnn['shared'] if item['node_id'] == 'conv2d')
        self.assertEqual({item['model_name'] for item in conv['implementations']}, {'ResNet-18', '2D CNN / LeNet', 'ViT', 'CLIP'})

    def test_operator_alias_inside_named_model_does_not_change_scope(self):
        retriever = main.retriever
        question = 'ResNet-18 中的 BasicBlock 包含算子 二维卷积，具体如何实现？请解释输入输出和这条关系的依据。'
        plan = rule_plan(question, retriever, 'resnet-18', 'resnet-18:overview')
        self.assertEqual(plan['model_ids'], ['resnet-18'])
        self.assertEqual(plan['mode'], 'relation')
        self.assertEqual(plan['section_id'], 'resnet-18:overview')
        compared = rule_plan('1D CNN 和 Conv2d 的输入有什么不同？', retriever)
        self.assertEqual(set(compared['model_ids']), {'cnn-1d', 'cnn-2d'})

    def test_vocab_projection_is_not_recorded_as_sequence_classification(self):
        graph = build_graph(main.retriever, 'tiny-seq2seq', 1)
        targets = {edge['target'] for edge in graph['edges'] if edge['source'] == 'tiny-seq2seq'}
        self.assertIn('vocabulary-projection', targets)
        self.assertNotIn('linear-head', targets)
        self.assertNotIn('linear-head', {item['node_id'] for item in graph['shared']})
