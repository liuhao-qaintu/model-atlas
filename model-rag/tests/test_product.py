import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.api import management
from backend.infra.database import Base, KnowledgeModel
from backend.api.evaluation import retrieval_metrics, aggregate
from backend.rag.graph import proven_edges
from backend.infra.knowledge import TOPICS, SECTION_NAMES, parse_sections
from backend.infra.config import CONTENT_ROOT
from backend.infra.learning_guides import GUIDES
from backend.rag.planning import rule_plan


class EvaluationMetricTests(unittest.TestCase):
    def test_metrics_deduplicate_and_use_labelled_denominators(self):
        result = retrieval_metrics(['x', 'x', 'y', 'a', 'z'], ['a', 'b'])
        self.assertIsNone(result['precision_at_5'])
        self.assertEqual(result['precision_lower_bound'], 0.2)
        self.assertEqual(result['unjudged'], 3)
        self.assertEqual(result['recall_at_5'], 0.5)
        self.assertEqual(result['reciprocal_rank'], 1 / 3)
        self.assertEqual(result['hit_at_5'], 1)

    def test_no_hit_and_empty_annotations_remain_unjudged(self):
        self.assertEqual(retrieval_metrics(['x'], ['a'])['reciprocal_rank'], 0)
        result = retrieval_metrics(['x'], [])
        self.assertIsNone(result['recall_at_5']); self.assertIsNone(result['precision_at_5'])

    def test_ungraded_answers_and_uncalibrated_confidence_are_unavailable(self):
        result = aggregate([{'metrics': retrieval_metrics(['x'], ['a'])}, {'error': 'timeout'}])
        self.assertIsNone(result['answer_accuracy']); self.assertIsNone(result['confidence'])
        self.assertEqual(result['evaluated'], 1); self.assertEqual(result['failed'], 1)

    def test_partial_answers_are_counted_separately(self):
        result = aggregate([{'grade': 'correct'}, {'grade': 'partial'}, {'grade': 'incorrect'}, {'grade': None}])
        self.assertEqual(result['answer_accuracy'], 1 / 3)
        self.assertEqual(result['partly_correct'], 1); self.assertEqual(result['graded_answers'], 3)


class DynamicModelTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
        Base.metadata.create_all(self.engine); self.mock = patch.object(management, 'engine', self.engine); self.mock.start()

    def tearDown(self): self.mock.stop(); self.engine.dispose()

    def test_create_edit_duplicate_and_whitespace(self):
        value = management.ModelInput(name='Custom Model', aliases=['custom'])
        model = management.store_model(value)
        self.assertIn('Custom Model', model['aliases']); self.assertFalse(model['ready'])
        updated = management.store_model(management.ModelInput(name='Custom v2'), model['id'])
        self.assertEqual(updated['id'], model['id']); self.assertEqual(updated['name'], 'Custom v2')
        self.assertIn('Custom Model', updated['aliases'])
        with self.assertRaises(ValueError): management.store_model(management.ModelInput(name='custom v2'))
        with self.assertRaises(ValueError): management.store_model(management.ModelInput(name=' '))

    def test_preinstalled_model_is_protected(self):
        with Session(self.engine) as db:
            db.add(KnowledgeModel(id='preset', payload={'name': 'Preset', 'order': 0})); db.commit()
        with self.assertRaises(ValueError): management.store_model(management.ModelInput(name='Changed'), 'preset')

    def context(self):
        model = management.store_model(management.ModelInput(name='Custom'))
        source = model['id']; quote = '本方法采用 ResNet-18 作为图像编码骨干，并读取其特征表示。'
        return source, quote, SimpleNamespace(models={source: model, 'resnet-18': {'name': 'ResNet-18', 'aliases': ['ResNet-18']}},
            sections=[{'id': 'proof', 'model_id': source, 'source_ids': ['D1'], 'content': quote, 'title': '原文', 'slug': 'p1', 'reader_url': '/documents/test?part=p1'}],
            sources={'D1': {'id': 'D1', 'kind': 'document', 'url': '/documents/test'}}, relations=[])

    def test_user_relation_requires_real_quote_and_is_visible_in_graph(self):
        source, quote, retriever = self.context()
        value = management.RelationInput(source=source, target='resnet-18', relation='采用架构', section_id='proof', quote=quote)
        result = management.store_relation(value, retriever); retriever.relations = [result]
        self.assertEqual(len(proven_edges(retriever)), 1)
        self.assertEqual(result['origin'], 'user')
        with self.assertRaises(ValueError): management.store_relation(value.model_copy(update={'quote': '本方法采用 ResNet-18，但这是一段编造的文字。'}), retriever)
        with self.assertRaises(ValueError): management.store_relation(value.model_copy(update={'source': 'resnet-18'}), retriever)

    def test_target_and_evidence_source_must_exist(self):
        source, quote, retriever = self.context()
        value = management.RelationInput(source=source, target='unknown', relation='采用架构', section_id='proof', quote=quote)
        with self.assertRaises(ValueError): management.store_relation(value, retriever)
        retriever.sections[0]['source_ids'] = []
        with self.assertRaises(ValueError): management.store_relation(value.model_copy(update={'target': 'resnet-18'}), retriever)


class LearningGuideTests(unittest.TestCase):
    def test_explicit_reference_to_previous_turn_is_a_followup_even_when_long(self):
        retriever = SimpleNamespace(models={'training': {'name': 'Training', 'aliases': ['Training']}}, sections=[], relations=[], sources={})
        retriever.identify_models = lambda query: ['training'] if 'Training' in query else []
        previous = [{'role': 'user', 'content': '训练步骤是什么？'}, {'role': 'assistant', 'metadata': {'model_id': 'training'}}]
        question = '请核对上一轮：' + '详细说明梯度累积与参数更新的条件，' * 6
        plan = rule_plan(question, retriever, previous=previous)
        self.assertTrue(plan['followup']); self.assertEqual(plan['model_id'], 'training')

    def test_introductory_models_have_conditional_guides(self):
        core={'cnn-1d','cnn-2d','resnet-18','autoencoder','lstm','transformer','vit','clip'}
        self.assertTrue(core.issubset(GUIDES))
        self.assertIn('16×5×5=400', GUIDES['cnn-2d'])
        self.assertIn('0.46', GUIDES['lstm'])

    def test_existing_lenet_configuration_matches_worked_example(self):
        card = (CONTENT_ROOT / '模型卡/03_2D_CNN_LeNet.md').read_text(encoding='utf-8')
        self.assertIn('padding=2', card); self.assertIn('nn.Conv2d(6, 16, kernel_size=5)', card)


if __name__ == '__main__': unittest.main()
