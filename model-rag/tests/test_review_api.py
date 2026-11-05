"""隔离审核库：引文有效性、合并、占位阻断与刷新后的图谱可见性。"""
import copy
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from backend.api import review
from backend.infra import database
from backend.rag.graph import proven_edges


class ReviewApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
        database.Base.metadata.create_all(self.engine)
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(database, 'engine', self.engine))
        self.models = [{'id': 'document', 'name': 'Paper', 'aliases': []},
                       {'id': 'ae', 'name': 'Autoencoder', 'aliases': ['autoencoder']}]
        database.upsert_records(database.KnowledgeModel, self.models)
        database.upsert_records(database.KnowledgeSource, [{'id': 'source', 'kind': 'document', 'url': '/documents/paper'}])
        self.section = {'id': 'section', 'model_id': 'document', 'document_id': 'paper',
                        'title': 'Background', 'slug': 'background', 'locator': 'Background',
                        'source_ids': ['source'], 'cloud_allowed': False,
                        'content': 'Background introduction. ' * 20 + 'The baseline uses an autoencoder for feature learning.'}
        self.item = {'id': 'review', 'type': 'review_item', 'target_id': 'section', 'document_id': 'paper',
                     'status': 'pending', 'proposed': {'review_origin': 'assistant',
                     'classification': {'domain_path': ['时序与信号', '故障诊断'], 'section_role': '背景', 'knowledge_type': '模型'},
                     'entity_links': [{'entity': 'autoencoder', 'link': {'decision': 'link_existing', 'model_id': 'ae'}}]}}
        database.upsert_records(database.KnowledgeSection, [self.section])
        database.upsert_records(database.KnowledgeAnalysis, [self.item])
        app = FastAPI()
        app.include_router(review.router)
        self.client = self.stack.enter_context(TestClient(app))

    def tearDown(self):
        self.stack.close()
        self.engine.dispose()

    def approve(self):
        return self.client.post('/api/review/items/review/approve', json={})

    def save_item(self):
        database.upsert_records(database.KnowledgeAnalysis, [self.item])

    def test_entity_after_first_300_chars_gets_real_located_quote(self):
        self.assertEqual(self.approve().status_code, 200)
        section = database.records(database.KnowledgeSection)[0]
        edge = database.records(database.KnowledgeRelation)[0]
        self.assertEqual(section['section_role'], '背景')
        self.assertEqual(section['review_origin'], 'assistant')
        self.assertFalse(section['cloud_allowed'])
        self.assertEqual(edge['owner_model_id'], 'document')
        self.assertEqual((edge['source_name'], edge['target_name']), ('Paper', 'Autoencoder'))
        self.assertIn('autoencoder', edge['evidence'][0]['quote'])
        self.assertIn(edge['evidence'][0]['quote'], self.section['content'])
        self.assertEqual(self.approve().status_code, 409)

    def test_invalid_manual_quote_does_not_partially_approve(self):
        self.item['proposed']['entity_links'][0]['evidence_quote'] = 'This autoencoder sentence is not in the source.'
        self.save_item()
        self.assertEqual(self.approve().status_code, 422)
        self.assertEqual(review._get('review')['status'], 'pending')
        self.assertNotIn('review_status', database.records(database.KnowledgeSection)[0])
        self.assertEqual(database.records(database.KnowledgeRelation), [])

    def test_existing_but_irrelevant_quote_is_rejected(self):
        self.item['proposed']['entity_links'][0]['evidence_quote'] = self.section['content'][:80]
        self.save_item()
        self.assertEqual(self.approve().status_code, 422)

    def test_unknown_target_and_skipped_placeholder_are_rejected(self):
        self.item['proposed']['classification']['skipped'] = 'cloud_not_allowed'
        self.save_item()
        self.assertEqual(self.approve().status_code, 422)
        del self.item['proposed']['classification']['skipped']
        self.item['proposed']['entity_links'][0]['link']['model_id'] = 'missing'
        self.save_item()
        self.assertEqual(self.approve().status_code, 422)
        self.assertEqual(database.records(database.KnowledgeRelation), [])

    def test_ambiguous_entity_does_not_create_a_node_or_edge(self):
        self.item['proposed']['entity_links'][0]['link'] = {'decision': 'ambiguous'}
        self.save_item()
        self.assertEqual(self.approve().status_code, 200)
        self.assertEqual(database.records(database.KnowledgeRelation), [])
        self.assertEqual(len(database.records(database.KnowledgeModel)), 2)

    def test_two_sections_merge_and_refresh_makes_graph_visible(self):
        from backend import main
        stale = SimpleNamespace(models={m['id']: m for m in self.models}, sections=[self.section],
                                relations=[], sources={'source': database.records(database.KnowledgeSource)[0]})
        self.assertEqual(self.approve().status_code, 200)
        second_section = {**copy.deepcopy(self.section), 'id': 'section-2'}
        second_item = {**copy.deepcopy(self.item), 'id': 'review-2', 'target_id': 'section-2'}
        database.upsert_records(database.KnowledgeSection, [second_section])
        database.upsert_records(database.KnowledgeAnalysis, [second_item])
        self.assertEqual(self.client.post('/api/review/items/review-2/approve', json={}).status_code, 200)
        edges = database.records(database.KnowledgeRelation)
        self.assertEqual(len(edges), 1)
        self.assertEqual(len(edges[0]['evidence']), 2)
        self.assertEqual(proven_edges(stale), [])
        refreshed = SimpleNamespace(models=stale.models, sources=stale.sources,
                                    sections=database.records(database.KnowledgeSection), relations=edges)
        with patch.object(main, 'retriever', stale), patch('backend.rag.retrieval.Retriever', return_value=refreshed):
            self.assertEqual(self.client.post('/api/review/refresh', json={}).status_code, 200)
            self.assertEqual(len(proven_edges(main.retriever)), 1)

    def test_edit_and_reject_leave_graph_unmodified(self):
        edited = copy.deepcopy(self.item['proposed'])
        edited['classification']['section_role'] = '方法'
        self.assertEqual(self.client.post('/api/review/items/review/edit', json={'proposed': edited}).status_code, 200)
        self.assertNotIn('review_status', database.records(database.KnowledgeSection)[0])
        self.assertEqual(self.client.post('/api/review/items/review/reject', json={}).status_code, 200)
        self.assertEqual(database.records(database.KnowledgeRelation), [])

    def test_legacy_relation_without_names_does_not_break_query_planning(self):
        from backend.rag.planning import graph_names
        active = SimpleNamespace(models={m['id']: m for m in self.models},
                                 relations=[{'source': 'document', 'target': 'ae'}])
        self.assertEqual(graph_names(active)['ae'], 'Autoencoder')


if __name__ == '__main__':
    unittest.main()
