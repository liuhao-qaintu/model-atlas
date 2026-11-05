"""隔离SQLite/上传目录：版本切换、旧引用、审核图谱与优先级，不访问模型。"""
import asyncio
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend import main
from backend.api import document_admin, review, ingest
from backend.infra.database import Base, KnowledgeModel, KnowledgeDocument, KnowledgeSection, KnowledgeAnalysis, KnowledgeRelation
from backend.infra.ingestion import preview_document, commit_document
from backend.rag.retrieval import Retriever
from backend.rag.graph import proven_edges


class DocumentVersionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.temp = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        self.engine = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
        Base.metadata.create_all(self.engine)
        for target in ('backend.infra.database.engine', 'backend.infra.ingestion.engine',
                       'backend.infra.document_versions.engine', 'backend.api.document_admin.engine'):
            self.stack.enter_context(patch(target, self.engine))
        for target in ('backend.infra.config.DATA_ROOT', 'backend.infra.public_sources.DATA_ROOT'):
            self.stack.enter_context(patch(target, self.root))
        for target in ('backend.infra.ingestion.UPLOAD_ROOT', 'backend.main.UPLOAD_ROOT'):
            self.stack.enter_context(patch(target, self.root / 'uploads'))
        self.stack.enter_context(patch.object(main, 'knowledge_lock', asyncio.Lock()))
        self.stack.enter_context(patch.object(main, 'retriever', None))
        with Session(self.engine) as db:
            for index, mid in enumerate(('demo', 'target')):
                db.add(KnowledgeModel(id=mid, payload={'id': mid, 'name': 'Target' if mid == 'target' else 'Demo',
                    'aliases': [mid], 'kind': '模型', 'description': 'test', 'ready': True, 'order': index}))
            db.commit()
        self.quote = 'This method uses the Target module to process input signals and return features.'
        self.raw = ('\ufeff# Version demo\n\n## Method\n\n' + self.quote + '\n\n## Results\n\nThe method returns features for the task.').encode('utf-8')
        self.document = preview_document('paper.md', self.raw, 'demo')
        commit_document(self.document['id'], 'Version demo', cloud_allowed=False)
        main.retriever = Retriever()
        self.old_section = next(s for s in main.retriever.sections if self.quote in s['content'])
        self.review_id = 'test-review'
        with Session(self.engine) as db:
            db.add(KnowledgeAnalysis(id=self.review_id, payload={'id': self.review_id, 'type': 'review_item',
                'status': 'pending', 'document_id': self.document['id'], 'target_id': self.old_section['id'], 'proposed': {}}))
            db.add(KnowledgeRelation(id='test-edge', payload={'id': 'test-edge', 'source': 'demo', 'target': 'target',
                'owner_model_id': 'demo', 'source_name': 'Demo', 'target_name': 'Target', 'relation': '提及',
                'status': 'accepted', 'origin': 'user', 'source_ids': self.old_section['source_ids'],
                'evidence': [{'section_id': self.old_section['id'], 'quote': self.quote}]}))
            db.commit()
        main.retriever = Retriever()
        app = FastAPI(); app.include_router(document_admin.router); app.include_router(review.router); app.include_router(ingest.router)
        app.add_api_route('/api/documents/{document_id}/import', main.import_document, methods=['POST'])
        app.add_api_route('/api/documents/{document_id}/file', main.original_file, methods=['GET'])
        app.add_api_route('/api/documents/{document_id}', main.document_detail, methods=['GET'])
        self.client = self.stack.enter_context(TestClient(app))

    def tearDown(self):
        self.stack.close(); self.engine.dispose()

    def preview(self):
        result = self.client.post(f"/api/documents/{self.document['id']}/reprocess/preview", json={})
        self.assertEqual(result.status_code, 200, result.text)
        return result.json()

    def confirm(self, document):
        return self.client.post(f"/api/documents/{document['id']}/import", json={'title': 'Version demo', 'cloud_allowed': False})

    def test_preview_is_local_and_leaves_current_knowledge_untouched(self):
        old_ids = {s['id'] for s in main.retriever.sections}
        result = self.preview()
        self.assertEqual(result['revision_of'], self.document['id'])
        self.assertEqual(result['revision_number'], 2)
        self.assertFalse(result['cloud_allowed'])
        self.assertEqual({s['id'] for s in main.retriever.sections}, old_ids)
        self.assertEqual(self.client.get('/api/documents/' + self.document['id']).json()['status'], 'imported')
        self.assertEqual((self.root / 'uploads' / (self.document['id'] + '.md')).read_bytes(), self.raw)

    def test_switch_preserves_old_links_and_requires_new_graph_review(self):
        self.assertEqual(len(proven_edges(main.retriever)), 1)
        updated = self.preview()
        response = self.confirm(updated)
        self.assertEqual(response.status_code, 200, response.text)
        old = self.client.get('/api/documents/' + self.document['id']).json()
        self.assertEqual(old['status'], 'archived'); self.assertEqual(old['superseded_by'], updated['id'])
        self.assertIn(self.old_section['id'], {s['id'] for s in old['sections']})
        self.assertEqual(self.client.get('/api/documents/' + self.document['id'] + '/file').content, self.raw)
        self.assertNotIn(self.old_section['id'], {s['id'] for s in main.retriever.sections})
        self.assertIn(self.old_section['id'], main.retriever.archived_section_ids)
        self.assertEqual([d['id'] for d in main.retriever.documents], [updated['id']])
        self.assertEqual(proven_edges(main.retriever), [])
        self.assertEqual(self.client.post('/api/review/items/test-review/approve', json={}).status_code, 409)
        self.assertEqual(self.client.post('/api/ingest/documents/' + self.document['id'], json={}).status_code, 409)
        section = next(s for s in main.retriever.sections if self.quote in s['content'])
        with Session(self.engine) as db:
            db.add(KnowledgeAnalysis(id='new-review', payload={'id': 'new-review', 'type': 'review_item', 'status': 'pending',
                'document_id': updated['id'], 'target_id': section['id'], 'proposed': {
                    'classification': {'section_role': '方法', 'entities': ['Target']},
                    'entity_links': [{'entity': 'Target', 'evidence_quote': self.quote, 'link': {'decision': 'link_existing', 'model_id': 'target'}}]}}))
            db.commit()
        result = self.client.post('/api/review/items/new-review/approve', json={})
        self.assertEqual(result.status_code, 200, result.text)
        edges = proven_edges(Retriever())
        self.assertEqual(len(edges), 1)
        self.assertEqual([e['section_id'] for e in edges[0]['evidence']], [section['id']])
        with Session(self.engine) as db:
            self.assertEqual(db.query(KnowledgeRelation).count(), 1)
            self.assertEqual(len(db.get(KnowledgeRelation, 'test-edge').payload['evidence']), 2)

    def test_competing_preview_cannot_overwrite_a_newer_version(self):
        first, second = self.preview(), self.preview()
        self.assertEqual(self.confirm(first).status_code, 200)
        self.assertEqual(self.confirm(second).status_code, 409)
        self.assertEqual(self.client.get('/api/documents/' + second['id']).json()['status'], 'preview')
        self.assertEqual([d['id'] for d in main.retriever.documents], [first['id']])

    def test_running_classification_blocks_commit(self):
        updated = self.preview()
        with patch('backend.infra.ingest_job.list_jobs', return_value=[{'status': 'running'}]):
            self.assertEqual(self.confirm(updated).status_code, 409)
        self.assertEqual(self.client.get('/api/documents/' + self.document['id']).json()['status'], 'imported')

    def test_priority_changes_metadata_without_reembedding_and_survives_reprocess(self):
        fingerprint = main.retriever.fingerprint
        original = self.client.get('/api/documents/' + self.document['id']).json()
        path = '/api/documents/' + self.document['id'] + '/priority'
        self.assertEqual(self.client.patch(path, json={'priority': 1.15}).status_code, 200)
        self.assertEqual(main.retriever.fingerprint, fingerprint)
        self.assertEqual(self.client.patch(path, json={'priority': 9}).status_code, 422)
        updated = self.preview(); self.assertEqual(self.confirm(updated).status_code, 200)
        current = self.client.get('/api/documents/' + updated['id']).json()
        self.assertEqual(current['retrieval_priority'], 1.15)
        self.assertEqual(current['retrieval_added_at'], original['retrieval_added_at'])
        self.assertEqual(self.client.patch(path, json={'priority': 1}).status_code, 409)
        from backend.infra.document_versions import active_document
        self.assertEqual(active_document(self.document['id'])['id'], updated['id'])
        self.assertTrue(preview_document('paper.md', self.raw, 'demo')['duplicate'])

    def test_missing_original_does_not_change_live_document(self):
        (self.root / 'uploads' / (self.document['id'] + '.md')).unlink()
        result = self.client.post('/api/documents/' + self.document['id'] + '/reprocess/preview', json={})
        self.assertEqual(result.status_code, 422)
        self.assertEqual(self.client.get('/api/documents/' + self.document['id']).json()['status'], 'imported')


if __name__ == '__main__': unittest.main()
